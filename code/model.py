import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
import gc
from attention import SelfAttention, CrossAttention

class BasicSD(torch.nn.Module):
    '''
    Currently set for inference only.
    '''
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16):
        super().__init__()

        self.torch_dtype = torch_dtype

        if model_id == "lambdalabs/miniSD-diffusers":
            # 1. Load the autoencoder model which will be used to decode the latents into image space.
            self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=self.torch_dtype) 

            # 2. Load the tokenizer and text encoder to tokenize and encode the text.
            self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", torch_dtype=self.torch_dtype)
            self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=self.torch_dtype)

            # 3. The UNet model for generating the latents.
            self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=self.torch_dtype)
        else:
            # 1. Load the autoencoder model which will be used to decode the latents into image space.
            self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=self.torch_dtype, use_safetensors=True) 

            # 2. Load the tokenizer and text encoder to tokenize and encode the text.
            self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", torch_dtype=self.torch_dtype, use_safetensors=True)
            self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=self.torch_dtype, use_safetensors=True)

            # 3. The UNet model for generating the latents.
            self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=self.torch_dtype, use_safetensors=True)
    
    def forward(self, scheduler, prompt, guidance_scale=7.5, gen_seed=None, torch_device='cuda'):
        generator = torch.Generator()
        if gen_seed is None:
            generator.seed()
        else:
            generator.manual_seed(gen_seed)
        batch_size = len(prompt)

        # First, we get the text_embeddings for the prompt. These embeddings will be used to condition the UNet model.
        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids.to(torch_device))[0]
        
        # We'll also get the unconditional text embeddings for classifier-free guidance, 
        # which are just the embeddings for the padding token (empty text). They need to 
        # have the same shape as the conditional `text_embeddings` (`batch_size` and `seq_length`)
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(torch_device))[0]

        # For classifier-free guidance, we need to do two forward passes. 
        # One with the conditioned input (`text_embeddings`), and another with the unconditional embeddings (`uncond_embeddings`). 
        # In practice, we can concatenate both into a single batch to avoid doing two forward passes.
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Generate the initial random noise
        height = 256
        width = 256
        latents = torch.randn(
        (batch_size, self.unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        dtype=self.torch_dtype,
        )
        latents = latents * scheduler.init_noise_sigma
        latents = latents.to(torch_device)

        for t in scheduler.timesteps:
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        
        return image

class Adapter(torch.nn.Module):
    def __init__(self, torch_dtype=torch.float16):
        super().__init__()

        # Reducing feature dimension first
        self.linear = torch.nn.Linear(in_features=4800, out_features=768)
        # Resizing sequence length
        self.conv = torch.nn.Conv1d(in_channels=768, out_channels=768, kernel_size=3, padding=1, stride=2) # stride to adjust sequence length
        self.adaptive_pool = torch.nn.AdaptiveAvgPool1d(77)
        self.norm = torch.nn.LayerNorm(768)

        if torch_dtype == torch.float16:
            self.linear.weight.data = self.linear.weight.data.half()
            self.linear.bias.data = self.linear.bias.data.half()
            self.conv.weight.data = self.conv.weight.data.half()
            self.conv.bias.data = self.conv.bias.data.half()
            self.norm.weight.data = self.norm.weight.data.half()
            self.norm.bias.data = self.norm.bias.data.half()

    def forward(self, input):
        # input expected shape: batch_size x seq_length x features (batch_size x 714 x 4800)
        # Reshape for linear
        input = self.linear(input)
        input = input.permute(0, 2, 1)
        # Convolve to adjust sequence length
        input = self.conv(input)
        input = self.adaptive_pool(input)
        input = input.permute(0, 2, 1)
        # Normalize
        input = self.norm(input)
        return input

class FusionModule(torch.nn.Module):
    def __init__(self, torch_dtype=torch.float16, model_type='concatenation'):
        super().__init__()
        self.model_type = model_type
        if self.model_type == 'concatenation':
            self.fusion = torch.nn.Linear(1536, 768)  # If using concatenation
            self.norm = torch.nn.LayerNorm(768)
        elif self.model_type == 'cross attention':
            self.fusion = CrossAttention(8, 768, 768, torch_dtype=torch_dtype)
            self.norm1 = torch.nn.LayerNorm(768)
            self.ffn1 = torch.nn.Linear(768, 768)
            self.norm2 = torch.nn.LayerNorm(768)
            self.relu = torch.nn.ReLU()
        elif self.model_type == 'linear layer':
            self.fusion = torch.nn.Linear(768, 768)  # If using concatenation
            self.norm = torch.nn.LayerNorm(768)

        if torch_dtype == torch.float16:
            if self.model_type == 'concatenation' or self.model_type == 'linear layer':
                self.fusion.weight.data = self.fusion.weight.data.half()
                self.fusion.bias.data = self.fusion.bias.data.half()
                self.norm.weight.data = self.norm.weight.data.half()
                self.norm.bias.data = self.norm.bias.data.half()
            elif self.model_type == 'cross attention':
                self.norm1.weight.data = self.norm1.weight.data.half()
                self.norm1.bias.data = self.norm1.bias.data.half()
                self.ffn1.weight.data = self.ffn1.weight.data.half()
                self.ffn1.bias.data = self.ffn1.bias.data.half()
                self.norm2.weight.data = self.norm2.weight.data.half()
                self.norm2.bias.data = self.norm2.bias.data.half()

    def forward(self, text_emb, audio_emb=None):
        if self.model_type == 'concatenation':
            # Concatenate the embeddings
            combined = torch.cat([text_emb, audio_emb], dim=-1)
            # Reduce the dimensionality
            combined = self.fusion(combined)
            # Normalize
            x2 = self.norm(combined)
        elif self.model_type == 'cross attention':
            # Reduce the dimensionality
            combined = self.fusion(text_emb, audio_emb)
            # Normalize
            x1 = self.norm1(combined)
            # # Send it through the feedforward network
            x2 = self.ffn1(x1)
            # Activation function
            x2 = self.relu(x2)
            # Residual addition
            x2 = x1 + x2
        elif self.model_type == 'linear layer':
            # Reduce the dimensionality
            combined = self.fusion(text_emb)
            # Normalize
            x2 = self.norm(combined)

        return x2

class AudioSD(torch.nn.Module):
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, model_type='concatenation'):
        super().__init__()
        
        self.torch_dtype = torch_dtype
        self.model_type = model_type

        if model_id == "lambdalabs/miniSD-diffusers" or "OFA-Sys/small-stable-diffusion-v0":
            # 1. Load the autoencoder model which will be used to decode the latents into image space.
            self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=self.torch_dtype) 

            # 3. The UNet model for generating the latents.
            self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=self.torch_dtype)
        else:
            # 1. Load the autoencoder model which will be used to decode the latents into image space.
            self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=self.torch_dtype, use_safetensors=True) 

            # 3. The UNet model for generating the latents.
            self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=self.torch_dtype, use_safetensors=True)

        # 4. Freeze the parameters for the original SD model.
        for param in self.vae.parameters():
            param.requires_grad = False
        
        for param in self.unet.parameters():
            param.requires_grad = False
        
        # 5. Define the Adapter and FusionModule
        if self.model_type != 'linear layer':
            self.adapter = Adapter(self.torch_dtype)

        self.fusion_module = FusionModule(self.torch_dtype, self.model_type)
    
    def forward(self, scheduler, text_embeddings, audio_embeddings, torch_device='cuda'):
        # Obtain the batch size
        batch_size = audio_embeddings.shape[0]
        
        # Obtain the adapter and fused embeddings
        if self.model_type != 'linear layer':
            adapter_embeddings = self.adapter(audio_embeddings)
            fused_embeddings = self.fusion_module(text_embeddings, adapter_embeddings)
            del text_embeddings, adapter_embeddings
        else:
            fused_embeddings = self.fusion_module(text_embeddings)
            del text_embeddings

        # Setup the generator for the latent noise
        generator = torch.Generator()
        generator.seed()
        
        # Generate the initial random noise
        height = 256
        width = 256
        latents = torch.randn(
        (batch_size, self.unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        dtype=self.torch_dtype,
        )
        latents = latents * scheduler.init_noise_sigma
        latents = latents.to(torch_device)

        # These commands help you when you face CUDA OOM error
        gc.collect()
        torch.cuda.empty_cache()

        # Forward pass
        for t in scheduler.timesteps:
            latent_model_input = latents

            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=fused_embeddings).sample

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            del latent_model_input, noise_pred
            # These commands help you when you face CUDA OOM error
            gc.collect()
            torch.cuda.empty_cache()

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        
        return image