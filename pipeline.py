import torch
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler, PNDMScheduler, DDPMScheduler, DDIMScheduler
from tqdm import tqdm
import os
import argparse

class SDPipeline():
    def __init__(self,
                model_id="runwayml/stable-diffusion-v1-5",
                num_inference_steps=50,
                sampler="ddpm",
                torch_dtype=torch.float32,
                output_dir=None):

        # Determine the data type for all the torch tensors
        self.torch_dtype = torch_dtype

        # Set up the sampler
        if sampler == "ddpm":
            self.scheduler = DDPMScheduler.from_config(model_id, subfolder="scheduler", torch_dtype=self.torch_dtype)
        elif sampler == "dpmsolver++":
            self.scheduler = DPMSolverMultistepScheduler.from_config(model_id, subfolder="scheduler", torch_dtype=self.torch_dtype)
        elif sampler == "pndm":
            self.scheduler = PNDMScheduler.from_config(model_id, subfolder="scheduler", torch_dtype=self.torch_dtype)
        elif sampler == "ddim":
            self.scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler", torch_dtype=self.torch_dtype)
        else:
            print("Please use either ddpm, dpmsolver++, pndm, or ddim as one of the samplers")
            return
        
        # Can only run SD 1.5 and SD 2.1 at the moment
        if model_id != "runwayml/stable-diffusion-v1-5" and model_id != "stabilityai/stable-diffusion-2-1":
            print("Please use runwayml/stable-diffusion-v1-5 or stabilityai/stable-diffusion-2-1 for model_id")
            return
        
        # Set up the number of inference steps
        self.scheduler.set_timesteps(num_inference_steps)

        # Determine the torch device
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.torch_dtype == torch.float16:
            # 1. Load the autoencoder model which will be used to decode the latents into image space.
            self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=self.torch_dtype, use_safetensors=True, variant="fp16") 

            # 2. Load the tokenizer and text encoder to tokenize and encode the text.
            self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", torch_dtype=self.torch_dtype, use_safetensors=True, variant="fp16")
            self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=self.torch_dtype, use_safetensors=True, variant="fp16")

            # 3. The UNet model for generating the latents.
            self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=self.torch_dtype, use_safetensors=True, variant="fp16")
        else:
            # 1. Load the autoencoder model which will be used to decode the latents into image space.
            self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=self.torch_dtype, use_safetensors=True) 

            # 2. Load the tokenizer and text encoder to tokenize and encode the text.
            self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", torch_dtype=self.torch_dtype, use_safetensors=True)
            self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=self.torch_dtype, use_safetensors=True)

            # 3. The UNet model for generating the latents.
            self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=self.torch_dtype, use_safetensors=True)

        # Move the models to the GPU
        self.vae = self.vae.to(self.torch_device)
        self.text_encoder = self.text_encoder.to(self.torch_device)
        self.unet = self.unet.to(self.torch_device)

        # Declare the output directory for the images
        if output_dir is None:
            self.output_dir = "output_imgs"
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def inference(self,
                album_ids=[],
                prompt=[],
                guidance_scale=7.5, # min of 1 and max of 14
                gen_seed=42):

        if len(prompt) == 0 or len(album_ids) == 0:
            print("Please send a prompt and album ID in")
            return
        batch_size = len(prompt)
        
        height = 512
        width = 512
        generator = torch.Generator()
        if gen_seed is None:
            generator.seed()
        else:
            generator.manual_seed(gen_seed)
        batch_size = len(prompt)

        # First, we get the text_embeddings for the prompt. These embeddings will be used to condition the UNet model.
        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.torch_device))[0]
        
        # We'll also get the unconditional text embeddings for classifier-free guidance, 
        # which are just the embeddings for the padding token (empty text). They need to 
        # have the same shape as the conditional `text_embeddings` (`batch_size` and `seq_length`)
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.torch_device))[0]

        # For classifier-free guidance, we need to do two forward passes. 
        # One with the conditioned input (`text_embeddings`), and another with the unconditional embeddings (`uncond_embeddings`). 
        # In practice, we can concatenate both into a single batch to avoid doing two forward passes.
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # Generate the initial random noise
        height = 512
        width = 512
        latents = torch.randn(
        (batch_size, self.unet.in_channels, height // 8, width // 8),
        generator=generator,
        dtype=self.torch_dtype,
        )
        latents = latents * self.scheduler.init_noise_sigma
        latents = latents.to(self.torch_device)

        for t in tqdm(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            image = self.vae.decode(latents).sample
        
        # Rescale the images
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        # Save the images
        for i in range(batch_size):
            pil_images[i].save(os.path.join(self.output_dir, f"{album_ids[i]}.png"))

    