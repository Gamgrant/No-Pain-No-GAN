import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import DPMSolverMultistepScheduler, PNDMScheduler, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
import os
from model import BasicSD, AudioSD
import gc
from transformers import CLIPTextModel, CLIPTokenizer
from torch.profiler import profile, record_function, ProfilerActivity

class SDPipeline():
    def __init__(self,
                model_id="runwayml/stable-diffusion-v1-5",
                num_inference_steps=50,
                sampler="ddpm",
                torch_dtype=torch.float16,
                output_dir=None,
                save_images=False):

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
        # if model_id != "runwayml/stable-diffusion-v1-5" and model_id != "stabilityai/stable-diffusion-2-1":
        #     print("Please use runwayml/stable-diffusion-v1-5 or stabilityai/stable-diffusion-2-1 for model_id")
        #     return
        
        # Set up the number of inference steps
        self.scheduler.set_timesteps(num_inference_steps)

        # Determine the torch device
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Define the model
        self.model = BasicSD(model_id, torch_dtype)

        # Define the criterion
        self.criterion = torch.nn.MSELoss()

        if self.torch_dtype == torch.float16:
            torch_name = 'fp16'
        else:
            torch_name = 'fp32'

        # Move the model to the GPU
        self.model = self.model.to(self.torch_device)

        # Declare the output directory for the images
        self.save_images = save_images
        if self.save_images:
            if output_dir is None:
                s = model_id.split("/")[-1]
                self.output_dir = f"{s}_{num_inference_steps}_{sampler}_{torch_name}"
            else:
                self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)

    def eval(self,
            eval_loader=None,
            guidance_scale=7.5, # min of 1 and max of 14
            gen_seed=None,
            ):

        if eval_loader is None:
            print("Please send a dataloader in")
            return

        # Set the model evaluation mode
        self.model.eval()

        # Declare the total loss and batch bar
        total_loss = 0.0
        batch_bar = tqdm(total=len(eval_loader), dynamic_ncols=True, position=0, leave=False, desc='Inference')
        i = 0

        for prompts, gt_images, album_ids in eval_loader:
            gt_images = gt_images.to(self.torch_device)

            # Perform the forward pass
            with torch.no_grad():
                inf_images = self.model.forward(self.scheduler, prompts, guidance_scale, gen_seed, self.torch_device)

            loss = self.criterion(gt_images, inf_images)
            total_loss += loss.item()

            if self.save_images:
                # Rescale the images
                rescaled_images = (inf_images / 2 + 0.5).clamp(0, 1)
                rescaled_images = rescaled_images.detach().cpu().permute(0, 2, 3, 1).numpy()
                rescaled_images = (rescaled_images * 255).round().astype("uint8")
                pil_images = [Image.fromarray(rescaled_image) for rescaled_image in rescaled_images]

                # Save the images
                for i in range(len(prompts)):
                    pil_images[i].save(os.path.join(self.output_dir, f"{album_ids[i]}.png"))
                
                del rescaled_images, pil_images

            batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))))
            i += 1

            batch_bar.update() # Update tqdm bar

            del gt_images, inf_images, prompts
        
        batch_bar.close()
        total_loss = total_loss / len(eval_loader)
        
        return total_loss

class AudioSDPipeline():
    def __init__(self,
                model_id="runwayml/stable-diffusion-v1-5",
                lr_num_training_steps=None,
                sampler="ddpm",
                torch_dtype=torch.float16,
                pretrained_ckpt=None,
                mode='train',
                model_type=''):

        # Define the model_id, sampler, and data type for all the torch tensors
        self.model_id = model_id
        self.sampler = sampler
        self.torch_dtype = torch_dtype

        # Set up the sampler
        if self.sampler == "ddpm":
            self.scheduler = DDPMScheduler.from_config(self.model_id, subfolder="scheduler", torch_dtype=self.torch_dtype)
        elif self.sampler == "dpmsolver++":
            self.scheduler = DPMSolverMultistepScheduler.from_config(self.model_id, subfolder="scheduler", torch_dtype=self.torch_dtype)
        elif self.sampler == "pndm":
            self.scheduler = PNDMScheduler.from_config(self.model_id, subfolder="scheduler", torch_dtype=self.torch_dtype)
        elif self.sampler == "ddim":
            self.scheduler = DDIMScheduler.from_config(self.model_id, subfolder="scheduler", torch_dtype=self.torch_dtype)
        else:
            print("Please use either ddpm, dpmsolver++, pndm, or ddim as one of the samplers")
            return

        # Determine the torch device
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Determine the model type
        self.model = AudioSD(model_id, self.torch_dtype, model_type)

        # Move the models to the GPU
        self.model = self.model.to(self.torch_device)
        
        if pretrained_ckpt is not None:
            ckpt = torch.load(pretrained_ckpt)
            self.model.load_state_dict(ckpt['model_state_dict'])
        
        if mode == 'train':
            # Determine the criterion
            self.criterion = torch.nn.MSELoss()

            # Determine the optimizer
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4)
            if pretrained_ckpt is not None:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            # Determine the lr scheduler
            self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer, 
                                                                num_warmup_steps=500, 
                                                                num_training_steps = lr_num_training_steps)
            if pretrained_ckpt is not None:                                                   
                self.lr_scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        # 2. Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", torch_dtype=self.torch_dtype)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=self.torch_dtype)
        self.text_encoder = self.text_encoder.to(self.torch_device)

    def train(self, 
              train_loader=None, 
              max_timesteps=50):        
        if train_loader is None:
            print("Please ensure the album dataloader is being sent in")
            return
        
        # Set the model to train mode
        self.model.train()

        # Set up the total loss variable and batch bar
        total_loss = 0.0
        batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, position=0, leave=False, desc='Train')
        i = 0

        # Train
        # These commands help you when you face CUDA OOM error
        gc.collect()
        torch.cuda.empty_cache()
        train_losses = []
        step = 0
        # Uncomment for profiler
        # with torch.profiler.profile(
        #         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/model'),
        #         record_shapes=True,
        #         profile_memory=True,
        #         with_stack=True
        # ) as prof:
        for prompts, gt_images, album_ids, audio_embeddings in train_loader:
            # Uncomment for profiler
            # prof.step()
            # if step >= 1 + 1 + 3:
            #     break
            self.optimizer.zero_grad()

            # First, we get the text_embeddings for the prompt. These embeddings will be used to condition the UNet model.
            text_input = self.tokenizer(prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            with torch.no_grad():
                text_embeddings = self.text_encoder(text_input.input_ids.to(self.torch_device))[0]
            del text_input

            # Send the ground truth images and audio_embeddings to device
            gt_images = gt_images.to(self.torch_dtype)
            gt_images = gt_images.to(self.torch_device)
            audio_embeddings = audio_embeddings.to(self.torch_device).to(self.torch_dtype)
            
            # Obtain a random number for the number of timesteps
            timesteps = torch.randint(1, max_timesteps, size=(1,), dtype=torch.int16).item()

            # Set the timesteps
            self.scheduler.set_timesteps(timesteps)

            # Perform the forward pass
            pred_images = self.model.forward(self.scheduler, text_embeddings, audio_embeddings, self.torch_device)
            loss = self.criterion(gt_images, pred_images)
            loss.backward()
            
            # Uncomment for profiler
            # Display the profiled memory and GPU usage
            # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

            # Calculate the total loss and update the batch bar
            total_loss += loss.item()
            train_losses.append(total_loss / (i + 1))
            if np.isnan(total_loss):
                print("Loss is NaN, exiting training.")
                exit()

            del gt_images, pred_images, audio_embeddings, prompts, loss, timesteps

            # Update the parameters and update the learning rate
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.5)
            self.optimizer.step()
            self.lr_scheduler.step()

            batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))))
            i += 1
            batch_bar.update() # Update tqdm bar

            # These commands help you when you face CUDA OOM error
            gc.collect()
            torch.cuda.empty_cache()
            # step += 1
        
        batch_bar.close()
        total_loss = total_loss / len(train_loader)

        return total_loss, train_losses

    def eval(self,
            eval_loader=None,
            num_timesteps=50,
            save_images=False,
            epoch=None,
            run_name=''):

        if eval_loader is None:
            print("Please ensure the album dataloader is being sent in")
            return

        # Set the model evaluation mode
        self.model.eval()

        # Set up the number of inference steps
        self.scheduler.set_timesteps(num_timesteps)

        # Set up the total loss variable and batch bar
        total_loss = 0.0
        batch_bar = tqdm(total=len(eval_loader), dynamic_ncols=True, position=0, leave=False, desc='Evaluation')
        i = 0

        # Evaluation
        eval_losses = []
        for prompts, gt_images, album_ids, audio_embeddings in eval_loader:
            gt_images = gt_images.to(self.torch_dtype)
            gt_images = gt_images.to(self.torch_device)
            audio_embeddings = audio_embeddings.to(self.torch_device)

            # First, we get the text_embeddings for the prompt. These embeddings will be used to condition the UNet model.
            text_input = self.tokenizer(prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            with torch.no_grad():
                text_embeddings = self.text_encoder(text_input.input_ids.to(self.torch_device))[0]

            # Perform the forward pass
            with torch.no_grad():
                inf_images = self.model.forward(self.scheduler, text_embeddings, audio_embeddings, self.torch_device)

            loss = self.criterion(gt_images, inf_images)
            total_loss += loss.item()
            eval_losses.append(total_loss / (i + 1))
            if save_images:
                s = self.model_id.split("/")[-1]
                if run_name != '':
                    output_dir = f"images/{s}_{self.sampler}_{run_name}_epoch_{epoch}"
                    os.makedirs(output_dir, exist_ok=True)
                else:
                    output_dir = f"images/{s}_{self.sampler}_epoch_{epoch}"
                    os.makedirs(output_dir, exist_ok=True)
                # Rescale the images
                rescaled_images = (inf_images / 2 + 0.5).clamp(0, 1)
                rescaled_images = rescaled_images.detach().cpu().permute(0, 2, 3, 1).numpy()
                rescaled_images = (rescaled_images * 255).round().astype("uint8")
                pil_images = [Image.fromarray(rescaled_image) for rescaled_image in rescaled_images]

                # Save the images
                for j in range(len(prompts)):
                    pil_images[j].save(os.path.join(output_dir, f"{album_ids[j]}.png"))
                
                del rescaled_images, pil_images

            batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))))
            i += 1

            batch_bar.update() # Update tqdm bar

            del gt_images, inf_images, audio_embeddings, prompts
        
        batch_bar.close()
        total_loss = total_loss / len(eval_loader)
        
        return total_loss, eval_losses
    
    def inference(self, dataloader=None, num_timesteps = 50, output_dir="inference_imgs/test"):
        if dataloader is None:
            print("Please ensure the album dataloader is being sent in")
            return

        # Set the model evaluation mode
        self.model.eval()

        # Set up the number of inference steps
        self.scheduler.set_timesteps(num_timesteps)

        # Inference
        for prompts, gt_images, album_ids, audio_embeddings in tqdm(dataloader):
            gt_images = gt_images.to(self.torch_dtype)
            gt_images = gt_images.to(self.torch_device)
            audio_embeddings = audio_embeddings.to(self.torch_device)

            # First, we get the text_embeddings for the prompt. These embeddings will be used to condition the UNet model.
            text_input = self.tokenizer(prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            with torch.inference_mode():
                text_embeddings = self.text_encoder(text_input.input_ids.to(self.torch_device))[0]

            # Perform the forward pass
            with torch.inference_mode():
                inf_images = self.model.forward(self.scheduler, text_embeddings, audio_embeddings, self.torch_device)

            # Rescale the images
            rescaled_images = (inf_images / 2 + 0.5).clamp(0, 1)
            rescaled_images = rescaled_images.detach().cpu().permute(0, 2, 3, 1).numpy()
            rescaled_images = (rescaled_images * 255).round().astype("uint8")
            pil_images = [Image.fromarray(rescaled_image) for rescaled_image in rescaled_images]

            # Save the images
            for j in range(len(prompts)):
                pil_images[j].save(os.path.join(output_dir, f"{album_ids[j]}.png"))

            del gt_images, inf_images, audio_embeddings, prompts, rescaled_images, pil_images

    