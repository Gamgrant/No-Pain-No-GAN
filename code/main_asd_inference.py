from dataset import AlbumCoverArtDataset
from pipeline import AudioSDPipeline
import torch
import argparse
from tqdm import tqdm
import numpy as np
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--album_root_path", type=str, default=None)
    parser.add_argument("--album_metadata_file", type=str, default=None)
    parser.add_argument("--audio_embed_folder", type=str, default=None)
    parser.add_argument("--model_id", type=str, default="lambdalabs/miniSD-diffusers")
    parser.add_argument("--pretrained_ckpt", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--sampler", type=str, default="ddpm")
    parser.add_argument("--torch_dtype", default=torch.float16)
    parser.add_argument("--model_type", type=str, default='')
    parser.add_argument("--run_name", default=None)

    args = parser.parse_args()

    if args.album_root_path is None:
        print("Please specify the path containing all the album cover art images (e.g., --album_root_path <path>)")
        exit()

    if args.album_metadata_file is None:
        print("Please specify the train and/or metadata CSV file path (e.g., --album_metadata_<train/test>_file <path>)")
        exit()

    if args.audio_embed_folder is None:
        print("Please specify the path containing all the audio embeddings (e.g., --audio_embed_folder <path>)")
        exit()

    if args.pretrained_ckpt is None:
        print("Please specify the path containing the model checkpoint (e.g., --pretrained_ckpt <path>)")
        exit()

    # Set up the train and test datasets
    dataset = AlbumCoverArtDataset(args.album_root_path, args.album_metadata_file, args.audio_embed_folder)

    # Set up the dataloaders
    dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 1, shuffle = True)

    # Set up the pipeline
    pipe = AudioSDPipeline(model_id = args.model_id, 
                           sampler = args.sampler, 
                           torch_dtype = args.torch_dtype, 
                           pretrained_ckpt = args.pretrained_ckpt,
                           mode='inference',
                           model_type=args.model_type)

    if args.run_name is None:
        output_dir = "inference_imgs/test"
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = f"inference_imgs/{args.run_name}"
        os.makedirs(output_dir, exist_ok=True)

    # Perform inference
    pipe.inference(dataloader, args.num_inference_steps, output_dir)
    
    print("Inference Done")