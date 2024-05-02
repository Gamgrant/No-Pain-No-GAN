from dataset import AlbumCoverArtDataset
from pipeline import SDPipeline
import torch
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--album_root_path", type=str, default=None)
    parser.add_argument("--album_metadata_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--sampler", type=str, default="ddpm")
    parser.add_argument("--torch_dtype", default=torch.float16)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--gen_seed", type=int, default=None)
    parser.add_argument("--save_images", type=bool, default=False)

    args = parser.parse_args()

    if args.album_root_path is None:
        print("Please specify the path containing all the album cover art images (e.g., --album_root_path <path>)")
        exit()

    if args.album_metadata_file is None:
        print("Please specify the metadata CSV file path (e.g., --album_metadata_file <path>)")
        exit()

    album_dataset = AlbumCoverArtDataset(args.album_root_path, args.album_metadata_file)

    album_loader = torch.utils.data.DataLoader(dataset = album_dataset, batch_size = 1, shuffle = False, num_workers = 2)

    pipe = SDPipeline(args.model_id, args.num_inference_steps, args.sampler, args.torch_dtype, args.output_dir, args.save_images)

    total_loss = pipe.eval(album_loader, guidance_scale=args.guidance_scale, gen_seed=args.gen_seed)

    print(f"{args.model_id} for {args.num_inference_steps} inference steps, sampler {args.sampler}, seed {args.gen_seed} done")
    print("Inference MSE Loss {:.04f}".format(total_loss))







