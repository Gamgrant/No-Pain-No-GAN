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
    parser.add_argument("--album_metadata_train_file", type=str, default=None)
    parser.add_argument("--album_metadata_test_file", type=str, default=None)
    parser.add_argument("--audio_embed_folder", type=str, default=None)
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=1)
    parser.add_argument("--max_timesteps", type=int, default=50)
    parser.add_argument("--model_type", type=str, default='')
    parser.add_argument("--sampler", type=str, default="ddpm")
    parser.add_argument("--torch_dtype", default=torch.float16)
    parser.add_argument("--run_name", default=None)

    args = parser.parse_args()

    if args.album_root_path is None:
        print("Please specify the path containing all the album cover art images (e.g., --album_root_path <path>)")
        exit()

    if args.album_metadata_train_file is None or args.album_metadata_test_file is None:
        print("Please specify the train and/or metadata CSV file path (e.g., --album_metadata_<train/test>_file <path>)")
        exit()

    if args.audio_embed_folder is None:
        print("Please specify the path containing all the audio embeddings (e.g., --audio_embed_folder <path>)")
        exit()

    # Set up the train and test datasets
    train_dataset = AlbumCoverArtDataset(args.album_root_path, args.album_metadata_train_file, args.audio_embed_folder)
    test_dataset = AlbumCoverArtDataset(args.album_root_path, args.album_metadata_test_file, args.audio_embed_folder)

    # Set up the dataloaders
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 1, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False)

    # Set up the pipeline
    lr_num_training_steps = len(train_loader) * args.epochs
    # Set up the pipeline
    pipe = AudioSDPipeline(model_id=args.model_id,
                           lr_num_training_steps=lr_num_training_steps,
                           sampler=args.sampler,
                           torch_dtype=torch.float16,
                           mode='train',
                           model_type=args.model_type)
    
    if args.run_name is None:
        os.makedirs("checkpoints/checkpoints_", exist_ok=True)
        os.makedirs("losses/losses_", exist_ok=True)
        run_name = ''
    else:
        os.makedirs(f"checkpoints/checkpoints_{args.run_name}", exist_ok=True)
        os.makedirs(f"losses/losses_{args.run_name}", exist_ok=True)
        run_name = args.run_name

    total_train_loss = []
    total_eval_loss = []
    train_loss = []
    eval_loss = []
    for i in range(args.epochs):
        print("\nEpoch {}/{}".format(i+1, args.epochs))

        # Train the model
        total_train_loss_temp, train_loss_temp = pipe.train(train_loader, args.max_timesteps)

        # Evaluate the model
        if (i + 1) % args.save_steps == 0:
            total_eval_loss_temp, eval_loss_temp = pipe.eval(test_loader, args.max_timesteps, True, i+1, run_name)
            torch.save({'model_state_dict':pipe.model.state_dict(),
                    'optimizer_state_dict':pipe.optimizer.state_dict(),
                    'scheduler_state_dict':pipe.lr_scheduler.state_dict(),
                    'eval_loss': eval_loss_temp,
                    'epoch': i+1}, f'checkpoints/checkpoints_{run_name}/checkpoint_asd_epoch{i+1}.pth')
            print("Saved Model and Images")
        else:
            total_eval_loss_temp, eval_loss_temp = pipe.eval(test_loader, args.max_timesteps)

        print("Train Loss {:.04f}".format(total_train_loss_temp))
        print("Eval Loss {:.04f}".format(total_eval_loss_temp))

        # Append the losses
        total_train_loss.append(total_train_loss_temp)
        total_eval_loss.append(total_eval_loss_temp)
        train_loss.extend(train_loss_temp)
        eval_loss.extend(eval_loss_temp)
    
        np.save(f'losses/losses_{run_name}/train_losses_epoch{i+1}.npy', np.array(train_loss))
        np.save(f'losses/losses_{run_name}/eval_losses_epoch{i+1}.npy', np.array(eval_loss))
        np.save(f'losses/losses_{run_name}/total_train_losses_epoch{i+1}.npy', np.array(total_train_loss))
        np.save(f'losses/losses_{run_name}/total_eval_losses_epoch{i+1}.npy', np.array(total_eval_loss))
    
    print("Training Done")