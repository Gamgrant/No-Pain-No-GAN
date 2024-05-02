import jukemirlib
import os
import torch.distributed as dist
from tqdm import tqdm
import numpy as np
import argparse

if __name__ == "__main__":
    # Setup initial process group
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'

    dist.init_process_group(backend='nccl', init_method='env://')

    # Obtain the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--audio_folder", default="./audio_new")
    parser.add_argument("--fp16", default=False)
    
    args = parser.parse_args()

    # Create the output directory
    if args.fp16 == True:
        outpath = "./jukebox_outputs_fp16"
        os.makedirs(outpath, exist_ok=True)
        flag = True
    else:
        outpath = "./jukebox_outputs_fp32"
        os.makedirs(outpath, exist_ok=True)
        flag = False
    
    # Obtain all the audio file paths
    audio_paths = sorted(os.listdir(args.audio_folder))

    # Process all the outputs
    for audio_path in tqdm(audio_paths):
        # This audio file is empty
        if audio_path == "7aRwE6mjXAIH1vcySepuvS.mp3":
            print(audio_path)
            continue
    
        # Obtain the representations
        audio = jukemirlib.load_audio(os.path.join("/home/ubuntu/audio_new", audio_path), offset=0.0, duration=25)
        rep = jukemirlib.extract(audio, layers=[36], fp16=flag, fp16_out=flag, downsample_target_rate=30)
        s = audio_path.split('.mp3')[0]
        np.save(os.path.join(outpath, f'{s}.npy'), rep[36])