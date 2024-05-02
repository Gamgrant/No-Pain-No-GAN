#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -t 03:00:00
#SBATCH --gres=gpu:v100-32:1
module load anaconda3
conda activate album_cover_art_gen

python final_code/main_sd.py --album_root_path "./album_cover_art" \
                      --album_metadata_file "./metadata/album_metadata_minitest.csv" \
                      --model_id "lambdalabs/miniSD-diffusers" \
                      --num_inference_steps 50 \
                      --sampler "ddpm" \
                      --gen_seed 42 \
                      --output_dir "prompt_only_imgs_256_ddpm_test" \
                      --save_images True
