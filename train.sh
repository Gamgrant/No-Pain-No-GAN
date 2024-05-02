#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -t 03:00:00
#SBATCH --gres=gpu:v100-32:1
module load anaconda3
conda activate album_cover_art_gen

python final_code/main_asd.py --album_root_path "./album_cover_art" \
                      --album_metadata_train_file "./metadata/album_metadata_train.csv" \
                      --album_metadata_test_file "./metadata/album_metadata_test.csv" \
                      --audio_embed_folder "./jukebox_outputs" \
                      --model_id "lambdalabs/miniSD-diffusers" \
                      --epochs 1 \
                      --save_steps 1 \
                      --max_timesteps 150 \
                      --model_type "concatenation" \
                      --sampler "pndm" \
                      --run_name "run_150t_concatenation_test"
