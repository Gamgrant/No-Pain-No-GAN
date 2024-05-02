#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -t 00:15:00
#SBATCH --gres=gpu:v100-32:1
module load anaconda3
conda activate album_cover_art_gen

python final_code/main_asd_inference.py --album_root_path "./album_cover_art" \
                      --album_metadata_file "./album_metadata_minitrain.csv" \
                      --audio_embed_folder "./jukebox_outputs" \
                      --model_id "lambdalabs/miniSD-diffusers" \
                      --pretrained_ckpt "./checkpoints/checkpoints_run_150t_linear_layer_nogradclip_pndm_50epochs/checkpoint_asd_epoch45.pth" \
                      --num_inference_steps 150 \
                      --sampler "pndm" \
                      --model_type "linear layer" \
                      --run_name "run_150t_linear_layer_pndm_minitrain_epoch45"

# python main_asd_inference.py --album_root_path "./album_cover_art" \
#                       --album_metadata_file "./album_metadata_minitest.csv" \
#                       --audio_embed_folder "./jukebox_outputs" \
#                       --model_id "lambdalabs/miniSD-diffusers" \
#                       --pretrained_ckpt "./checkpoints/checkpoints_run_150t_linear_layer_nogradclip_pndm_50epochs/checkpoint_asd_epoch45.pth" \
#                       --num_inference_steps 150 \
#                       --sampler "pndm" \
#                       --model_type "linear layer" \
#                       --run_name "run_150t_linear_layer_pndm_minitest_epoch45"

# python main_asd_inference.py --album_root_path "./album_cover_art" \
#                       --album_metadata_file "./album_metadata_minitrain.csv" \
#                       --audio_embed_folder "./jukebox_outputs" \
#                       --model_id "lambdalabs/miniSD-diffusers" \
#                       --pretrained_ckpt "./checkpoints/checkpoints_ablation_150t_concatenation_nogradclip_ddpm/checkpoint_asd_epoch1.pth" \
#                       --num_inference_steps 150 \
#                       --sampler "ddpm" \
#                       --model_type "concatenation" \
#                       --run_name "ablation_150t_concatenation_ddpm_minitrain"

# python main_asd_inference.py --album_root_path "./album_cover_art" \
#                       --album_metadata_file "./album_metadata_minitest.csv" \
#                       --audio_embed_folder "./jukebox_outputs" \
#                       --model_id "lambdalabs/miniSD-diffusers" \
#                       --pretrained_ckpt "./checkpoints/checkpoints_ablation_150t_concatenation_nogradclip_ddpm/checkpoint_asd_epoch1.pth" \
#                       --num_inference_steps 150 \
#                       --sampler "ddpm" \
#                       --model_type "concatenation" \
#                       --run_name "ablation_150t_concatenation_ddpm_minitest"

# python main_asd_inference.py --album_root_path "./album_cover_art" \
#                       --album_metadata_file "./album_metadata_minitrain.csv" \
#                       --audio_embed_folder "./jukebox_outputs" \
#                       --model_id "lambdalabs/miniSD-diffusers" \
#                       --pretrained_ckpt "./checkpoints/checkpoints_ablation_150t_ca_8heads_nogradclip_pndm/checkpoint_asd_epoch1.pth" \
#                       --num_inference_steps 150 \
#                       --sampler "pndm" \
#                       --model_type "cross attention" \
#                       --run_name "ablation_150t_ca_8heads_pndm_minitrain"

# python main_asd_inference.py --album_root_path "./album_cover_art" \
#                       --album_metadata_file "./album_metadata_minitest.csv" \
#                       --audio_embed_folder "./jukebox_outputs" \
#                       --model_id "lambdalabs/miniSD-diffusers" \
#                       --pretrained_ckpt "./checkpoints/checkpoints_ablation_150t_ca_8heads_nogradclip_pndm/checkpoint_asd_epoch1.pth" \
#                       --num_inference_steps 150 \
#                       --sampler "pndm" \
#                       --model_type "cross attention" \
#                       --run_name "ablation_150t_ca_8heads_pndm_minitest"

# python main_asd_inference.py --album_root_path "./album_cover_art" \
#                       --album_metadata_file "./album_metadata_minitrain.csv" \
#                       --audio_embed_folder "./jukebox_outputs" \
#                       --model_id "lambdalabs/miniSD-diffusers" \
#                       --pretrained_ckpt "./checkpoints/checkpoints_ablation_150t_concatenation_nogradclip_pndm/checkpoint_asd_epoch1.pth" \
#                       --num_inference_steps 150 \
#                       --sampler "pndm" \
#                       --model_type "concatenation" \
#                       --run_name "ablation_150t_concatenation_pndm_minitrain"

# python main_asd_inference.py --album_root_path "./album_cover_art" \
#                       --album_metadata_file "./album_metadata_minitest.csv" \
#                       --audio_embed_folder "./jukebox_outputs" \
#                       --model_id "lambdalabs/miniSD-diffusers" \
#                       --pretrained_ckpt "./checkpoints/checkpoints_ablation_150t_concatenation_nogradclip_pndm/checkpoint_asd_epoch1.pth" \
#                       --num_inference_steps 150 \
#                       --sampler "pndm" \
#                       --model_type "concatenation" \
#                       --run_name "ablation_150t_concatenation_pndm_minitest"