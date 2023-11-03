CUDA_VISIBLE_DEVICES=0  python src/eval/eval_stable_lsd_generation.py \
--ckpt_path /path_to_your_logs/lsd/coco/stable_lsd/checkpoint-400000/ \
--output_dir /path_to_your_image_logs \
--enable_xformers_memory_efficient_attention --mixed_precision fp16 --num_workers 4