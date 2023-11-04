CUDA_VISIBLE_DEVICES=3,4,5,6 accelerate launch --multi_gpu --num_processes=4 --main_process_port 29500 train_lsd.py \
--enable_xformers_memory_efficient_attention --dataloader_num_workers 4 --learning_rate 2e-5 \
--mixed_precision fp16 --num_validation_images 32 --val_batch_size 32 --max_train_steps 500000 \
--checkpointing_steps 25000 --checkpoints_total_limit 2 --gradient_accumulation_steps 1 \
--seed 42 --encoder_lr_scale 1.0 --train_split_portion 1.0 \
--output_dir /path_to_your_logs/lsd/coco/ \
--backbone_config pretrain_dino \
--slot_attn_config src/configs/coco/slot_attn/config.json \
--scheduler_config none \
--unet_config pretrain_sd \
--dataset_root /path_to_your_coco/train2017 \
--dataset_glob '**/*.jpg' --flip_images --train_batch_size 32 --resolution 256 --validation_steps 5000 \
--tracker_project_name stable_lsd