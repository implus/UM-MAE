pretrain_mae_vit_base_patch16_dec512d2b_224_200e:
	python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py \
			--batch_size 128 \
			--accum_iter 4 \
			--model mae_vit_base_patch16_dec512d2b \
			--token_size 14 \
			--norm_pix_loss \
			--mask_ratio 0.75 \
			--epochs 200 \
			--warmup_epochs 10 \
			--blr 1.5e-4 \
			--weight_decay 0.05 \
			--data_path /path/to/ImageNet/ \
			--log_dir ./work_dirs/pretrain_mae_vit_base_patch16_dec512d2b_224_200e \
			--output_dir ./work_dirs/pretrain_mae_vit_base_patch16_dec512d2b_224_200e

pretrain_mae_vit_base_patch16_dec512d2b_224_mask_vmr025_200e:
	python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py \
			--batch_size 128 \
			--accum_iter 4 \
			--model mae_vit_base_patch16_dec512d2b \
			--mask_regular \
			--vis_mask_ratio 0.25 \
			--token_size 14 \
			--norm_pix_loss \
			--mask_ratio 0.75 \
			--epochs 200 \
			--warmup_epochs 10 \
			--blr 1.5e-4 \
			--weight_decay 0.05 \
			--data_path /path/to/ImageNet/ \
			--log_dir ./work_dirs/pretrain_mae_vit_base_patch16_dec512d2b_224_mask_vmr025_200e \
			--output_dir ./work_dirs/pretrain_mae_vit_base_patch16_dec512d2b_224_mask_vmr025_200e  \

pretrain_mae_swin_tiny_256_mask_vmr025_200e:
	python -m torch.distributed.launch --nproc_per_node=8 --master_port 29502 main_pretrain.py \
		--batch_size 128 \
		--accum_iter 4 \
		--model mae_swin_tiny_256 \
		--mask_regular \
		--vis_mask_ratio 0.25 \
		--input_size 256 \
		--token_size 16 \
		--norm_pix_loss \
		--mask_ratio 0.75 \
		--epochs 200 \
		--warmup_epochs 10 \
		--blr 1.5e-4 \
		--weight_decay 0.05 \
		--data_path /path/to/ImageNet/ \
		--log_dir ./work_dirs/pretrain_mae_swin_tiny_256_mask_vmr025_200e \
		--output_dir ./work_dirs/pretrain_mae_swin_tiny_256_mask_vmr025_200e 

pretrain_simmim_swin_tiny_256_200e:
	python -m torch.distributed.launch --nproc_per_node=8 --master_port 29502 main_pretrain.py \
		--batch_size 64 \
		--accum_iter 8 \
		--model simmim_swin_tiny_256 \
		--input_size 256 \
		--token_size 16 \
		--mask_ratio 0.75 \
		--epochs 200 \
		--warmup_epochs 10 \
		--blr 1.5e-4 \
		--weight_decay 0.05 \
		--data_path /path/to/ImageNet/ \
		--log_dir ./work_dirs/pretrain_simmim_swin_tiny_256_200e \
		--output_dir ./work_dirs/pretrain_simmim_swin_tiny_256_200e

pretrain_mae_pvt_small_256_mask_vmr025_200e:
	python -m torch.distributed.launch --nproc_per_node=8 --master_port 29502 main_pretrain.py \
		--batch_size 128 \
		--accum_iter 4 \
		--model mae_pvt_small_256 \
		--mask_regular \
		--vis_mask_ratio 0.25 \
		--input_size 256 \
		--token_size 16 \
		--norm_pix_loss \
		--mask_ratio 0.75 \
		--epochs 200 \
		--warmup_epochs 10 \
		--blr 1.5e-4 \
		--weight_decay 0.05 \
		--data_path /path/to/ImageNet/ \
		--log_dir ./work_dirs/pretrain_mae_pvt_small_256_mask_vmr025_200e \
		--output_dir ./work_dirs/pretrain_mae_pvt_small_256_mask_vmr025_200e 

pretrain_simmim_pvt_small_256_200e:
	python -m torch.distributed.launch --nproc_per_node=8 --master_port 29501 main_pretrain.py \
		--batch_size 128 \
		--accum_iter 4 \
		--model simmim_pvt_small_256 \
		--input_size 256 \
		--token_size 16 \
		--mask_ratio 0.75 \
		--epochs 200 \
		--warmup_epochs 10 \
		--blr 1.5e-4 \
		--weight_decay 0.05 \
		--data_path /path/to/ImageNet/ \
		--log_dir ./work_dirs/pretrain_simmim_pvt_small_256_200e \
		--output_dir ./work_dirs/pretrain_simmim_pvt_small_256_200e

pretrain_mae_swin_large_256_mask_vmr025_800e:
	python -m torch.distributed.launch --nproc_per_node=8 --master_port 29501 main_pretrain.py \
		--batch_size 64 \
		--accum_iter 8 \
		--model mae_swin_large_256 \
		--mask_regular \
		--vis_mask_ratio 0.25 \
		--input_size 256 \
		--token_size 16 \
		--norm_pix_loss \
		--mask_ratio 0.75 \
		--epochs 800 \
		--warmup_epochs 40 \
		--blr 1.5e-4 \
		--weight_decay 0.05 \
		--data_path /path/to/ImageNet/ \
		--log_dir ./work_dirs/pretrain_mae_swin_large_256_mask_vmr025_800e \
		--output_dir ./work_dirs/pretrain_mae_swin_large_256_mask_vmr025_800e




finetune_mae_vit_base_patch16_dec512d2b_mask_vmr025_200e_100e:
	python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
			--batch_size 64 \
			--accum_iter 2 \
			--model vit_base_patch16 \
			--finetune ./work_dirs/pretrain_mae_vit_base_patch16_dec512d2b_224_mask_vmr025_200e//checkpoint-199.pth \
			--epochs 100 \
			--blr 5e-4 --layer_decay 0.8 \
			--weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
			--dist_eval --data_path /path/to/ImageNet/ \
			--log_dir ./work_dirs/finetune_mae_vit_base_patch16_dec512d2b_mask_vmr025_200e_100e \
			--output_dir ./work_dirs/finetune_mae_vit_base_patch16_dec512d2b_mask_vmr025_200e_100e

finetune_mae_vit_base_patch16_dec512d2b_200e_100e:
	python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
			--batch_size 64 \
			--accum_iter 2 \
			--model vit_base_patch16 \
			--finetune ./work_dirs/pretrain_mae_vit_base_patch16_dec512d2b_224_200e//checkpoint-199.pth \
			--epochs 100 \
			--blr 5e-4 --layer_decay 0.8 \
			--weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
			--dist_eval --data_path /path/to/ImageNet/ \
			--log_dir ./work_dirs/finetune_mae_vit_base_patch16_dec512d2b_200e_100e \
			--output_dir ./work_dirs/finetune_mae_vit_base_patch16_dec512d2b_200e_100e

finetune_mae_swin_tiny_256_mask_vmr025_200e_100e:
	python -m torch.distributed.launch --nproc_per_node=8 --master_port 29502 main_finetune.py \
		--input_size 256 \
		--batch_size 64 \
		--accum_iter 2 \
		--model swin_tiny_256 \
		--finetune ./work_dirs/pretrain_mae_swin_tiny_256_mask_vmr025_200e//checkpoint-199.pth \
		--epochs 100 \
		--blr 5e-4 --layer_decay 0.85 \
		--weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
		--dist_eval --data_path /path/to/ImageNet/ \
		--log_dir ./work_dirs/finetune_mae_swin_tiny_256_mask_vmr025_200e_100e \
		--output_dir ./work_dirs/finetune_mae_swin_tiny_256_mask_vmr025_200e_100e 

finetune_mae_pvt_small_256_mask_vmr025_200e_100e:
	python -m torch.distributed.launch --nproc_per_node=8 --master_port 29502 main_finetune.py \
		--input_size 256 \
		--batch_size 64 \
		--accum_iter 2 \
		--model pvt_small_256 \
		--finetune ./work_dirs/pretrain_mae_pvt_small_256_mask_vmr025_200e//checkpoint-199.pth \
		--epochs 100 \
		--blr 5e-4 --layer_decay 0.85 \
		--weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
		--dist_eval --data_path /path/to/ImageNet/ \
		--log_dir ./work_dirs/finetune_mae_pvt_small_256_mask_vmr025_200e_100e \
		--output_dir ./work_dirs/finetune_mae_pvt_small_256_mask_vmr025_200e_100e

finetune_simmim_pvt_small_256_200e_100e:
	python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
		--input_size 256 \
		--batch_size 64 \
		--accum_iter 2 \
		--model pvt_small_256 \
		--finetune ./work_dirs/pretrain_simmim_pvt_small_256_200e/checkpoint-199.pth \
		--epochs 100 \
		--blr 5e-4 --layer_decay 0.85 \
		--weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
		--dist_eval --data_path /path/to/ImageNet/ \
		--log_dir ./work_dirs/finetune_simmim_pvt_small_256_200e_100e \
		--output_dir ./work_dirs/finetune_simmim_pvt_small_256_200e_100e

finetune_simmim_swin_tiny_256_200e_100e:
	python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
		--input_size 256 \
		--batch_size 64 \
		--accum_iter 2 \
		--model swin_tiny_256 \
		--finetune ./work_dirs/pretrain_simmim_swin_tiny_256_200e/checkpoint-199.pth \
		--epochs 100 \
		--blr 5e-4 --layer_decay 0.85 \
		--weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
		--dist_eval --data_path /path/to/ImageNet/ \
		--log_dir ./work_dirs/finetune_simmim_swin_tiny_256_200e_100e \
		--output_dir ./work_dirs/finetune_simmim_swin_tiny_256_200e_100e

finetune_mae_swin_large_256_mask_vmr025_200e_100e:
	python -m torch.distributed.launch --nproc_per_node=8 --master_port 29502 main_finetune.py \
		--input_size 256 \
		--batch_size 32 \
		--accum_iter 4 \
		--model swin_large_256 \
		--finetune ./work_dirs/pretrain_mae_swin_large_256_mask_vmr025_200e//checkpoint-199.pth \
		--epochs 100 \
		--blr 5e-4 --layer_decay 0.7 \
		--weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
		--dist_eval --data_path /path/to/ImageNet/ \
		--log_dir ./work_dirs/finetune_mae_swin_large_256_mask_vmr025_200e_100e \
		--output_dir ./work_dirs/finetune_mae_swin_large_256_mask_vmr025_200e_100e