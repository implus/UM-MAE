## Fine-tuning UM-MAE

A typical command To fine-tune Swin-T (recommended default) with **single-node distributed training**, run the following on 1 node with 8 GPUs each:
```
python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
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
```
Please modify your data_path /path/to/ImageNet/ and possibly the dataloader.

## More detailed training script follows:
| Models  | Pre-train Method| Sampling Strategy | Secondary Mask Ratio | Encoder Ratio | Finetune Epochs | Finetune Command |
| :---:   | :---: | :---: | :---: | :---: | :---: | :---: |
| ViT-B   | MAE          | RS | --  | 25%  | 100 | ```make finetune_mae_vit_base_patch16_dec512d2b_200e_100e```|
| ViT-B   | MAE          | UM | 25% | 25%  | 100 | ```make finetune_mae_vit_base_patch16_dec512d2b_mask_vmr025_200e_100e```| 
| PVT-S   | SimMIM       | RS | --  | 100% | 100 | ```make finetune_simmim_pvt_small_256_200e_100e```| 
| PVT-S   | UM-MAE (ours)| UM | 25% | 25%  | 100 | ```make finetune_mae_pvt_small_256_mask_vmr025_200e_100e```| 
| Swin-T  | SimMIM       | RS | --  | 100% | 100 | ```make finetune_simmim_swin_tiny_256_200e_100e```| 
| Swin-T  | UM-MAE (ours)| UM | 25% | 25%  | 100 | ```make finetune_mae_swin_tiny_256_mask_vmr025_200e_100e```| 
| Swin-L  | SimMIM       | RS | --  | 100% | 100 | see [official](https://github.com/microsoft/SimMIM) | 
| Swin-L  | UM-MAE (ours)| UM | 25% | 25%  | 100 | ```make finetune_mae_swin_large_256_mask_vmr025_200e_100e```| 
