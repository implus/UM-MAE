## Pre-training UM-MAE

A typical command To pre-train Swin-T (recommended default) with **single-node distributed training**, run the following on 1 node with 8 GPUs each:
```
python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py \
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
```
Please modify your data_path /path/to/ImageNet/ and possibly the dataloader.
You can also move the txt files **IN1K/train.txt** and **IN1K/val.txt** to your imagenet root path.

## More detailed training script follows:
| Models  | Pre-train Method| Sampling Strategy | Secondary Mask Ratio | Encoder Ratio | Pretrain Epochs | Pretrain Command |
| :---:   | :---: | :---: | :---: | :---: | :---: | :---: |
| ViT-B   | MAE          | RS | --  | 25%  | 200 | ```make pretrain_mae_vit_base_patch16_dec512d2b_224_200e```|
| ViT-B   | MAE          | UM | 25% | 25%  | 200 | ```make pretrain_mae_vit_base_patch16_dec512d2b_224_mask_vmr025_200e```| 
| PVT-S   | SimMIM       | RS | --  | 100% | 200 | ```make pretrain_simmim_pvt_small_256_200e``` | 
| PVT-S   | UM-MAE (ours)| UM | 25% | 25%  | 200 | ```make pretrain_mae_pvt_small_256_mask_vmr025_200e```| 
| Swin-T  | SimMIM       | RS | --  | 100% | 200 | ```make pretrain_simmim_swin_tiny_256_200e```| 
| Swin-T  | UM-MAE (ours)| UM | 25% | 25%  | 200 | ```make pretrain_mae_swin_tiny_256_mask_vmr025_200e```| 
| Swin-L  | SimMIM       | RS | --  | 100% | 800 | see [official](https://github.com/microsoft/SimMIM) | 
| Swin-L  | UM-MAE (ours)| UM | 25% | 25%  | 800 | ```make pretrain_mae_swin_large_256_mask_vmr025_800e```| 
