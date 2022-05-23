## Pre-training UM-MAE

A typical command To pre-train Swin-T (recommended default) with **single-node distributed training**, run the following on 1 node with 8 GPUs each:
```
python submitit_pretrain.py \
    --job_dir ${JOB_DIR} \
    --nodes 8 \
    --use_volta32 \
    --batch_size 64 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR}
```

# More detailed training script follows:
| Models  | Pre-train Method| Sampling Strategy | Secondary Mask Ratio | Encoder Ratio | Pretrain Epochs | Pretrain Command |
| :---:   | :---: | :---: | :---: | :---: | :---: | :---: |
| ViT-B   | MAE          | RS | --  | 25%  | 200 | make  |
| ViT-B   | UM-MAE (ours)| UM | 25% | 25%  | 200 | make  | 
| PVT-S   | SimMIM       | RS | --  | 100% | 200 | make  | 
| PVT-S   | UM-MAE (ours)| UM | 25% | 25%  | 200 | ```make pretrain_mae_pvt_small_256_mask_vmr025_200e```| 
| Swin-T  | SimMIM       | RS | --  | 100% | 200 | make  | 
| Swin-T  | UM-MAE (ours)| UM | 25% | 25%  | 200 | ```make pretrain_mae_swin_tiny_256_mask_vmr025_200e```| 
| Swin-L  | SimMIM       | RS | --  | 100% | 800 | see [official](https://github.com/microsoft/SimMIM) | 
| Swin-L  | UM-MAE (ours)| UM | 25% | 25%  | 800 | make  | 
