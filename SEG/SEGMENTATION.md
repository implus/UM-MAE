## Suggested Environment
```
conda create -n segenv python=3.8 pytorch==1.9.0 cudatoolkit=11.1 torchvision -c pytorch -y
conda activate segenv
pip install timm==0.3.2
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmseg==0.11.0
```

## Train Command
```
./dist_train.sh configs/upernet/upernet_mae_swin_tiny_256_mask_vmr025_200e_100e_512_slide_160k_ade20k_lr1e4_lrd1_pt.py 8
```