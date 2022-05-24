## Suggested Environment
```
conda create -n detenv python=3.8 pytorch==1.9.0 cudatoolkit=11.1 torchvision -c pytorch -y
conda activate detenv
pip install timm==0.4.12
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.20.0
```

## Train Command
```
./dist_train.sh configs/gfl/gfl_mae_swin_tiny_256_mask_vmr025_200e_100e_fpn_25ep_1024x1024_coco.py 8
```