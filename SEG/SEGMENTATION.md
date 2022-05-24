## Suggested Environment
```
conda create -n segenv python=3.8 pytorch==1.9.0 cudatoolkit=11.1 torchvision -c pytorch -y
conda activate segenv
pip install mmcv-full==1.3.0 
pip install mmseg==0.11.0
pip install scipy timm==0.3.2
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Data Prepare
Follow the guide in [mmseg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md) to prepare the ADE20k dataset.

## Train Command
```
./dist_train.sh configs/upernet/upernet_mae_swin_tiny_256_mask_vmr025_200e_100e_512_slide_160k_ade20k_lr1e4_lrd1_pt.py 8
```