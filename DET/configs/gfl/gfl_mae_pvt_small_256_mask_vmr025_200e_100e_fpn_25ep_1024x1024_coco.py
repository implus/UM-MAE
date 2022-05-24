_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
size = 1024
model = dict(
    type='GFL',
    backbone=dict(type='PVTDet',
                  img_size=size,
                  drop_path_rate=0.1,
                  #pretrained='pretrained/mae_pretrain_vit_base_.pth'),
                  #pretrained='/xiangli/mae/pretrain/mae_pretrain_vit_base_.pth'),
                  #pretrained='/xiangli/mae_fast/output_dir/pretrain_mae_vit_base_patch16_dec512d2b_224_200e/checkpoint-199_.pth'),
                  pretrained='/xiangli/UniformMasking/output_dir/finetune_mae_pvt_small_256_mask_vmr025_200e_100e/checkpoint-99.pth'),
                  #'/xiangli/UniformMasking/output_dir/finetune_mae_swin_tiny_256_mask_vmr025_200e_100e/checkpoint-99.pth'),
    neck=dict(
        type='FPN',
        in_channels=[768, 768, 768, 768],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        norm_cfg=dict(type='MMSyncBN', requires_grad=True)),
    bbox_head=dict(
        type='GFLHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100)
)
# optimizer
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=(size, size),
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(size, size),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(size, size)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(
    train=dict(pipeline=train_pipeline),
    #test=dict(pipeline=test_pipeline)
)
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    min_lr_ratio=0.01,
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=0.001)
runner = dict(type='EpochBasedRunner', max_epochs=25)
optimizer = dict(
    _delete_=True, type='AdamW', lr=0.0001, weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'bias': dict(decay_mult=0.)
        }))
optimizer_config = dict(grad_clip=None)
fp16 = dict(loss_scale=dict(init_scale=512))
checkpoint_config=dict(interval=5, create_symlink=False)

find_unused_parameters=True
