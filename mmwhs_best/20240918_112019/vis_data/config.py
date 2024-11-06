checkpoint_file = './work_dirs/fcn_psvt_256x256_80k_cardiac/20240916_112410/iter_80000.pth'
crop_size = (
    256,
    256,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        256,
        256,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = 'data/mr-cardiac'
dataset_type = 'CardiacDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=2000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
find_unused_parameters = True
img_norm_cfg = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
img_scale = (
    300,
    300,
)
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        ape=False,
        depths=[
            2,
            2,
            6,
            2,
        ],
        drop_path_rate=0.2,
        img_size=256,
        in_channels=3,
        type='PSVT',
        window_size=[
            8,
            8,
            8,
            8,
        ]),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            256,
            256,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        channels=96,
        in_channels=96,
        in_index=-1,
        loss_decode=[
            dict(
                loss_name='loss_ce', loss_weight=1.0, type='CrossEntropyLoss'),
            dict(loss_name='loss_dice', loss_weight=1.0, type='DiceLoss'),
        ],
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=8,
        type='FCNHead'),
    test_cfg=dict(crop_size=(
        512,
        512,
    ), mode='whole', stride=(
        85,
        85,
    )),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.01, type='AdamW', weight_decay=0.1),
    type='OptimWrapper')
optimizer = dict(lr=0.01, type='AdamW', weight_decay=0.1)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=12000, start_factor=0.03,
        type='LinearLR'),
    dict(
        begin=12000,
        by_epoch=False,
        end=24000,
        eta_min_ratio=0.03,
        power=0.9,
        type='PolyLRRatio'),
    dict(begin=24000, by_epoch=False, end=25000, factor=1, type='ConstantLR'),
]
resume = False
step = 's1'
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        data_prefix=dict(img_path='test/image', seg_map_path='test/label'),
        data_root='data/mr-cardiac',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                256,
                256,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='CardiacDataset'),
    drop_last=False,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        256,
        256,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=25000, type='IterBasedTrainLoop', val_interval=1000)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        data_prefix=dict(img_path='train/image', seg_map_path='train/label'),
        data_root='data/mr-cardiac',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    300,
                    300,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    256,
                    256,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='CardiacDataset'),
    num_workers=16,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            300,
            300,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        256,
        256,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        data_prefix=dict(img_path='test/image', seg_map_path='test/label'),
        data_root='data/mr-cardiac',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                256,
                256,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='CardiacDataset'),
    drop_last=False,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = './work_dirs/fcn_psvt_256x256_25k_cardiac'
