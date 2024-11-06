# img_size = (256, 256)
norm_cfg = dict(type='SyncBN', requires_grad=True)
checkpoint_file = './work_dirs/fcn_psvt_256x256_80k_cardiac/20240916_112410/iter_80000.pth'
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
    backbone=dict(
        type='PSVT',
        in_channels=3,
        window_size=[8, 8, 8, 8],
        img_size=256,
        depths=[2,2,6,2],
        # load_pretrain_path='./checkpoint/swin_tiny_patch4_window7_224.pth',
        drop_path_rate=0.2, ape=False
        # img_size=img_size, embedding_dim=96, window_size=7,
        # dropout=0.2,depth=2
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=96,
        in_index=-1,
        channels=96,
        num_classes=2,
        # kernel_size=1,
        # num_convs=1,
        norm_cfg=norm_cfg,
        # ignore_index=0,
        # sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0
                #  class_weight=[1.0, 1.3, 1.051, 1.036, 1.144, 1.2, 1.15, 1.304]
                 ),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0,
                #  avg_non_ignore=True
                 ),
            # dict(type='FocalLoss',loss_weight=1.0)
        ]
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole', crop_size=(512, 512), stride=(85, 85)))