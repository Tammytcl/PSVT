_base_ = [
    '../_base_/models/fcn_r50-d8.py',
    '../_base_/datasets/cityscapes_769x769.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
crop_size = (769, 769)
preprocess_cfg = dict(size=crop_size)
model = dict(
    preprocess_cfg=preprocess_cfg,
    backbone=dict(dilations=(1, 1, 1, 2), strides=(1, 2, 2, 1)),
    decode_head=dict(align_corners=True, dilation=6),
    auxiliary_head=dict(align_corners=True, dilation=6),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))
