_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/coco-stuff10k.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 512)
preprocess_cfg = dict(size=crop_size)
model = dict(
    preprocess_cfg=preprocess_cfg,
    decode_head=dict(num_classes=171),
    auxiliary_head=dict(num_classes=171))
