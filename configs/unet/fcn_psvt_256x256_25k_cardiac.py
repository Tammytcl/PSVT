_base_ = [
    '../_base_/models/psvt.py','../_base_/datasets/cardiac.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_25k.py'
]
crop_size = (256,256)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
