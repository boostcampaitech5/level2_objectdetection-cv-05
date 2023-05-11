# 변수 조정
dr = ['/opt/ml/dataset', '/opt/ml/HS_aug_dataset']
DATA_DIR = dr[0]
# '/opt/ml/train_new.json'
TRAIN_JSON_DIR = '/opt/ml/train_new.json'
VAL_JSON_DIR = '/opt/ml/val_new.json'
TEST_JSON_DIR = '/opt/ml/dataset/test.json'


# dataset settings
dataset_type = 'CocoDataset'
data_root = DATA_DIR ## data set 위치 
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing") ## class 정의

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
#############
albu_train_transforms = [
    dict(
        type='HueSaturationValue',
        hue_shift_limit = 20,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=0.8),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]
######
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512,512), keep_ratio=True), ## image size 변경 
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
###################

    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512,512), ## image size 변경 
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4, ## gpu당 batch사이즈 몇으로 할건지 , 2->4 
    workers_per_gpu=2, # data loader 를 만들때 worker개수 선언해주는 것과 동일
    train=dict(
        type=dataset_type,
        ann_file=TRAIN_JSON_DIR, ## train annotation file 위치
        img_prefix=data_root, ## data root 위치
        classes = classes, ## classes 추가
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=VAL_JSON_DIR, ## validation annotation file 위치
        img_prefix=data_root, ## data root 위치
        classes = classes, ## classes 추가
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=TEST_JSON_DIR, ## test annotation file 위치
        img_prefix=data_root , ## data root 위치
        classes = classes, ## classes 추가
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')