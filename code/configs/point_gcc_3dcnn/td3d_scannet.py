voxel_size = .02
n_points = 100000

model = dict(
    type='ColorPointMink',
    voxel_size=voxel_size,
    backbone=dict(type='MinkResNet', in_channels=64, depth=34, norm='batch', return_stem=True, stride=1),
    neck=dict(
        type='NgfcTinySegmentationNeck',
        in_channels=(64, 128, 256, 512),
        out_channels=128,
        feat_channels=32,),
    decode_head=dict(
        type='ColorPointHead',
        channels=32,
        hidden_dim=128,
        dropout_ratio=0.5,
        num_classes=20,
        pseudo_obj_thr=2.0,
        temperature=0.4,
        epsilon=0.05,
    ),
    train_cfg=dict(num_rois=2),
    test_cfg=dict(
        gt_classes=20,
        nms_pre=1200,
        iou_thr=.4,
        score_thr=.1,
        binary_score_thr=0.2))

# runtime settings
checkpoint_config = dict(interval=50, max_keep_ckpts=10)
runner = dict(type='EpochBasedRunner', max_epochs=200)
lr_config = dict(policy='CosineAnnealing', warmup=None, min_lr=1e-5)

optimizer = dict(type='AdamW', lr=1e-3, weight_decay=0.0001)

optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

dataset_type = 'ScanNetDataset'
data_root = './data/scannet/'
class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D'),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(type='PointSample', num_points=100000),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-.02, .02],
        scale_ratio_range=[.9, 1.1],
        translation_std=[.1, .1, .1],
        shift_height=False),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='NormalizePointsColor', color_mean=None),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=15,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'scannet_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=False,
            classes=class_names,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))