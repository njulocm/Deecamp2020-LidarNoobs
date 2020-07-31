import itertools
import logging

from det3d.builder import build_box_coder
from det3d.utils.config_tool import get_downsample_factor

# norm_cfg = dict(type='SyncBN', eps=1e-3, momentum=0.01)
norm_cfg = None

tasks = [
    dict(num_class=1, class_names=["Car"]),
    dict(num_class=1, class_names=["Truck"]),
    dict(num_class=1, class_names=["Tricar"]),
    dict(num_class=1, class_names=['Cyclist']),
    dict(num_class=1, class_names=['Pedestrian'])
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    type="iou",
    anchor_generators=[
        dict(
            type="anchor_generator_range",
            sizes=[4.6, 1.8, 1.4],
            anchor_ranges=[-84.0, -42.0, 1.0, 84.0, 42.0, 1.0],
            rotations=[0, 1.57],
            matched_threshold=0.6,
            unmatched_threshold=0.45,
            class_name="Car",
        ),
        dict(
            type="anchor_generator_range",
            sizes=[13.0, 3.0, 3.4],
            anchor_ranges=[-84.0, -42.0, 1.0, 84.0, 42.0, 1.0],
            rotations=[0, 1.57],
            matched_threshold=0.6,
            unmatched_threshold=0.45,
            class_name="Truck",
        ),
        dict(
            type="anchor_generator_range",
            sizes=[3.0, 1.5, 1.7],
            anchor_ranges=[-84.0, -42.0, 1.0, 84.0, 42.0, 1.0],
            rotations=[0, 1.57],
            matched_threshold=0.4,
            unmatched_threshold=0.2,
            class_name="Tricar",
        ),
        dict(
            type="anchor_generator_range",
            sizes=[1.7, 0.7, 1.6],
            anchor_ranges=[-84.0, -42.0, 1.0, 84.0, 42.0, 1.0],
            rotations=[0, 1.57],
            matched_threshold=0.4,
            unmatched_threshold=0.2,
            class_name="Cyclist",
        ),
        dict(
            type="anchor_generator_range",
            sizes=[0.8, 0.8, 1.56],
            anchor_ranges=[-84.0, -42.0, 1.0, 84.0, 42.0, 1.0],
            rotations=[0, 1.57],
            matched_threshold=0.4,
            unmatched_threshold=0.2,
            class_name="Pedestrian",
        ),
    ],
    sample_positive_fraction=-1,
    sample_size=512,
    region_similarity_calculator=dict(type="nearest_iou_similarity", ),
    pos_area_threshold=-1,
    tasks=tasks,
)

box_coder = dict(
    type="ground_box3d_coder", n_dim=7, linear_dim=False, encode_angle_vector=False,
)

# model settings
model = dict(
    type="VoxelNet",

    reader=dict(
        type="VoxelFeatureExtractorV3",
        # type='SimpleVoxel',
        num_input_features=4,
        norm_cfg=norm_cfg,
    ),
    backbone=dict(
        type="SpMiddleFHD", num_input_features=4, ds_factor=8, norm_cfg=norm_cfg,
    ),
    neck=dict(
        type="RPN",
        layer_nums=[5, ],
        ds_layer_strides=[1, ],
        ds_num_filters=[128, ],
        us_layer_strides=[1, ],
        us_num_filters=[128, ],
        num_input_features=128,
        norm_cfg=norm_cfg,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        # type='RPNHead',
        type="MultiGroupHead",
        mode="3d",
        in_channels=sum([128, ]),
        norm_cfg=norm_cfg,
        tasks=tasks,
        weights=[1, ],
        box_coder=build_box_coder(box_coder),
        encode_background_as_zeros=True,
        loss_norm=dict(  # 这边用的是默认，什么含义暂时不知道
            type="NormByNumPositives", pos_cls_weight=1.0, neg_cls_weight=1.0,
        ),
        loss_cls=dict(type="SigmoidFocalLoss", alpha=0.25, gamma=2.0, loss_weight=1.0, ),
        # Focal loss down-weights well classified examples and focusses on the hard examples. 
        # See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.

        use_sigmoid_score=True,
        loss_bbox=dict(
            type="WeightedSmoothL1Loss",
            sigma=3.0,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            codewise=True,
            loss_weight=2.0,
        ),
        encode_rad_error_by_sin=True,
        loss_aux=dict(
            type="WeightedSoftmaxClassificationLoss",
            name="direction_classifier",
            loss_weight=0.2,
        ),
        direction_offset=0.0,
    ),

    pretrained=None,
)

assigner = dict(  # train_pipeline 的 AssignTarget的cfg
    box_coder=box_coder,
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    debug=False,
)

train_cfg = dict(assigner=assigner)

test_cfg = dict(
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=100,
        nms_iou_threshold=0.01,
    ),
    score_threshold=0.3,
    post_center_limit_range=[-84.48, -42.24, -1.0, 84.48, 42.24, 3.0],
    max_per_img=100,
)

# dataset settings
dataset_type = "KittiDataset"
data_root = "/home/ma-user/work/workspace/DeepCamp_Lidar"

train_preprocessor = dict(  # train_pipeline中preprocess的cfg
    mode="train",
    shuffle_points=True,
    gt_loc_noise=[1.0, 1.0, 0.5],
    gt_rot_noise=[-0.785, 0.785],
    global_rot_noise=[-0.785, 0.785],
    global_scale_noise=[0.95, 1.05],
    global_rot_per_obj_range=[0, 0],
    global_trans_noise=[0.0, 0.0, 0.0],
    remove_points_after_sample=True,
    gt_drop_percentage=0.0,
    gt_drop_max_keep_points=15,
    remove_unknown_examples=False,
    remove_environment=False,
    class_names=class_names,


    db_sampler=dict(
        type="GT-AUG",
        enable=True,
        db_info_path="/home/ma-user/work/workspace/DeepCamp_Lidar/dee_dbinfos_train.pkl",
        sample_groups=[dict(Car=10), dict(Truck=20), dict(Tricar=20), dict(Cyclist=20), dict(Pedestrian=20)],
        db_prep_steps=[
            dict(filter_by_min_num_points=dict(Car=5, Truck=5, Tricar=5, Cyclist=5, Pedestrian=5)),
            dict(filter_by_difficulty=[-1], ),
        ],
        global_random_rotation_range_per_object=[0, 0],
        rate=1.0,
    )
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    remove_environment=False,
    remove_unknown_examples=False,
)

voxel_generator = dict(  # train_pipeline的Voxelization的cfg
    range=[-84.48, -42.24, -1.0, 84.48, 42.24, 3.0],
    voxel_size=[0.08, 0.08, 0.1], # NEW size
    max_points_in_voxel=20,
    max_voxel_num=140000,
)

train_pipeline = [  # 用于构建dataset的__getitem__方法，也就是每个数据先过这个pipeline进行处理
    dict(type="LoadPointCloudFromFile"),  # 主要是读取point数据，放入res["lidar"]["points"]
    dict(type="LoadPointCloudAnnotations", with_bbox=True),  # 主要是读取annos数据，包括calib、locs、dims、rots、name、bbox
    dict(type="Preprocess", cfg=train_preprocessor),  # 对points进行预处理
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignTarget", cfg=train_cfg["assigner"]),  # 这边应该是处理框的
    dict(type="Reformat"),
    # dict(type='PointCloudCollect', keys=['points', 'voxels', 'annotations', 'calib']),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile"),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignTarget", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

train_anno = "/home/ma-user/work/workspace/DeepCamp_Lidar/kitti_infos_train.pkl"
val_anno = "/home/ma-user/work/workspace/DeepCamp_Lidar/kitti_infos_val.pkl"
test_anno = "/home/ma-user/work/workspace/DeepCamp_Lidar/video_infos.pkl"

data = dict(
    samples_per_gpu=8,  # 这个参数与batch_size有关，batch_size = num_gpus*samples_per_gpu ，在build_loader中发现的，
    workers_per_gpu=8,  # 这个参数决定着有几个进程来处理data_loading
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        class_names=class_names,
        pipeline=train_pipeline,  # 在data_loader里面要用到
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        ann_file=val_anno,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)

# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
"""training hooks """
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy in training hooks
lr_config = dict(
    type="one_cycle", lr_max=0.001, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 20
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = "/data/Outputs/det3d_Outputs/SECOND"
load_from = None
resume_from = None
workflow = [("train", 2), ("val", 1)]  # Trainer.run()里面用到，用来控制每一轮是train还是val
