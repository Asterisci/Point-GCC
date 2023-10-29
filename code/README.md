# Point-GCC: Universal Self-supervised 3D Scene Pre-training via Geometry-Color Contrast

## Installation
This implementation is based on [mmdetection3d v1.0](https://github.com/open-mmlab/mmdetection3d/tree/1.0) framework.
Please refer to the original installation guide [getting_started.md](https://github.com/open-mmlab/mmdetection3d/blob/1.0/docs/en/getting_started.md), including MinkowskiEngine installation.
## Getting Started

Please refer to the original guide [getting_started.md](https://github.com/open-mmlab/mmdetection3d/blob/1.0/docs/en/getting_started.md) for basic usage examples and data preparation for [scannet](https://github.com/open-mmlab/mmdetection3d/blob/1.0/docs/en/datasets/scannet_det.md), [sunrgbd](https://github.com/open-mmlab/mmdetection3d/blob/1.0/docs/en/datasets/sunrgbd_det.md), and [s3dis](https://github.com/open-mmlab/mmdetection3d/blob/1.0/docs/en/datasets/s3dis_sem_seg.md).

## Usage

### Pre-train

**3DCNN**

To start pre-training, run [train](tools/train.py) with 3DCNN backbone such as for [TD3D](configs/point_gcc_3dcnn/td3d_scannet.py):
```shell
./tools/dist_train.sh configs/point_gcc_3dcnn/td3d_scannet.py 4 --no-validate
```

**PointNet**

To start pre-training, run [train](tools/train.py) with PointNet backbone such as for [VoteNet](configs/point_gcc_pointnet/votenet-scannet.py):
```shell
./tools/dist_train.sh configs/point_gcc_pointnet/votenet-scannet.py 4 --no-validate
```

### Fine-tuning

**VoteNet**

To start fine-tuning, modify the `load_from` field to your pretrain model path, and run [train](tools/train.py) with [VoteNet](configs/votenet/votenet_8x8_scannet-3d-18class-fine.py) for object detection:
```shell
./tools/dist_train.sh configs/votenet/votenet_8x8_scannet-3d-18class-fine.py 8
```

**GroupFree3D**

To start fine-tuning, modify the `load_from` field to your pretrain model path, and run [train](tools/train.py) with [GroupFree3D](configs/groupfree3d/groupfree3d_8x4_scannet-3d-18class-L6-O256-fine.py) for object detection:
```shell
./tools/dist_train.sh configs/groupfree3d/groupfree3d_8x4_scannet-3d-18class-L6-O256-fine.py 4
```

**PointNet++(SSG)**

To start fine-tuning, modify the `load_from` field to your pretrain model path, and run [train](tools/train.py) with [PointNet++(SSG)](configs/pointnet2/pointnet2_ssg_16x2_cosine_200e_scannet_seg-3d-20class.py) for semantic segmentation:
```shell
./tools/dist_train.sh configs/pointnet2/pointnet2_ssg_16x2_cosine_200e_scannet_seg-3d-20class.py 2
```

**TR3D and TD3D**

To start fine-tuning with [TR3D](https://github.com/SamsungLabs/tr3d) and [TD3D](https://github.com/SamsungLabs/td3d), please follow the official repo and modify the `load_from` field to your pretrain model path in corresponding config file.