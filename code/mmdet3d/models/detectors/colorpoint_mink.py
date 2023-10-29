try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

from mmdet3d.models import DETECTORS, build_backbone, build_neck, build_head
from .base import Base3DDetector
import torch
from torch import nn
import numpy as np

@DETECTORS.register_module()
class ColorPointMink(Base3DDetector):
    def __init__(self,
                 backbone,
                 neck,
                 decode_head,
                 voxel_size,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(ColorPointMink, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.decode_head = build_head(decode_head)
        self.voxel_size = voxel_size

        self.geo_emb = nn.Sequential(
            nn.Linear(3,64),
        )

        self.color_emb = nn.Sequential(
            nn.Linear(3,64),
        )

        self.pos_emb = nn.Sequential(
            nn.Linear(1,64),
        )
        
        self.init_weights()

    def extract_feat(self, points):
        """Extract features from points.

        Args:
            points (list[Tensor]): Raw point clouds.

        Returns:
            SparseTensor: Voxelized point clouds.
        """
        x = self.backbone(points)
        x = self.neck(x)
        return x   
    
    def collate(self, points):
        _points = []
        for p in points:
            point_geo = p[:, :3]
            norm_coord = point_geo.clone()
            # norm_coord = point_geo - point_geo.mean(dim=0, keepdim=True)
            # norm_coord = norm_coord / torch.abs(norm_coord).max(0, keepdim=True)[0]
            _points.append(torch.cat([p, norm_coord], dim=-1))
        coordinates, features = ME.utils.batch_sparse_collate(
            [(p[:, :3] / self.voxel_size, p) for p in _points],
            device=points[0].device)
        return coordinates, features
    
    def embedding(self, point_geo, point_color, point_geo_norm):
        weak_pos = torch.norm(point_geo_norm, p=2, dim=-1, keepdim=True)
        geo_feat = self.geo_emb(point_geo)
        color_feat = self.color_emb(point_color) + self.pos_emb(weak_pos)

        return geo_feat, color_feat

    def forward_train(self, points, img_metas):
        """Forward of training.

        Args:
            points (list[Tensor]): Raw point clouds.
            gt_bboxes_3d (list[BaseInstance3DBoxes]): Ground truth
                bboxes of each sample.
            gt_labels_3d (list[torch.Tensor]): Labels of each sample.
            pts_semantic_mask (list[torch.Tensor]): Per point semantic labels
                of each sample.
            pts_instance_mask (list[torch.Tensor]): Per point instance labels
                of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            dict: Loss values.
        """

        # st+tf version
        coordinates, features = self.collate(points)

        point = ME.TensorField(
            features=features,
            coordinates=coordinates,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=points[0].device,
        )

        spoint = point.sparse()

        geo_feat, color_feat = self.embedding(spoint.features[:, :3], spoint.features[:, 3:6], spoint.features[:, 6:9])

        point_geo = ME.SparseTensor(
            features=geo_feat,
            coordinate_map_key=spoint.coordinate_map_key,
            coordinate_manager=spoint.coordinate_manager,
        )

        point_color = ME.SparseTensor(
            features=color_feat,
            coordinate_map_key=spoint.coordinate_map_key,
            coordinate_manager=spoint.coordinate_manager,
        )

        geo_sfea = self.extract_feat(point_geo)[0]
        color_sfea = self.extract_feat(point_color)[0]

        # dense loss
        geo_field = geo_sfea.slice(point)
        color_field = color_sfea.slice(point)

        geo_field = torch.stack(geo_field.decomposed_features).transpose(1,2)
        color_field = torch.stack(color_field.decomposed_features).transpose(1,2)
        gt_field = torch.stack(point.decomposed_features)[:, :, :6]

        # print('field', geo_field.shape, color_field.shape, gt_field.shape)

        losses = self.decode_head.losses(geo_field, color_field, gt_field)
        return losses

    def simple_test(self, points, img_metas, img, pts_semantic_mask, rescale=True):
        """Test without augmentations.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            list[dict]: Predicted 3d instances.
        """
        return []
    
    def aug_test(self, points, img_metas, **kwargs):
        """Test with augmentations.

        Args:
            points (list[list[torch.Tensor]]): Points of each sample.
            img_metas (list[dict]): Contains scene meta infos.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        raise NotImplementedError
