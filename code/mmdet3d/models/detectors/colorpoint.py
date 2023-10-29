# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from mmseg.core import add_prefix
from mmdet3d.models import DETECTORS, build_backbone, build_neck, build_head
from .base import Base3DDetector


@DETECTORS.register_module()
class ColorPoint(Base3DDetector):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 loss_regularization=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(ColorPoint, self).__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        self._init_decode_head(decode_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg


        self.geo_emb = nn.Sequential(
            nn.Linear(3,64),
        )

        self.color_emb = nn.Sequential(
            nn.Linear(3,64),
        )

        self.pos_emb = nn.Sequential(
            nn.Linear(1,64),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = build_head(decode_head)
        self.num_classes = self.decode_head.num_classes

    def extract_feat(self, points):
        """Extract features from points."""
        x = self.backbone(points)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, points, img_metas):
        """Forward function for training.

        Args:
            points (list[torch.Tensor]): List of points of shape [N, C].
            img_metas (list): Image metas.
            pts_semantic_mask (list[torch.Tensor]): List of point-wise semantic
                labels of shape [N].

        Returns:
            dict[str, Tensor]: Losses.
        """
        point_geo = [p[:, :3] for p in points]
        point_color = [p[:, 3:] for p in points]
        point_geo = torch.stack(point_geo)
        point_color = torch.stack(point_color)

        norm_coord = point_geo.clone()
        # norm_coord = point_geo - point_geo.mean(dim=1, keepdim=True)
        # norm_coord = norm_coord / torch.abs(norm_coord).max(1, keepdim=True)[0]
        weak_pos = torch.norm(norm_coord, p=2, dim=2, keepdim=True)

        feat_geo = self.extract_feat(torch.cat([point_geo, self.geo_emb(point_geo)], dim=-1))['fp_features'][-1]
        feat_color = self.extract_feat(torch.cat([point_geo, self.color_emb(point_color)+self.pos_emb(weak_pos)], dim=-1))['fp_features'][-1]
        losses = dict()

        loss_decode = self.decode_head.losses(feat_geo, feat_color, torch.stack(points))
        losses.update(loss_decode)
        return losses

    def simple_test(self, points, img_metas, img, pts_semantic_mask, rescale=True):
        return []

    
    def aug_test(self, points, img_metas, rescale=True):
        return []