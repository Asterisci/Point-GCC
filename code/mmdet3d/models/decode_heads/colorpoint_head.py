# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn.bricks import ConvModule
from torch import nn as nn

from mmdet3d.ops import PointFPModule
from ..builder import HEADS, build_loss
from mmcv.runner import BaseModule, auto_fp16, force_fp32
import torch.nn.functional as F

import torch.distributed as dist

@HEADS.register_module()
class ColorPointHead(BaseModule):
    r"""PointNet2 decoder head.

    Decoder head used in `PointNet++ <https://arxiv.org/abs/1706.02413>`_.
    Refer to the `official code <https://github.com/charlesq34/pointnet2>`_.

    Args:
        fp_channels (tuple[tuple[int]]): Tuple of mlp channels in FP modules.
        fp_norm_cfg (dict): Config of norm layers used in FP modules.
            Default: dict(type='BN2d').
    """

    def __init__(self,
                 channels,
                 pseudo_obj_thr,
                 temperature,
                 epsilon,
                 num_classes=20,
                 dropout_ratio=0.5,
                 hidden_dim=128,
                 mse_loss=dict(type='MSELoss', loss_weight=100.0),
                 nce_loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 **kwargs):
        super(ColorPointHead, self).__init__(**kwargs)

        self.num_classes = num_classes
        self.temperature = temperature
        self.pseudo_obj_thr = pseudo_obj_thr / self.num_classes
        self.channels = channels

        self.epsilon = epsilon
        self.sinkhorn_iterations = 3

        self.geo_proj = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
        )

        self.color_proj = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
        )

        self.geo_rebuild_head = nn.Linear(hidden_dim, 3)
        
        self.color_rebuild_head = nn.Linear(hidden_dim, 3)

        self.geo_prot = nn.Linear(hidden_dim, num_classes, bias=False)

        self.color_prot = nn.Linear(hidden_dim, num_classes, bias=False)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)
        else:
            self.dropout = None

        self.geo_mse_loss = build_loss(mse_loss)
        self.color_mse_loss = build_loss(mse_loss)
        self.point_nce_loss = build_loss(nce_loss)
        self.obj_nce_loss = build_loss(nce_loss)

    def forward_test(self, feat_geo, feat_color):
        """Forward pass.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            torch.Tensor: Segmentation map of shape [B, num_classes, N].
        """

        feat_geo = self.geo_proj(feat_geo).transpose(1,2)
        feat_color = self.color_proj(feat_color).transpose(1,2)

        pseudo_geo = nn.functional.normalize(feat_geo, dim=-1, p=2)
        pseudo_geo = self.geo_prot(pseudo_geo)

        pseudo_color = nn.functional.normalize(feat_color, dim=-1, p=2)
        pseudo_color = self.color_prot(pseudo_color)

        pseudo_mean = (pseudo_geo + pseudo_color)/2

        # #FIXME: debug
        # pseudo_geo_dist = pseudo_geo.softmax(dim=-1)
        # pseudo_color_dist = pseudo_color.softmax(dim=-1)
        # pseudo_mean_dist = pseudo_mean.softmax(dim=-1)
        # pseudo_geo_sco, pseudo_geo_label = pseudo_geo_dist.max(-1)
        # pseudo_color_sco, pseudo_color_label = pseudo_color_dist.max(-1)
        # pseudo_mean_sco, pseudo_mean_label = pseudo_mean_dist.max(-1)

        # geo_count = []
        # color_count = []
        # mean_count = []

        # for i in range(self.num_classes):
        #     geo_corr_mat_all = (pseudo_geo_label == i)
        #     color_corr_mat_all = (pseudo_color_label == i)
        #     mean_corr_mat_all = (pseudo_mean_label == i)

        #     geo_count.append(geo_corr_mat_all.nonzero().shape[0])
        #     color_count.append(color_corr_mat_all.nonzero().shape[0])
        #     mean_count.append(mean_corr_mat_all.nonzero().shape[0])

        # print('geo_count', geo_count)
        # print('color_count', color_count)
        # print('mean_count', mean_count)

        return pseudo_mean
    
    def losses(self, feat_geo, feat_color, points):
        """Compute semantic segmentation loss.

        Args:
            seg_logit (torch.Tensor): Predicted per-point segmentation logits
                of shape [B, num_classes, N].
            seg_label (torch.Tensor): Ground-truth segmentation label of
                shape [B, N].
        """
        if self.dropout is not None:
            feat_geo = self.dropout(feat_geo)
            feat_color = self.dropout(feat_color)
        feat_geo = self.geo_proj(feat_geo).transpose(1,2)
        feat_color = self.color_proj(feat_color).transpose(1,2)

        loss = dict()

        pred_geo = self.geo_rebuild_head(feat_color)
        pred_color = self.color_rebuild_head(feat_geo)

        loss.update(self._mse_loss(pred_geo, pred_color, points))
        loss.update(self._nce_loss(feat_geo, feat_color))

        pseudo_geo = nn.functional.normalize(feat_geo, dim=-1, p=2)
        pseudo_geo = self.geo_prot(pseudo_geo)

        pseudo_color = nn.functional.normalize(feat_color, dim=-1, p=2)
        pseudo_color = self.color_prot(pseudo_color)

        loss.update(self._pseudo_loss(pseudo_geo, pseudo_color, feat_geo, feat_color))
        return loss
    
    def _mse_loss(self, pred_geo, pred_color, targets):
        coord = targets[:, :, :3]
        # print('coord', coord.max(1)[0].shape, coord.max(1)[0][0], coord.max(1, keepdim=True)[0].shape)
        # mae norm :img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        norm_coord = coord - coord.mean(dim=1, keepdim=True)
        norm_coord = norm_coord / torch.abs(norm_coord).max(1, keepdim=True)[0]
        # print('norm_coord', norm_coord.shape, norm_coord.max(1)[0][0])
        geo_mse_loss = self.geo_mse_loss(pred_geo, norm_coord)
        color_mse_loss = self.color_mse_loss(pred_color, targets[:, :, 3:])

        return dict(geo_mse_loss=geo_mse_loss,
                    color_mse_loss=color_mse_loss)
    
    def _nce_loss(self, feat_geo, feat_color):
        B, N, C = feat_geo.shape
        feat_geo_norm = F.normalize(feat_geo.clone(), dim=2)
        feat_color_norm = F.normalize(feat_color.clone(), dim=2)

        pick_num = 2048
        pick_index = torch.randperm(N)[:pick_num]
        feat_geo_norm = feat_geo_norm[:, pick_index, :]
        feat_color_norm = feat_color_norm[:, pick_index, :]

        logits = torch.bmm(feat_geo_norm, feat_color_norm.transpose(-1, -2)) # npos by npos
        labels = torch.arange(pick_num).cuda().long().unsqueeze(0).expand(B, -1)
        out = torch.div(logits, self.temperature)
        nce_loss = self.point_nce_loss(out.reshape(-1, pick_num), labels.reshape(-1))
        return dict(nce_loss=nce_loss)

    def _pseudo_loss(self, pseudo_geo, pseudo_color, feat_geo, feat_color):

        # normalize the prototypes
        with torch.no_grad():
            self.geo_prot.weight.data = F.normalize(self.geo_prot.weight.data, dim=-1, p=2)
            self.color_prot.weight.data = F.normalize(self.color_prot.weight.data, dim=-1, p=2)

        pseudo_geo_reshape = pseudo_geo.reshape(-1, self.num_classes)
        pseudo_color_reshape = pseudo_color.reshape(-1, self.num_classes)

        # print('pseudo_geo_reshape', pseudo_geo_reshape.shape, pseudo_color_reshape.shape)
        q_geo = self.distributed_sinkhorn(pseudo_geo_reshape)
        q_color = self.distributed_sinkhorn(pseudo_color_reshape)

        # p_geo = torch.div(pseudo_geo_reshape, self.temperature)
        # p_color = torch.div(pseudo_color_reshape, self.temperature)
        p_geo = torch.div(pseudo_geo_reshape, 0.1)
        p_color = torch.div(pseudo_color_reshape, 0.1)

        swap_loss = -50 * torch.mean(q_geo * F.log_softmax(p_color, dim=-1)+ q_color * F.log_softmax(p_geo, dim=-1))

        # obj nce loss
        pseudo_geo_dist = pseudo_geo.softmax(dim=-1)
        pseudo_color_dist = pseudo_color.softmax(dim=-1)
        pseudo_geo_sco, pseudo_geo_label = pseudo_geo_dist.max(-1)
        pseudo_color_sco, pseudo_color_label = pseudo_color_dist.max(-1)

        pseudo_geo_mat = pseudo_geo_sco > self.pseudo_obj_thr
        pseudo_color_mat = pseudo_color_sco > self.pseudo_obj_thr

        # #FIXME: DEBUG
        # q_geo_dist = q_geo.softmax(dim=-1)
        # q_color_dist = q_color.softmax(dim=-1)
        # q_geo_sco, q_geo_label = q_geo_dist.max(-1)
        # q_color_sco, q_color_label = q_color_dist.max(-1)

        # print('softmax', pseudo_geo_sco.max(-1)[0], pseudo_color_sco.max(-1)[0], pseudo_geo_sco.min(-1)[0], pseudo_color_sco.min(-1)[0])

        pseudo_geo_sco_obj = pseudo_geo_sco[pseudo_geo_mat]
        pseudo_color_sco_obj = pseudo_color_sco[pseudo_color_mat]

        pseudo_geo_label_obj = pseudo_geo_label[pseudo_geo_mat]
        pseudo_color_label_obj = pseudo_color_label[pseudo_color_mat]

        pseudo_geo_feat = feat_geo[pseudo_geo_mat]
        pseudo_color_feat = feat_color[pseudo_color_mat]

        geo_samples_avg = []
        color_samples_avg = []
        # geo_samples_inner = []
        # geo_samples_inner_center = []
        # color_samples_inner = []
        # color_samples_inner_center = []

        # #FIXME: debug
        # geo_count = []
        # color_count = []
        # q_geo_count = []
        # q_color_count = []
        # geo_count_obj = []
        # color_count_obj = []

        for i in range(self.num_classes):
            geo_corr_mat = (pseudo_geo_label_obj == i)
            color_corr_mat = (pseudo_color_label_obj == i)

            if geo_corr_mat.nonzero().shape[0] > 0 and color_corr_mat.nonzero().shape[0] > 0:

                # mean
                curr_geo_avg = torch.mean(pseudo_geo_feat[geo_corr_mat], dim=0)
                curr_color_avg = torch.mean(pseudo_color_feat[color_corr_mat], dim=0)

                geo_samples_avg.append(curr_geo_avg)
                color_samples_avg.append(curr_color_avg)

        if len(geo_samples_avg) > 0 and len(color_samples_avg) > 0:
            geo_samples_avg = torch.stack(geo_samples_avg)
            color_samples_avg = torch.stack(color_samples_avg)

            feat_geo_norm = F.normalize(geo_samples_avg.clone(), dim=-1)
            feat_color_norm = F.normalize(color_samples_avg.clone(), dim=-1)
            
            logits = torch.mm(feat_geo_norm, feat_color_norm.transpose(1, 0)) # npos by npos
            labels = torch.arange(logits.shape[0]).cuda().long()
            out = torch.div(logits, self.temperature)

            obj_nce_loss_avg = self.obj_nce_loss(out, labels)
        else:
            obj_nce_loss_avg = swap_loss.new([0]).squeeze()

        return dict(swap_loss=swap_loss,
                    obj_nce_loss=obj_nce_loss_avg)

    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        Q = torch.exp(out / self.epsilon).t() # Q is K-by-B for consistency with notations from our paper
        world_size = torch.distributed.get_world_size()
        # print('world_size', world_size)
        B = Q.shape[1] * world_size # number of samples to assign
        K = Q.shape[0] # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(self.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

