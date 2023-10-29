try:
    import MinkowskiEngine as ME
    from MinkowskiEngine.modules.resnet_block import BasicBlock
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

from torch import nn

from mmcv.runner import BaseModule
from mmdet3d.models.builder import NECKS

@NECKS.register_module()
class NgfcTinySegmentationNeck(BaseModule):
    def __init__(self, in_channels, out_channels, feat_channels=128, extra_stride=None, generative=False):
        super(NgfcTinySegmentationNeck, self).__init__()
        self.extra_stride = extra_stride

        self._init_layers(in_channels, out_channels, generative)

        # conv = ME.MinkowskiGenerativeConvolutionTranspose if generative \
        # else ME.MinkowskiConvolutionTranspose
        
        self.upsample_st_4 = nn.Sequential(
                        ME.MinkowskiConvolutionTranspose(
                            128,
                            64,
                            kernel_size=3,
                            stride=4,
                            dimension=3),
                        ME.MinkowskiBatchNorm(64),
                        ME.MinkowskiReLU(inplace=True))
        
        self.conv_32_ch = nn.Sequential(
                        ME.MinkowskiConvolution(
                            64,
                            feat_channels,
                            kernel_size=3,
                            stride=1,
                            dimension=3),
                        ME.MinkowskiBatchNorm(feat_channels),
                        ME.MinkowskiReLU(inplace=True))
                        
        if self.extra_stride != None:
            self.convtr = ME.MinkowskiConvolutionTranspose(
                            64,
                            64,
                            kernel_size=3,
                            stride=extra_stride,
                            dimension=3)

    def _init_layers(self, in_channels, out_channels, generative):
        for i in range(len(in_channels)):
            if i > 0:
                self.__setattr__(
                    f'up_block_{i}',
                    make_up_block(in_channels[i], in_channels[i - 1], generative=generative))
            if i < len(in_channels) - 1:
                self.__setattr__(
                    f'lateral_block_{i}',
                    make_block(in_channels[i], in_channels[i]))
        self.__setattr__(
            f'out_block',
            make_block(in_channels[0], out_channels))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x):
        feats_st_2 = x[0]
        inputs = x[1:]
        x = inputs[-1]
        for i in range(len(inputs) - 1, -1, -1):
            if i < len(inputs) - 1:
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                x = inputs[i] + x
                x = self.__getattr__(f'lateral_block_{i}')(x)
        out = self.__getattr__(f'out_block')(x)

        seg_feats = self.upsample_st_4(out) + feats_st_2

        if self.extra_stride != None:
            seg_feats = self.convtr(seg_feats)

        seg_feats = self.conv_32_ch(seg_feats)
        return [seg_feats]

def make_block(in_channels, out_channels, kernel_size=3):
    return nn.Sequential(
        ME.MinkowskiConvolution(in_channels, out_channels,
                                kernel_size=kernel_size, dimension=3),
        ME.MinkowskiBatchNorm(out_channels),
        ME.MinkowskiReLU(inplace=True))


def make_down_block(in_channels, out_channels):
    return nn.Sequential(
        ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3,
                                stride=2, dimension=3),
        ME.MinkowskiBatchNorm(out_channels),
        ME.MinkowskiReLU(inplace=True))


def make_up_block(in_channels, out_channels, generative=False):
    conv = ME.MinkowskiGenerativeConvolutionTranspose if generative \
        else ME.MinkowskiConvolutionTranspose
    return nn.Sequential(
        conv(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            dimension=3),
        ME.MinkowskiBatchNorm(out_channels),
        ME.MinkowskiReLU(inplace=True))
