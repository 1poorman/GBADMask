from typing import Dict
from torch import nn
from torch.nn import functional as F
import torch
from .fdc_loss import DC_and_CE_loss

from detectron2.utils.registry import Registry
from detectron2.layers import ShapeSpec

from adet.layers import conv_with_kaiming_uniform
from adet.config.defaults import _C

from .cbam import CBAMLayer as cbam
from .GCblock import GlobalContextBlock as gcblock

# from ..backbone.PSA import PSA_p

BASIS_MODULE_REGISTRY = Registry("BASIS_MODULE2")
BASIS_MODULE_REGISTRY.__doc__ = """
Registry for basis module, which produces global bases from feature maps.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
"""


class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


def build_basis_module2(cfg, input_shape):
    name = cfg.MODEL.BASIS_MODULE.NAME
    return BASIS_MODULE_REGISTRY.get(name)(cfg, input_shape)


@BASIS_MODULE_REGISTRY.register()
class ProtoNet(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        """
        TODO: support deconv and variable channel width
        """
        # official protonet has a relu after each conv
        super().__init__()
        # fmt: off
        mask_dim = cfg.MODEL.BASIS_MODULE.NUM_BASES
        planes = cfg.MODEL.BASIS_MODULE.CONVS_DIM
        self.in_features = cfg.MODEL.BASIS_MODULE.IN_FEATURES  # ["res2", "res3", "res4"]
        self.loss_on = cfg.MODEL.BASIS_MODULE.LOSS_ON
        norm = cfg.MODEL.BASIS_MODULE.NORM
        num_convs = cfg.MODEL.BASIS_MODULE.NUM_CONVS
        self.visualize = cfg.MODEL.BLENDMASK.VISUALIZE
        # fmt: on

        feature_channels = {k: v.channels for k, v in input_shape.items()}
        self.conv1 = nn.Conv2d(feature_channels[self.in_features[0]], 24, kernel_size=1,
                               stride=1, padding=1, bias=False)

        self.gc = gcblock(24, ratio=1 / 16)
        self.dc_ce_loss = DC_and_CE_loss()
        conv_block = conv_with_kaiming_uniform(norm, True)  # conv relu bn
        self.concat = conv_block(planes + 24, planes, 3, 1)
        self.conv2 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3,
                                             stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(planes),
                                   nn.ReLU(),
                                   nn.Conv2d(planes, mask_dim, 1))
        self.refine = nn.ModuleList()
        for in_feature in self.in_features:
            self.refine.append(conv_block(
                feature_channels[in_feature], planes, 3, 1))

        tower = []
        tower.append(gcblock(planes, ratio=1 / 16))

        for i in range(num_convs):
            tower.append(
                conv_block(planes, planes, 3, 1))
        tower.append(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        tower.append(
            conv_block(planes, planes, 3, 1))
        # tower.append(
        #     nn.Conv2d(planes, mask_dim, 1))
        self.add_module('tower', nn.Sequential(*tower))

        if self.loss_on:
            # fmt: off
            self.common_stride = cfg.MODEL.BASIS_MODULE.COMMON_STRIDE
            num_classes = cfg.MODEL.BASIS_MODULE.NUM_CLASSES + 1
            self.sem_loss_weight = cfg.MODEL.BASIS_MODULE.LOSS_WEIGHT
            # fmt: on

            inplanes = feature_channels[self.in_features[0]]
            self.seg_head = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
                                                    stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(planes),
                                          nn.ReLU(),
                                          nn.Conv2d(planes, planes, kernel_size=3,
                                                    stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(planes),
                                          nn.ReLU(),
                                          nn.Conv2d(planes, num_classes, kernel_size=1,
                                                    stride=1))

    def forward(self, features, targets=None):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.refine[i](features[f])
            else:
                x_p = self.refine[i](features[f])
                x_p = F.interpolate(x_p, x.size()[2:], mode="bilinear", align_corners=True)
                # x_p = aligned_bilinear(x_p, x.size(3) // x_p.size(3))
                x = x + x_p
        fre = self.tower(x)
        low_feat = self.conv1(features[self.in_features[0]])
        low_feat = F.interpolate(low_feat, fre.size()[2:], mode="bilinear", align_corners=True)

        x = torch.cat((self.gc(low_feat), fre), 1)
        # x = torch.cat((low_feat, fre), 1)
        x = self.concat(x)
        outputs = {"bases": [self.conv2(x)]}
        losses = {}
        # auxiliary thing semantic loss
        if self.training and self.loss_on:
            sem_out = self.seg_head(features[self.in_features[0]])
            # resize target to reduce memory
            gt_sem = targets.unsqueeze(1).float()
            gt_sem = F.interpolate(
                gt_sem, scale_factor=1 / self.common_stride)  # 1/8
            seg_loss = self.dc_ce_loss(
            #seg_loss = F.cross_entropy(
                sem_out, gt_sem.squeeze(1).long())
            losses['loss_basis_sem'] = seg_loss * self.sem_loss_weight

        elif self.visualize and hasattr(self, "seg_head"):
            outputs["seg_thing_out"] = self.seg_head(features[self.in_features[0]])
        return outputs, losses
