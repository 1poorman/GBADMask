import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import DropPath

from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.layers import Conv2d, ShapeSpec
from torch.nn import BatchNorm2d

'''
@article{han2022vision,
  title={Vision GNN: An Image is Worth Graph of Nodes},
  author={Han, Kai and Wang, Yunhe and Guo, Jianyuan and Tang, Yehui and Wu, Enhua},
  journal={arXiv preprint arXiv:2206.00272},
  year={2022}
}
'''
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool

# def _cfg(url='', **kwargs):
#     return {
#         'url': url,
#         'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
#         'crop_pct': .9, 'interpolation': 'bicubic',
#         'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 
#         'classifier': 'head',
#         **kwargs
#     }


# default_cfgs = {
#     'mobilevig': _cfg(crop_pct=0.9, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
# }
    
class Stem(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.GELU):
        super(Stem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim // 2),
            nn.GELU(),
            nn.Conv2d(output_dim // 2, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU()   
        )
        
    def forward(self, x):
        return self.stem(x)
    
class MLP(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, drop=0., mid_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mid_conv = mid_conv
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

        if self.mid_conv:
            self.mid = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features)
            self.mid_norm = nn.BatchNorm2d(hidden_features)

        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.norm2 = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)

        if self.mid_conv:
            x_mid = self.mid(x)
            x_mid = self.mid_norm(x_mid)
            x = self.act(x_mid)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.norm2(x)

        x = self.drop(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, mid_conv=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.mlp(x))
        return x


class MRConv4d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    
    K is the number of superpatches, therefore hops equals res // K.
    """
    def __init__(self, in_channels, out_channels, K=2):
        super(MRConv4d, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU()
            )
        self.K = K

    def forward(self, x):
        B, C, H, W = x.shape
            
        x_j = x - x
        for i in range(self.K, H, self.K):
            x_c = x - torch.cat([x[:, :, -i:, :], x[:, :, :-i, :]], dim=2)
            x_j = torch.max(x_j, x_c)
        for i in range(self.K, W, self.K):
            x_r = x - torch.cat([x[:, :, :, -i:], x[:, :, :, :-i]], dim=3)
            x_j = torch.max(x_j, x_r)

        x = torch.cat([x, x_j], dim=1)
        return self.nn(x)


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, drop_path=0.0, K=2):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.K = K

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = MRConv4d(in_channels, in_channels * 2, K=self.K)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )  # out_channels back to 1x}
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

       
    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp

        return x


class Downsample(nn.Module):
    """ Convolution-based downsample
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()        
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features # same as input
        hidden_features = hidden_features or in_features # x4
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = nn.GELU()
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x
    
def swish(x):
    return x * x.sigmoid()

class BN_Conv2d_swish(nn.Module):
    """
    BN_CONV_LeakyRELU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False) -> object:
        super(BN_Conv2d_swish, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return swish(self.seq(x))
    
class BN_Conv2d_Leaky(nn.Module):
    """
    BN_CONV_LeakyRELU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False) -> object:
        super(BN_Conv2d_Leaky, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return F.leaky_relu(self.seq(x))
    
class BN_Conv2d_act(nn.Module):
    """
    BN_CONV_LeakyRELU
    """ 

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False) -> object:
        super(BN_Conv2d_act, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish()
        )

    def forward(self, x):
        return self.seq(x)


class MobileViG(Backbone):
    def __init__(self,  local_blocks, local_channels,
                 global_blocks, global_channels,
                 dropout=0., drop_path=0., emb_dims=512,
                 K=2, distillation=True, num_classes=1000,
                  out_indices=None):
        super(MobileViG, self).__init__()

        self.distillation = distillation
        self.out_indices = out_indices
        
        # global stage 里每个 block 由 Grapher + FFN 两个子模块组成，
        # 各自占用一个 drop rate，因此按 2 倍计数（与下方 dpr_idx += 2 对应）。
        n_blocks = sum(local_blocks) + 2 * sum(global_blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path, n_blocks)]  # stochastic depth decay rule 
        dpr_idx = 0

        self.stem = Stem(input_dim=3, output_dim=local_channels[0])

        local_backbone1 = []
        for _ in range(local_blocks[0]):
            local_backbone1.append(InvertedResidual(dim=local_channels[0]//2, mlp_ratio=4, drop_path=dpr[dpr_idx]))
            dpr_idx += 1 
        local_backbone1.append(BN_Conv2d_act(local_channels[0]//2, local_channels[0]//2, 1, 1,0))
        self.add_module('local_backbone1', nn.Sequential(*local_backbone1))

        self.transition1 = BN_Conv2d_act(local_channels[0], local_channels[0], 1, 1,0)
        self.down1 = Downsample(local_channels[0], local_channels[1])

        local_backbone2 = []
        for _ in range(local_blocks[1]):
            local_backbone2.append(InvertedResidual(dim=local_channels[1]//2, mlp_ratio=4, drop_path=dpr[dpr_idx]))
            dpr_idx += 1 
        local_backbone2.append(BN_Conv2d_act(local_channels[1]//2, local_channels[1]//2, 1, 1,0))
        self.add_module('local_backbone2', nn.Sequential(*local_backbone2))

        self.transition2 = BN_Conv2d_act(local_channels[1], local_channels[1], 1, 1,0)
        self.down2 = Downsample(local_channels[1], local_channels[2])

        local_backbone3 = []
        for _ in range(local_blocks[2]):
            local_backbone3.append(InvertedResidual(dim=local_channels[2]//2, mlp_ratio=4, drop_path=dpr[dpr_idx]))
            dpr_idx += 1 
        local_backbone3.append(BN_Conv2d_act(local_channels[2]//2, local_channels[2]//2, 1, 1,0))
        self.add_module('local_backbone3', nn.Sequential(*local_backbone3))

        self.transition3 = BN_Conv2d_act(local_channels[2], local_channels[2], 1, 1,0)
        self.down3 = Downsample(local_channels[2], global_channels[0])

        backbone = []
        for j in range(global_blocks[0]):
            # Grapher 与 FFN 各用一个 drop rate，末尾各 +1（原实现两者共用同一个）
            backbone += [nn.Sequential(
                                Grapher(global_channels[0]//2, drop_path=dpr[dpr_idx], K=K),
                                FFN(global_channels[0]//2, global_channels[0]//2* 4, drop_path=dpr[dpr_idx + 1]))
                                ]
            dpr_idx += 2
        backbone.append(BN_Conv2d_act(global_channels[0]//2, global_channels[0]//2, 1, 1,0))
        self.add_module('backbone', nn.Sequential(*backbone))

        self.transition4 = BN_Conv2d_act(global_channels[0], global_channels[0], 1, 1,0)

        self._initialize_weights()


    def forward(self, inputs):
        x = self.stem(inputs)
        outs = []
        B, C, H, W = x.shape
        xs, xb = x.chunk(2, dim=1)
        xb = self.local_backbone1(xb)
        x = self.transition1(torch.cat([xs, xb], dim=1))
        outs.append(x)

        x = self.down1(x)
        xs, xb = x.chunk(2, dim=1)
        xb = self.local_backbone2(xb)
        x = self.transition2(torch.cat([xs, xb], dim=1))
        outs.append(x)

        x = self.down2(x)
        xs, xb = x.chunk(2, dim=1)
        xb = self.local_backbone3(xb)
        x = self.transition3(torch.cat([xs, xb], dim=1))
        outs.append(x)

        x = self.down3(x)
        xs, xb = x.chunk(2, dim=1)
        xb = self.backbone(xb)
        x = self.transition4(torch.cat([xs, xb], dim=1))
        outs.append(x)
        #return outs
        return {'res{}'.format(i + 2): r for i, r in enumerate(outs)}

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** 0.5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    # @torch.no_grad()
    # def train(self, mode=True):
    #     super().train(mode) 
    #     for m in self.modules():
    #         if isinstance(m, nn.BatchNorm2d):
    #             m.eval()

    # def _freeze_backbone(self, freeze_at):
    #         for layer_index in range(freeze_at):
    #             for p in self.features[layer_index].parameters():
    #                 p.requires_grad = False

@BACKBONE_REGISTRY.register()
def build_cspvigm_backbone(cfg, input_shape):
    model = MobileViG(local_blocks=[3, 3, 9],
                    local_channels=[42, 84, 224],
                    global_blocks=[3],
                    global_channels=[400],
                    dropout=0.,
                    drop_path=0.1,
                    emb_dims=768,
                    K=2,
                    distillation=True,
                    num_classes=1000,
                    out_indices=[2, 6, 16, 20])

    out_features = cfg.MODEL.RESNETS.OUT_FEATURES
    out_feature_channels = {"res2": 42, "res3": 84,
                            "res4": 224, "res5": 400}
    out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
#    model.default_cfg = default_cfgs['mobilevig']

    model._out_features = out_features
    model._out_feature_channels = out_feature_channels
    model._out_feature_strides = out_feature_strides
    return model

@BACKBONE_REGISTRY.register()
def build_cspvigm_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    """
    bottom_up = build_cspvigm_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        # top_block=LastLevelP6P7_P5(out_channels, out_channels),
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
 
@BACKBONE_REGISTRY.register()
def build_cspvigb_backbone(cfg, input_shape): 
    model = MobileViG(local_blocks=[5, 5, 15],
                    local_channels=[42, 84, 240],
                    global_blocks=[5],
                    global_channels=[464],
                    dropout=0.,
                    drop_path=0.1,
                    emb_dims=768,
                    K=2,
                    distillation=True,
                    num_classes=1000,
                    out_indices=[4, 10, 26, 32])
 #   model.default_cfg = default_cfgs['mobilevig']
    return model