import torch
from torch import nn
 
 
class CA_Block(nn.Module):
    """Coordinate Attention (https://arxiv.org/abs/2103.02907).

    原实现要求构造时传入固定的 h / w，而检测/分割中特征图尺寸随输入分辨率变化，
    导致该模块无法在不固定输入尺寸的网络中使用。这里改为运行时从输入推断 h / w，
    配合 AdaptiveAvgPool2d 的 (None, 1) / (1, None) 输出规格实现尺寸无关。
    """

    def __init__(self, channel, h=None, w=None, reduction=16):
        super(CA_Block, self).__init__()
        if channel // reduction == 0:
            raise ValueError(
                f"CA_Block: channel({channel}) 需 >= reduction({reduction})"
            )

        # None 表示该维保持输入尺寸，从而在 forward 中适配任意分辨率
        self.avg_pool_x = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, None))

        self.conv_1x1 = nn.Conv2d(
            in_channels=channel, out_channels=channel // reduction,
            kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(
            in_channels=channel // reduction, out_channels=channel,
            kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(
            in_channels=channel // reduction, out_channels=channel,
            kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)   # (b, c, w, 1) -> (b, c, 1, w)
        x_w = self.avg_pool_y(x)                        # (b, c, 1, w)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        # 切回两条分支。注意顺序：cat 时先放的是 x_h（长度 h），后放 x_w（长度 w），
        # 因此这里必须按 [h, w] 切。h != w 时若写成 [w, h] 会在 expand_as 处报
        # "expanded size must match" 之类的 shape 错误。
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        return x * s_h.expand_as(x) * s_w.expand_as(x)
 
 
# if __name__ == '__main__':
#     x = torch.randn(1, 16, 128, 64)    # b, c, h, w
#     ca_model = CA_Block(channel=16, h=128, w=64)
#     y = ca_model(x)
#     print(y.shape)