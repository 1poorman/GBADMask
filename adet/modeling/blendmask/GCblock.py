'''
Descripttion: 
Result: 
Author: Philo
Date: 2023-03-11 17:55:19
LastEditors: Philo
LastEditTime: 2023-03-13 20:25:19
'''
import torch
import torch.nn as nn

class GlobalContextBlock(nn.Module):
    def __init__(self, inplanes, ratio, pooling_type="att", fusion_types=('channel_mul')) -> None:
        super().__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']

        assert pooling_type in ['avg', 'att']
        # assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes*ratio)
        self.pooling_type = pooling_type
        self.fusion_type = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None

        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
    
    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':  # 这里其实就是空间注意力 最后得到一个b c 1 1的权重
            input_x = x
            input_x = input_x.view(batch, channel, height*width) # -> b c h*w
            input_x = input_x.unsqueeze(1) # -> b 1 c hw
            context_mask = self.conv_mask(x)  # b 1 h w
            context_mask = context_mask.view(batch, 1, height*width)  # b 1 hw
            context_mask = self.softmax(context_mask) # b 1 hw
            context_mask = context_mask.unsqueeze(-1) # b 1 hw 1
            context = torch.matmul(input_x, context_mask) # b(1 c hw  *  1 hw 1) -> b 1 c 1
            context = context.view(batch, channel, 1, 1)  # b c 1 1
        else:
            context = self.avg_pool(x)   # b c 1 1
        return context
    
    def forward(self, x):
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context)) # 将权重进行放大缩小
            out = out * channel_mul_term  # 与x进行相乘
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out

# if __name__ == "__main__":
#     input = torch.randn(16, 64, 32, 32)
#     net = GlobalContextBlock(64, ratio=1/16)
#     out = net(input)
#     print(out.shape)
