import torch.nn.functional as F
from torch import nn
import torch
import numpy as np


class PixelCNN(nn.Module):
    """
    PixelCNN Network
    """
    def __init__(self, num_layers, channels=128, res_num=12):
        super(PixelCNN, self).__init__()
        self.num_layers = num_layers
        self.channels = channels
        self.res_num = res_num
        
        #TODO - Args

        net = MaskedConv2d('A', kernel_size=7, in_c= , 1, padding=kernel // 2)
        
        for _ in range(res_num):
            net.add_module('ResidualBlock', self.ResidualBlock(channels))

        net.add_module('Mask B', MaskedConv2d(mask_type='B', kernel_size=1, padding=1))
        net.add_module('Mask B', MaskedConv2d(mask_type='B', kernel_size=1, padding=1))
        
        net.add_module('ReLU', nn.ReLU())
        net.add_module('Conv2d', nn.functional.softmax(dim=4))


class MaskedConv2d(nn.Conv2d):
    """
    Masked Convolutional Layer
    """
    def __init__(self, mask_type, kernel_size, in_c, out_c, data_channels=3, *args, **kwargs):
        self.mask_type = mask_type
        assert mask_type in {'A', 'B'}, "Invalid mask type"
        super(MaskedConv2d, self).__init__(in_c, out_c, kernel_size, *args, **kwargs)

        # Initialize mask with 1s, don't need in/out channels here
        self.register_buffer('mask', self.weight.data.clone())
        out_channels, in_channels, kHeight, kWidth = self.weight.size()
        self.mask.fill_(1)

        # Mask out future pixels: if mask type is B, the pixel at the center of the kernel is also not masked out
        # ??
        self.mask[:, :, kHeight // 2, kWidth // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kHeight // 2 + 1:] = 0

        def color_mask(color_out, color_in):
            a = (np.arange(out_channels) % data_channels == color_out)[:, None]
            b = (np.arange(in_channels) % data_channels == color_in)[None, :]
            return a * b
        
        for i in range(data_channels):
            for j in range(i + 1, data_channels):
                self.mask[color_mask(i, j), kHeight // 2, kWidth // 2] = 0
        
        # ??
        # if mask_type == 'A':
        #     for k in range(data_channels):
        #         self.mask[color_mask(k, k), kHeight // 2, kWidth // 2] = 1  

        self.mask = torch.from_numpy(self.mask).float()

    def forward(self, x):
        # forward pass with all values that are not masked out
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)
    

class ResidualBlock(nn.Module):
    """
    Residual Blocks:
    ReLU -> Conv1x1 -> ReLU -> MaskedConv2d -> ReLU -> Conv1x1
    """
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        
        self.net = nn.Sequential(
            nn.ReLU(),
            MaskedConv2d(mask_type='B', filters=filters, kernel_size=1, padding=1),
            nn.ReLU(),
            MaskedConv2d(mask_type='B', filters=filters // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            MaskedConv2d(mask_type='B', filters=filters, kernel_size=1, padding=1)
        )


    # x being the layers in the residual block
    def forward(self, input):
        return self.net(input) + input