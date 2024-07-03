import torch.nn.functional as F
from torch import nn
import torch
import numpy as np



class PixelCNN(nn.Module):
    """
    PixelCNN Network
    """
    def __init__(self, num_kernels=128,  num_residual=7, in_channels=3):
        super(PixelCNN, self).__init__()

        self.net = nn.Sequential(
            MaskedConv2d(mask_type='A', kernel_size=7, in_channels=in_channels ,out_channels=2*num_kernels, padding=3),
            *[ResidualBlock(2*num_kernels) for _ in range(num_residual)],
            nn.BatchNorm2d(2*num_kernels),
            nn.ReLU(),
            MaskedConv2d(in_channels=2*num_kernels, out_channels=2*num_kernels, mask_type='B', kernel_size=1, padding=0),
            nn.BatchNorm2d(2*num_kernels),
            nn.ReLU(),
            MaskedConv2d(in_channels=2*num_kernels, out_channels=2*num_kernels, mask_type='B', kernel_size=1, padding=0),
            nn.BatchNorm2d(2*num_kernels),
            nn.ReLU(),
            nn.Conv2d(in_channels=2*num_kernels, out_channels=in_channels*256, kernel_size=1, stride=1, padding=0 )
        )

    def forward(self, x):
        return self.net(x)

class MaskedConv2d(nn.Conv2d):
    """
    Masked Convolutional Layer
    """
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, padding=0, data_channels=3, stride=1):
        self.mask_type = mask_type
        assert mask_type in {'A', 'B'}, "Invalid mask type"
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, padding=padding, stride=stride)

        # Initialize mask with 1s, don't need in/out channels here
        mask = np.zeros(self.weight.size(), dtype=np.float32)
        kHeight, kWidth = kernel_size, kernel_size

        # Only mask out future pixels
        mask[:, :, :kHeight//2, :] = 1
        mask[:, :, kHeight//2, :kWidth//2 + 1] = 1

         # Create Boolean mask for each kernel to determine wether to mask out the color channel or not
        def color_mask(color_out, color_in):
            a = (np.arange(out_channels) % data_channels == color_out)[:, None]
            b = (np.arange(in_channels) % data_channels == color_in)[None, :]
            return a * b

         # Mask out color channels according to the Paper: R -> R, G -> RG, B -> RGB
        for output_channel in range(data_channels):
            for input_channel in range(output_channel + 1, data_channels):
                mask[color_mask(output_channel, input_channel), kHeight//2, kWidth//2] = 0
        
        if mask_type == 'A':
            mask[:, :, kHeight//2, kWidth//2] = 0
        
        self.register_buffer('mask', torch.from_numpy(mask))

    def forward(self, x):
        # forward pass with all values that are not masked out
        self.weight.data *= self.mask
        return super().forward(x)
    

class ResidualBlock(nn.Module):
    """
    Residual Blocks:
    ReLU -> Conv1x1 -> ReLU -> MaskedConv2d -> ReLU -> Conv1x1
    """
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            MaskedConv2d(mask_type='B', in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            MaskedConv2d(mask_type='B', in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(),
            MaskedConv2d(mask_type='B', in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, padding=0)
        )


    # x being the layers in the residual block
    def forward(self, input):
        return self.net(input) + input