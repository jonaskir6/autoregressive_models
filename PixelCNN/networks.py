import torch.nn.functional as F
from torch import nn


class PixelCNN(nn.Module):
    """
    PixelCNN Network
    """
    def __init__(self, num_layers, kernel=7, channels=128, res_num=5, device=None):
        super(PixelCNN, self).__init__()
        self.num_layers = num_layers
        self.kernel = kernel
        self.channels = channels
        self.res_num = res_num
        self.device = device

        net = self.MaskedConv2d('A', 1, kernel, 1, padding=kernel // 2)
        
        for _ in range(res_num):
            net.add_module('ResidualBlock', self.ResidualBlock(channels))
        
        net.add_module('Conv2d', nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=1))


class MaskedConv2d(nn.Conv2d):
    """
    Masked Convolutional Layer
    """
    def __init__(self, mask_type, *args, **kwargs):
        self.mask_type = mask_type
        assert mask_type in {'A', 'B'}, "Invalid mask type"
        super(MaskedConv2d, self).__init__(*args, **kwargs)

        # Initialize mask with 1s, don't need in/out channels here
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kHeight, kWidth = self.weight.size()
        self.mask.fill_(1)

        # Mask out future pixels: if mask type is B, the mask will be shifted one pixel to the right
        self.mask[:, :, kHeight // 2, kWidth // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kHeight // 2 + 1:] = 0

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
        self.conv1 = MaskedConv2d(mask_type='B', filters=filters, kernel_size=1, padding=1)
        self.maskedconv = MaskedConv2d(mask_type='B', filters=filters // 2, kernel_size=3, padding=1)
        self.conv2 = MaskedConv2d(mask_type='B', filters=filters, kernel_size=1, padding=1)
        self.relu = nn.ReLU()

    # x being the layers in the residual block
    def forward(self, input):
        x = self.relu(input)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maskedconv(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + input