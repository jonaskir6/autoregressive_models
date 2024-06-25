import torch.nn.functional as F
from torch import nn
import torch
import numpy as np

# TODO: Residual Blocks: 2h out of context or the out_channels of Mask A 256?


class PixelCNN(nn.Module):
    """
    PixelCNN Network
    """
    def __init__(self, num_kernels=64, dataset='mnist'):
        super(PixelCNN, self).__init__()

        # if dataset == 'mnist':
        #     self.net = nn.Sequential(
        #         MaskedConv2d(mask_type='A', kernel_size=7, in_channels=1 ,out_channels=2*num_kernels, padding=3),
        #         *[ResidualBlock(2*num_kernels) for _ in range(num_residual)],
        #         nn.ReLU(),
        #         MaskedConv2d(in_channels=2*num_kernels, out_channels=2*num_kernels, mask_type='B', kernel_size=1, padding=0),
        #         nn.ReLU(),
        #         MaskedConv2d(in_channels=2*num_kernels, out_channels=2*num_kernels, mask_type='B', kernel_size=1, padding=0),
        #         nn.ReLU(),
        #         nn.Conv2d(in_channels=2*num_kernels, out_channels=256, kernel_size=1)
        #     )

        if dataset == 'mnist':
            self.a = MaskedConv2d(mask_type='A', kernel_size=7, in_channels=1 ,out_channels=2*num_kernels, padding=3)

            self.r1 = ResidualBlock(2*num_kernels)
            self.r2 = ResidualBlock(2*num_kernels)
            self.r3 = ResidualBlock(2*num_kernels)
            self.r4 = ResidualBlock(2*num_kernels)
            self.r5 = ResidualBlock(2*num_kernels)
            self.r6 = ResidualBlock(2*num_kernels)
            self.r7 = ResidualBlock(2*num_kernels)
            self.r8 = ResidualBlock(2*num_kernels)
            self.r9 = ResidualBlock(2*num_kernels)
            self.r10 = ResidualBlock(2*num_kernels)

            self.b1 = MaskedConv2d(in_channels=2*num_kernels, out_channels=2*num_kernels, mask_type='B', kernel_size=1, padding=0)
            self.b2 = MaskedConv2d(in_channels=2*num_kernels, out_channels=2*num_kernels, mask_type='B', kernel_size=1, padding=0)

            self.channels = nn.Conv2d(in_channels=2*num_kernels, out_channels=256, kernel_size=1, padding=0)
        
        # elif dataset == 'cifar10':
        #     self.net = nn.Sequential(
        #         ColorMaskedConv2d(mask_type='A', kernel_size=7, in_channels=3 ,out_channels=2*num_kernels, padding=3),
        #         *[ColorResidualBlock(2*num_kernels) for _ in range(num_residual)],
        #         nn.ReLU(),
        #         ColorMaskedConv2d(in_channels=2*num_kernels, out_channels=2*num_kernels, mask_type='B', kernel_size=1, padding=0),
        #         nn.ReLU(),
        #         ColorMaskedConv2d(in_channels=2*num_kernels, out_channels=2*num_kernels, mask_type='B', kernel_size=1, padding=0),
        #         nn.ReLU(),
        #         nn.Conv2d(in_channels=2*num_kernels, out_channels=3*256, kernel_size=1, stride=1, padding=0 )
        #     )

        else :
            raise NotImplementedError("Dataset not supported")

    def forward(self, x):
        out = F.relu(self.a(x))

        out = self.r1(out)
        out = self.r2(out)
        out = self.r3(out)
        out = self.r4(out)
        out = self.r5(out)
        out = self.r6(out)
        out = self.r7(out)
        out = self.r8(out)
        out = self.r9(out)
        out = self.r10(out)

        out = F.relu(self.b1(out))
        out = F.relu(self.b2(out))

        channels = F.relu(self.channels(out))

        return channels
    

class MaskedConv2d(nn.Conv2d):
    """
    Masked Convolutional Layer
    """
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, padding=0, data_channels=1, stride=1):
        self.mask_type = mask_type
        assert mask_type in {'A', 'B'}, "Invalid mask type"
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.register_buffer('mask', self.weight.data.clone())

        _, depth, height, width = self.weight.size()
        self.mask.fill_(1)
        if mask_type =='A':
            self.mask[:,:,height//2,width//2:] = 0
            self.mask[:,:,height//2+1:,:] = 0
        else:
            self.mask[:,:,height//2,width//2+1:] = 0
            self.mask[:,:,height//2+1:,:] = 0

    def forward(self, x):
        # forward pass with all values that are not masked out
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class ColorMaskedConv2d(nn.Conv2d):
    """
    Masked Convolutional Layer
    """
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, padding=0, data_channels=3, stride=1):
        self.mask_type = mask_type
        assert mask_type in {'A', 'B'}, "Invalid mask type"
        super(ColorMaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, padding=padding, stride=stride)

        # Initialize mask with 1s, don't need in/out channels here
        mask = np.zeros(self.weight.size(), dtype=np.float32)
        kHeight, kWidth = kernel_size, kernel_size

        # Only mask out future pixels
        mask[:, :, :kHeight//2, :] = 1
        mask[:, :, kHeight//2, :kWidth//2 + 1] = 1

        # # Create Boolean mask for each kernel to determine wether to mask out the color channel or not
        # def color_mask(color_out, color_in):
        #     a = (np.arange(out_channels) % data_channels == color_out)[:, None]
        #     b = (np.arange(in_channels) % data_channels == color_in)[None, :]
        #     return a * b

        # # Mask out color channels according to the Paper: R -> R, G -> RG, B -> RGB
        # for output_channel in range(data_channels):
        #     for input_channel in range(output_channel + 1, data_channels):
        #         mask[color_mask(output_channel, input_channel), kHeight//2, kWidth//2] = 0
        
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
    ReLU -> Conv1x1 -> ReLU -> Conv3x3 -> ReLU -> Conv1x1
    """
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        
        # self.net = nn.Sequential(
        #     nn.ReLU(),
        #     MaskedConv2d(mask_type='B', in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, padding=0),
        #     nn.ReLU(),
        #     MaskedConv2d(mask_type='B', in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     MaskedConv2d(mask_type='B', in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, padding=0)
        # )

        self.conv1x1_1 = MaskedConv2d(mask_type='B', in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, padding=0, stride=1)
        self.conv3x3 = MaskedConv2d(mask_type='B', in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=3, padding=1, stride=1)
        self.conv1x1_2 = MaskedConv2d(mask_type='B', in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, padding=0, stride=1)

    # x being the layers in the residual block
    def forward(self, input):
        x = input
        out = F.relu(input)
        out = F.relu(self.conv1x1_1(out))
        out = F.relu(self.conv3x3(out))
        out = F.relu(self.conv1x1_2(out))
        return out + x
    

class ColorResidualBlock(nn.Module):
    """
    Residual Blocks:
    ReLU -> Conv1x1 -> ReLU -> ColorMaskedConv2d -> ReLU -> Conv1x1
    """
    def __init__(self, in_channels):
        super(ColorResidualBlock, self).__init__()
        
        self.net = nn.Sequential(
            nn.ReLU(),
            ColorMaskedConv2d(mask_type='B', in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, padding=0),
            nn.ReLU(),
            ColorMaskedConv2d(mask_type='B', in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=3, padding=1),
            nn.ReLU(),
            ColorMaskedConv2d(mask_type='B', in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, padding=0)
        )


    # x being the layers in the residual block
    def forward(self, input):
        return self.net(input) + input