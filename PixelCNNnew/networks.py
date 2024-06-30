import torch.nn.functional as F
from torch import nn
import torch
import numpy as np


# for 3 channels
class PixelCNN(nn.Module):
    """
    PixelCNN Network
    """
    def __init__(self, num_kernels=64, num_residual=10):
        super(PixelCNN, self).__init__()

        self.num_residual = num_residual
        self.conv_layers = nn.ModuleList()
        
        self.conv_layers.append(MaskedConv2d(mask_type='A', kernel_size=7, in_channels=3 ,out_channels=2*num_kernels, padding=3))

        for _ in range(num_residual):
            self.conv_layers.append(ResidualBlock(2*num_kernels))

        self.conv_layers.append(MaskedConv2d(in_channels=2*num_kernels, out_channels=2*num_kernels, mask_type='B', kernel_size=1, padding=0))
        self.conv_layers.append(MaskedConv2d(in_channels=2*num_kernels, out_channels=2*num_kernels, mask_type='B', kernel_size=1, padding=0))

        self.conv_out = nn.Conv2d(in_channels=2*num_kernels, out_channels=256, kernel_size=1)


    def forward(self, x, channel):

        # forward pass for each channel individually to respect the dependecies
        assert channel in ['r', 'g', 'b'], "Invalid channel"
        r, g, b = x[:, 0, :, :].unsqueeze(1), x[:, 1, :, :].unsqueeze(1), x[:, 2, :, :].unsqueeze(1)

        if channel == 'r':
            r_zeros_one = torch.zeros(r.size(), device=r.device)
            r_zeros_two = torch.zeros(r.size(), device=r.device)
            # shape of r_out [bs, 3, 32, 32]
            r_out = torch.cat((r_zeros_one, r_zeros_two, r), 1)

            r_out = F.relu(self.conv_layers[0](r_out))
            for layer in range(self.num_residual):
                r_out = self.conv_layers[layer+1](r_out)
            
            r_out = F.relu(self.conv_layers[self.num_residual+1](r_out))
            r_out = F.relu(self.conv_layers[self.num_residual+2](r_out))

            return self.conv_out(r_out)

        elif channel == 'g':
            g_zeros_one = torch.zeros(g.size(), device=g.device)
            g_out = torch.cat((g_zeros_one, r, g), 1)

            g_out = F.relu(self.conv_layers[0](g_out))
            for layer in range(self.num_residual):
                g_out = self.conv_layers[layer+1](g_out)
            
            g_out = F.relu(self.conv_layers[self.num_residual+1](g_out))
            g_out = F.relu(self.conv_layers[self.num_residual+2](g_out))

            return self.conv_out(g_out)
        
        else:
            b_out = torch.cat((r, g, b), 1)

            b_out = F.relu(self.conv_layers[0](b_out), 1)
            for layer in range(self.num_residual):
                b_out = self.conv_layers[layer+1](b_out)
            
            b_out = F.relu(self.conv_layers[self.num_residual+1](b_out))
            b_out = F.relu(self.conv_layers[self.num_residual+2](b_out))

            return self.conv_out(b_out)

# For one channel
class PixelCNNv2(nn.Module):
    def __init__(self, num_kernels=64, num_residual=10):
        super(PixelCNNv2, self).__init__()
        self.layers = nn.ModuleList()

        self.num_residual = num_residual
        self.conv_layers = nn.ModuleList()
        
        self.conv_layers.append(MaskedConv2d(mask_type='A', kernel_size=7, in_channels=1 ,out_channels=2*num_kernels, padding=3))

        for _ in range(num_residual):
            self.conv_layers.append(ResidualBlock(2*num_kernels))

        self.conv_layers.append(MaskedConv2d(in_channels=2*num_kernels, out_channels=2*num_kernels, mask_type='B', kernel_size=1, padding=0))
        self.conv_layers.append(MaskedConv2d(in_channels=2*num_kernels, out_channels=2*num_kernels, mask_type='B', kernel_size=1, padding=0))

        self.conv_out = nn.Conv2d(in_channels=2*num_kernels, out_channels=256, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.conv_layers[0](x))
        for layer in range(self.num_residual):
            out = self.conv_layers[layer+1](out)
        
        out = F.relu(self.conv_layers[self.num_residual+1](out))
        out = F.relu(self.conv_layers[self.num_residual+2](out))

        return self.conv_out(out)

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

        mask[:, :, :kHeight//2, :] = 1
        mask[:, :, kHeight//2, :kWidth//2 + 1] = 1
        
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
    ReLU -> Conv1x1 -> ReLU -> ColorMaskedConv2d -> ReLU -> Conv1x1
    """
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1x1_1 = MaskedConv2d(mask_type='B', in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, padding=0, stride=1)
        self.conv3x3 = MaskedConv2d(mask_type='B', in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=3, padding=1, stride=1)
        self.conv1x1_2 = MaskedConv2d(mask_type='B', in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, padding=0, stride=1)


    def forward(self, input):
        x = input
        out = F.relu(input)
        out = F.relu(self.conv1x1_1(out))
        out = F.relu(self.conv3x3(out))
        out = self.conv1x1_2(out)
        return out + x
    