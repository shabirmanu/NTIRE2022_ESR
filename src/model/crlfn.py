import torch
import torch.nn as nn
import torch.nn.functional as F
def make_model(args, parent=False):
    model = CRLFN()
    return model

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x_channel = self.channel_attention(x) * x
        x_spatial = self.spatial_attention(x_channel) * x_channel
        return x_spatial

class RLFB(nn.Module):
    def __init__(self, in_channels, mid_channel, out_channels=None, cbam_channels=16):
        super(RLFB, self).__init__()

        if mid_channel is None:
            mid_channel = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = nn.Conv2d(in_channels, mid_channel, kernel_size=3, padding=1)
        self.c2_r = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1)
        self.c3_r = nn.Conv2d(mid_channel, in_channels, kernel_size=3, padding=1)

        self.c5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.cbam = CBAM(out_channels, cbam_channels)

        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.act(self.c1_r(input))
        out = self.act(self.c2_r(out))
        out = self.act(self.c3_r(out))
        out = out + input
        out = self.cbam(self.c5(out))
        return out

class CRLFN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=46, mf=48, upscale=4, cbam_channels=16):
        super(CRLFN, self).__init__()

        self.fea_conv = nn.Conv2d(in_nc, nf, kernel_size=3, padding=1)

        self.B1 = RLFB(in_channels=nf, mid_channel=mf, cbam_channels=cbam_channels)
        self.B2 = RLFB(in_channels=nf, mid_channel=mf, cbam_channels=cbam_channels)
        self.B3 = RLFB(in_channels=nf, mid_channel=mf, cbam_channels=cbam_channels)
        self.B4 = RLFB(in_channels=nf, mid_channel=mf, cbam_channels=cbam_channels)

        self.LR_conv = nn.Conv2d(nf, nf, kernel_size=3, padding=1)

        self.upsampler = nn.Sequential(
            nn.Conv2d(nf, out_nc * upscale ** 2, kernel_size=3, padding=1),  # Adjust the number of output channels here
            nn.PixelShuffle(upscale)
        )

    def forward(self, input):
        out_fea = self.fea_conv(input)

        out_B = self.B1(out_fea)
        out_B = self.B2(out_B)
        out_B = self.B3(out_B)
        out_B = self.B4(out_B)

        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)

        return output
# # Usage example:
# model = CRLFN()
#
# output = model(torch.randn(1, 3, 64, 64))  # Example input shape: (batch_size, channels, height, width)
# # Pass the input through the model
#
#
# # Print the shape of the output tensor
# print("Output shape:", output.shape)