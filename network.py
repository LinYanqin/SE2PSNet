import torch
import torch.nn as nn

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        queries = self.query_conv(x).view(batch_size, -1, height * width)
        keys = self.key_conv(x).view(batch_size, -1, height * width)
        values = self.value_conv(x).view(batch_size, -1, height * width)

        attention_scores = torch.bmm(queries.permute(0, 2, 1), keys)  # (B, H*W, H*W)
        attention_scores = torch.softmax(attention_scores, dim=-1)

        out = torch.bmm(values, attention_scores).view(batch_size, -1, height, width)

        return out + x

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=0, bias=False):
        super(ConvolutionalBlock, self).__init__()
        self.sub_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.sub_module(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            ConvolutionalBlock(out_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.sub_module(x) + x
        return x

class DownSamplingBlock1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSamplingBlock1, self).__init__()
        self.sub_module = nn.Sequential(
            ConvolutionalBlock(in_channels, out_channels, kernel_size=3, stride=(1, 2), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.sub_module(x)
        return x

class SpinEchoNet(nn.Module):
    def __init__(self):
        super(SpinEchoNet, self).__init__()

        self.input = nn.Sequential(
            ConvolutionalBlock(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            DownSamplingBlock1(in_channels=16, out_channels=32),

            ResidualBlock(in_channels=32, out_channels=32),
            ResidualBlock(in_channels=32, out_channels=32),

            DownSamplingBlock1(in_channels=32, out_channels=64),

            ResidualBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64),

            DownSamplingBlock1(in_channels=64, out_channels=128),

            ResidualBlock(in_channels=128, out_channels=128),
            ResidualBlock(in_channels=128, out_channels=128),

            SpatialAttention(128, 128)

        )

        self.output = nn.Sequential(
            ResidualBlock(in_channels=128, out_channels=128),
            ResidualBlock(in_channels=128, out_channels=128),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            ResidualBlock(in_channels=64, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
            ResidualBlock(in_channels=32, out_channels=32),
            ResidualBlock(in_channels=32, out_channels=32),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0),
            ResidualBlock(in_channels=16, out_channels=16),
            ResidualBlock(in_channels=16, out_channels=16),

            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, stride=1, padding=0),
            ResidualBlock(in_channels=8, out_channels=8),
            ResidualBlock(in_channels=8, out_channels=8),

            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=1, stride=1, padding=0),
            ResidualBlock(in_channels=4, out_channels=4),
            ResidualBlock(in_channels=4, out_channels=4),

            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(2),
        )

    def forward(self, x, y):
        y1 = y[:, 0:1, :, :].permute(0, 3, 2, 1) ## one chemical shift spectrum
        x = torch.cat((x[:, :, :, 0:7], y1[:, :, :, 0:1]), dim=3) ## seven spin echo spectra
        x1 = self.input(x)
        x2 = self.output(x1)
        out = x2
        return out, x1, y1