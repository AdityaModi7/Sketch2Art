import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def get_resnet_structure(n_layers=50):
    n_layer_structs = {
        18:  [2, 2, 2, 2],
        34:  [3, 4, 6, 3],
        50:  [3, 4, 14, 3],
        101: [3, 13, 30, 3],
        152: [3, 8, 36, 3],
    }

    struct = n_layer_structs[n_layers]

    return [
        {
            "in_channels": 64,
            "out_channels": 64,
            "num_units": struct[0]
        },
        {
            "in_channels": 64,
            "out_channels": 128,
            "num_units": struct[1]
        },
        {
            "in_channels": 128,
            "out_channels": 256,
            "num_units": struct[2]
        },
        {
            "in_channels": 256,
            "out_channels": 512,
            "num_units": struct[3]
        },
    ]


def create_resnet_block(in_channels, out_channels, num_units, stride=2):
    modules = [Bottleneck(in_channels, out_channels, stride)]
    for i in range(num_units - 1):
        modules.append(Bottleneck(out_channels, out_channels, 1))
    return nn.Sequential(*modules)


class Bottleneck(nn.Module):
    """
    Essentially a ResNet block
    """

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        if in_channels == out_channels:
            self.shortcut = nn.MaxPool2d(1, stride)
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            SqueezeExcitation(out_channels, 16)
        )

    def forward(self, x):
        return self.shortcut(x) + self.res_layer(x)


class SqueezeExcitation(nn.Module):
    """
    Calculates a value for each channel in the input between [0, 1]
    and multiplies every value in the channel by the amount
    """

    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels //
                             reduction, 1, padding=0, bias=False)
        self.fc2 = nn.Conv2d(channels // reduction,
                             channels, 1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = self.relu(self.fc1(self.avg_pool(x)))
        x = self.sigmoid(self.fc2(x))
        return identity * x


class Map2Style(nn.Module):

    def __init__(self, in_channels, out_channels, spatial):
        super().__init__()
        self.out_c = out_channels
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class ResnetEncoder(nn.Module):

    def __init__(self, n_layers=50, input_nc=6, n_styles=16):
        super().__init__()

        self.input = nn.Sequential(
            nn.Conv2d(input_nc, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )

        self.body = nn.Sequential(*[create_resnet_block(**block)
                                  for block in get_resnet_structure(n_layers=n_layers)])

        self.styles = nn.ModuleList()
        for i in range(n_styles):
            self.styles.append(Map2Style(512, 512, 16))

    def forward(self, x):
        x = self.input(x)
        x = self.body(x)
        latents = []
        for i in range(len(self.styles)):
            latents.append(self.styles[i](x))
        return torch.stack(latents, dim=1)


class ResnetModel(nn.Module):

    def __init__(self, input_nc, output_nc, activation=None, n_layers=50, res=(256, 256)):
        super().__init__()

        self.activation = activation

        self.input = nn.Sequential(
            nn.Conv2d(input_nc, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )

        self.body = nn.Sequential(*[create_resnet_block(**block)
                                  for block in get_resnet_structure(n_layers=n_layers)])
        self.fc1 = nn.Linear(self.get_size(input_nc, res), output_nc)

    def get_size(self, input_nc, resolution):
        x = torch.zeros(1, input_nc, resolution[0], resolution[1])
        x = self.input(x)
        x = self.body(x)
        x = x.view(x.size(0), -1)
        return x.size(1)

    def forward(self, x):
        x = self.input(x)
        x = self.body(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.activation:
            x = self.activation(x)
        return x
