import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_activation(name="relu", negative_slope=0.01):
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leakyrelu":
        return nn.LeakyReLU(negative_slope, inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, activation="relu", negative_slope=0.01):
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=dilation, dilation=dilation, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            _make_activation(activation, negative_slope),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, activation="relu", negative_slope=0.01):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            _make_activation(activation, negative_slope),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        atrous_rates=(3, 6, 12, 18),
        dropout=0.5,
        activation="relu",
        negative_slope=0.01,
    ):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                _make_activation(activation, negative_slope),
            )
        ]
        for rate in atrous_rates:
            modules.append(
                ASPPConv(
                    in_channels, out_channels, rate,
                    activation=activation, negative_slope=negative_slope
                )
            )
        modules.append(
            ASPPPooling(
                in_channels, out_channels,
                activation=activation, negative_slope=negative_slope
            )
        )

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            _make_activation(activation, negative_slope),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.project(res)


class CNN_ASPP(nn.Module):
    def __init__(
        self,
        in_channels=1,
        dropout_rate=0.5,
        atrous_rates=(3, 6, 12, 18),
        activation="relu",
        negative_slope=0.01,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.aspp = ASPP(
            256, 128,
            atrous_rates=tuple(atrous_rates),
            dropout=dropout_rate,
            activation=activation,
            negative_slope=negative_slope,
        )

        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.bn = nn.BatchNorm2d(32)
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.aspp(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        x = self.bn(x)
        return torch.sigmoid(self.final(x))