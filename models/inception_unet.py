import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        branch_ch = out_ch // 4
        self.branch1 = nn.Conv2d(in_ch, branch_ch, 1)
        self.branch3 = nn.Conv2d(in_ch, branch_ch, 3, padding=1)
        self.branch5 = nn.Conv2d(in_ch, branch_ch, 5, padding=2)
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, branch_ch, 1),
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = torch.cat([
            self.branch1(x),
            self.branch3(x),
            self.branch5(x),
            self.branch_pool(x),
        ], dim=1)
        return self.relu(self.bn(out))


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.incep = InceptionBlock(in_ch, out_ch)

    def forward(self, x):
        return self.incep(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.incep = InceptionBlock(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.incep(torch.cat([x2, x1], dim=1))


class InceptionUNetStandard(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        self.incep1 = InceptionBlock(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.up4 = Up(32, 16)
        self.outc = nn.Conv2d(16, n_classes, 1)

    def forward(self, x):
        x1 = self.incep1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return torch.sigmoid(self.outc(x))
