import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=(3, 3), stride=(1,), padding=0, acti=nn.LeakyReLU(),
                 groups=1):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(channel_in, channel_out, kernel_size, stride, padding, groups)
        self.norm = nn.BatchNorm2d(channel_out)
        self.acti = acti

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.acti(x)
        return x


class DoNothing(nn.Module):
    def forward(self, x):
        return x


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, acti=nn.LeakyReLU(), bn=True):
        super(DenseBlock, self).__init__()
        self.linear = nn.Linear(channel_in, channel_out)
        if bn:
            self.norm = nn.BatchNorm1d(channel_out)
        else:
            self.norm = DoNothing()
        if acti is not None:
            self.acti = acti
        else:
            self.acti = DoNothing()

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.acti(x)
        return x


