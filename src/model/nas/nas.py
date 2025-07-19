import os
import sys
import torch.nn as nn

train_path = os.environ.get("AI8X_TRAIN_PATH")
if not train_path:
    raise EnvironmentError("AI8X_TRAIN_PATH is not set.")

sys.path.append(train_path)

import ai8x

class AI85NASCifarNet(nn.Module):
    def __init__(self, num_classes=10, num_channels=3, dimensions=(32, 32), bias=True, **kwargs):
        super().__init__()

        self.conv1_1 = ai8x.FusedConv2dReLU(num_channels, 64, 3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        self.conv1_2 = ai8x.FusedConv2dReLU(64, 32, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.conv1_3 = ai8x.FusedConv2dReLU(32, 64, 3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        self.conv2_1 = ai8x.FusedMaxPoolConv2dReLU(64, 32, 3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        self.conv2_2 = ai8x.FusedConv2dReLU(32, 64, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.conv3_1 = ai8x.FusedMaxPoolConv2dReLU(64, 128, 3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        self.conv3_2 = ai8x.FusedConv2dReLU(128, 128, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.conv4_1 = ai8x.FusedMaxPoolConv2dReLU(128, 64, 3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        self.conv4_2 = ai8x.FusedConv2dReLU(64, 128, 3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        self.conv5_1 = ai8x.FusedMaxPoolConv2dReLU(128, 128, 1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.avgpool = ai8x.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = ai8x.Linear(128, num_classes, bias=bias, wide=True, **kwargs)

    def forward(self, x): 
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv5_1(x)
        x = self.avgpool(x)
        x = x.contiguous().reshape(-1, 128)
        x = self.fc(x)
        return x

def ai85nascifarnet(pretrained=False, **kwargs):
    assert not pretrained
    return AI85NASCifarNet(**kwargs)
