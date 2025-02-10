"""
CIFAR model arch found by NAS.
"""
import sys
import torch
import torch.nn as nn

sys.path.append("/Users/joshmillar/Desktop/phd/mcu-nn-eval/ai8x-training")  # TODO fix
import ai8x

class AI85NAS(nn.Module):
    """
    NAS-based model for CIFAR
    """
    def __init__(self, num_classes=10, num_channels=3, dimensions=(32, 32), bias=True, **kwargs):
        super().__init__()

        self.conv1_1 = ai8x.FusedConv2dBNReLU(num_channels, 64, 3, stride=1, padding=1,
                                              bias=bias, batchnorm='NoAffine', **kwargs)
        self.conv1_2 = ai8x.FusedConv2dBNReLU(64, 32, 1, stride=1, padding=0,
                                              bias=bias, batchnorm='NoAffine', **kwargs)
        self.conv1_3 = ai8x.FusedConv2dBNReLU(32, 64, 3, stride=1, padding=1,
                                              bias=bias, batchnorm='NoAffine', **kwargs)
        self.conv2_1 = ai8x.FusedMaxPoolConv2dBNReLU(64, 32, 3, stride=1, padding=1,
                                                     bias=bias, batchnorm='NoAffine', **kwargs)
        self.conv2_2 = ai8x.FusedConv2dBNReLU(32, 64, 1, stride=1, padding=0,
                                              bias=bias, batchnorm='NoAffine', **kwargs)
        self.conv3_1 = ai8x.FusedMaxPoolConv2dBNReLU(64, 128, 3, stride=1, padding=1,
                                                     bias=bias, batchnorm='NoAffine', **kwargs)
        self.conv3_2 = ai8x.FusedConv2dBNReLU(128, 128, 1, stride=1, padding=0,
                                              bias=bias, batchnorm='NoAffine', **kwargs)
        self.conv4_1 = ai8x.FusedMaxPoolConv2dBNReLU(128, 64, 3, stride=1, padding=1,
                                                     bias=bias, batchnorm='NoAffine', **kwargs)
        self.conv4_2 = ai8x.FusedConv2dBNReLU(64, 128, 3, stride=1, padding=1,
                                              bias=bias, batchnorm='NoAffine', **kwargs)
        
        self.conv5_1 = ai8x.FusedMaxPoolConv2dBNReLU(128, 512, 1, stride=1, padding=0,
                                                     bias=bias, batchnorm='NoAffine', **kwargs)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = ai8x.Linear(512, num_classes, bias=bias, wide=True, **kwargs)

    def forward(self, x):
        """Forward pass without unsupported transpose"""
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

        x = self.global_avg_pool(x) 
        x = torch.flatten(x, start_dim=1)

        x = self.fc(x)
        return x


def ai85nas(pretrained=False, **kwargs):
    """
    Constructs a NAS v1 model.
    """
    assert not pretrained
    return AI85NAS()
