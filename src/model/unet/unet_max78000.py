"""
UNet network for MAX7800X
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("/Users/joshmillar/Desktop/phd/mcu-nn-eval/ai8x-training") # TODO fix
import ai8x

class AI85UNet(nn.Module):
    """
    Small UNet_v3 model with ConvTranspose2d replaced by Interpolate + Conv2d
    """
    def __init__(
            self,
            num_classes=4,
            num_channels=3,
            dimensions=(128, 128),
            bias=True,
            **kwargs
    ):
        super().__init__()

        self.enc1 = ai8x.FusedConv2dBNReLU(num_channels, 4, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)
        self.enc2 = ai8x.FusedMaxPoolConv2dBNReLU(4, 8, 3, stride=1, padding=1,
                                                  bias=bias, batchnorm='NoAffine', **kwargs)
        self.enc3 = ai8x.FusedMaxPoolConv2dBNReLU(8, 32, 3, stride=1, padding=1,
                                                  bias=bias, batchnorm='NoAffine', **kwargs)

        self.bneck = ai8x.FusedMaxPoolConv2dBNReLU(32, 64, 3, stride=1, padding=1,
                                                   bias=bias, batchnorm='NoAffine', **kwargs)
        
        self.upconv3 = ai8x.FusedConv2dBNReLU(64, 32, 3, stride=1, padding=1,
                                              bias=bias, batchnorm='NoAffine', **kwargs)
        self.dec3 = ai8x.FusedConv2dBNReLU(64, 32, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)

        self.upconv2 = ai8x.FusedConv2dBNReLU(32, 8, 3, stride=1, padding=1,
                                              bias=bias, batchnorm='NoAffine', **kwargs)
        self.dec2 = ai8x.FusedConv2dBNReLU(16, 8, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)

        self.upconv1 = ai8x.FusedConv2dBNReLU(8, 4, 3, stride=1, padding=1,
                                              bias=bias, batchnorm='NoAffine', **kwargs)
        self.dec1 = ai8x.FusedConv2dBNReLU(8, 16, 3, stride=1, padding=1,
                                           bias=bias, batchnorm='NoAffine', **kwargs)

        self.conv = ai8x.FusedConv2dBN(16, 1, 1, stride=1, padding=0,
                                       bias=bias, batchnorm='NoAffine', **kwargs)

    def forward(self, x):
        """Forward prop"""

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        bottleneck = self.bneck(enc3)

        # upsampling instead of tranpose conv
        dec3 = F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=False)
        dec3 = self.upconv3(dec3)  # learnable conv layer after upsampling
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=False)
        dec2 = self.upconv2(dec2)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=False)
        dec1 = self.upconv1(dec1)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        return self.conv(dec1)

def ai85unet(pretrained=False, **kwargs):
    """
    Constructs a small unet (unet_v3) model.
    """
    assert not pretrained
    return AI85UNet(num_classes=3, num_channels=3, dimensions=(80, 80))

