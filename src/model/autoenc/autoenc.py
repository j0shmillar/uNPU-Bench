import os
import sys
import torch.nn as nn

train_path = os.environ.get("AI8X_TRAIN_PATH")
if not train_path:
    raise EnvironmentError("AI8X_TRAIN_PATH is not set.")

sys.path.append(train_path)

import ai8x

class CNN_BASE(nn.Module):
    """
    Auto Encoder Network
    """
    def __init__(self,
                 num_channels=3,  # pylint: disable=unused-argument
                 bias=True,  # pylint: disable=unused-argument
                 weight_init="kaiming",  # pylint: disable=unused-argument
                 num_classes=0,  # pylint: disable=unused-argument
                 **kwargs):  # pylint: disable=unused-argument
        super().__init__()

    def initWeights(self, weight_init="kaiming"):
        weight_init = weight_init.lower()
        assert weight_init in ('kaiming', 'xavier', 'glorot')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if weight_init == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                elif weight_init in ('glorot', 'xavier'):
                    nn.init.xavier_uniform_(m.weight)

            elif isinstance(m, nn.ConvTranspose2d):
                if weight_init == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                elif weight_init in ('glorot', 'xavier'):
                    nn.init.xavier_uniform_(m.weight)

            elif isinstance(m, nn.Linear):
                if weight_init == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                elif weight_init in ('glorot', 'xavier'):
                    nn.init.xavier_uniform_(m.weight)


class AI85AutoEncoder(CNN_BASE):

    def __init__(self,
                 num_channels=256,
                 dimensions=None,  # pylint: disable=unused-argument
                 num_classes=1,  # pylint: disable=unused-argument
                 n_axes=3,
                 bias=True,
                 weight_init="kaiming",
                 batchNorm=False,
                 bottleNeckDim=4,
                 **kwargs):

        super().__init__()

        print("Batchnorm setting in model = ", batchNorm)

        weight_init = weight_init.lower()
        assert weight_init in ('kaiming', 'xavier', 'glorot')

        self.num_channels = num_channels
        self.n_axes = n_axes

        S = 1
        P = 0

        n_in = num_channels
        n_out = 128
        if batchNorm:
            self.en_conv1 = ai8x.FusedConv1dBNReLU(n_in, n_out, 1, stride=S, padding=P, dilation=1,
                                                   bias=bias, batchnorm='Affine', **kwargs)
        else:
            self.en_conv1 = ai8x.FusedConv1dReLU(n_in, n_out, 1, stride=S, padding=P, dilation=1,
                                                 bias=bias, **kwargs)
        self.layer1_n_in = n_in
        self.layer1_n_out = n_out

        n_in = n_out
        n_out = 64
        if batchNorm:
            self.en_conv2 = ai8x.FusedConv1dBNReLU(n_in, n_out, 3, stride=S, padding=P, dilation=1,
                                                   bias=bias, batchnorm='Affine', **kwargs)
        else:
            self.en_conv2 = ai8x.FusedConv1dReLU(n_in, n_out, 3, stride=S, padding=P, dilation=1,
                                                 bias=bias, **kwargs)
        self.layer2_n_in = n_in
        self.layer2_n_out = n_out

        n_in = n_out
        n_out = 32
        self.en_lin1 = ai8x.FusedLinearReLU(n_in, n_out, bias=bias, **kwargs)

        n_in = n_out
        self.bottleNeckDim = bottleNeckDim
        n_out = self.bottleNeckDim
        self.en_lin2 = ai8x.Linear(n_in, n_out, bias=0, **kwargs)

        n_in = n_out
        n_out = 32
        self.de_lin1 = ai8x.FusedLinearReLU(n_in, n_out, bias=bias, **kwargs)

        n_in = n_out
        n_out = 96
        self.de_lin2 = ai8x.FusedLinearReLU(n_in, n_out, bias=bias, **kwargs)

        n_in = n_out
        n_out = num_channels*n_axes
        self.out_lin = ai8x.Linear(n_in, n_out, bias=0, **kwargs)

        self.initWeights(weight_init)

    def forward(self, x):
        x = self.en_conv1(x)
        x = self.en_conv2(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.en_lin1(x)
        x = self.en_lin2(x)

        x = self.de_lin1(x)
        x = self.de_lin2(x)
        x = self.out_lin(x)
        x = x.view(x.shape[0], self.num_channels, self.n_axes)

        return x


def ai85autoencoder(pretrained=False, **kwargs):
    assert not pretrained
    return AI85AutoEncoder(**kwargs)
