import os 
import sys

train_path = os.environ.get("AI8X_TRAIN_PATH")
if not train_path:
    raise EnvironmentError("AI8X_TRAIN_PATH is not set.")

sys.path.append(train_path)

import torch
import torch.nn as nn

import ai8x
from ai8x import Conv2d
from ai8x import FusedConv2dReLU, FusedMaxPoolConv2dReLU, FusedConv2dReLU, FusedMaxPoolConv2dBNReLU

ai8x.set_device(85, 0, True)

class Yolov1_net(nn.Module):

    def __init__(self, B=2, num_classes=2, bias=True, **kwargs):
        super().__init__()
        print("YOLO v1 Model_Z {} class (96 input), {} bounding boxes.".format(num_classes, B))
        self.B = B
        self.Classes_Num = num_classes

        self.Conv_96 = FusedConv2dReLU(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.Conv_48 = FusedMaxPoolConv2dReLU(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
       
        self.Conv_24_1 = FusedMaxPoolConv2dReLU(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
        self.Conv_24_2 = FusedConv2dReLU(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_24_3 = FusedConv2dReLU(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_24_4 = FusedConv2dReLU(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_24_5 = FusedConv2dReLU(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_24_6 = FusedConv2dReLU(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        
        self.Conv_12_1 = FusedMaxPoolConv2dBNReLU(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_12_2 = FusedConv2dReLU(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_12_3 = FusedConv2dReLU(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_12_4 = FusedConv2dReLU(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)    
        self.Conv_12_5 = FusedConv2dReLU(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)  
        self.Conv_12_6 = FusedConv2dReLU(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        
        self.Conv_7_1 = FusedConv2dReLU(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_7_2 = FusedConv2dReLU(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        
        self.Conv_Res_1 = FusedConv2dReLU(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_Res_2 = FusedConv2dReLU(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_Res_3 = FusedConv2dReLU(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_Res_4 = Conv2d(in_channels=16, out_channels=self.B * 5 + self.Classes_Num, kernel_size=1, stride=1, padding=0, bias=True, wide=True, **kwargs)

        self.identity_1 = nn.Identity()
        self.identity_2 = nn.Identity()

    def forward(self, x):
        x = self.Conv_96(x)
        x = self.Conv_48(x)
        x = self.Conv_24_1(x)
        x = self.Conv_24_2(x)
        x = self.Conv_24_3(x)
        x = self.Conv_24_4(x)
        x = self.Conv_24_5(x)
        x = self.Conv_24_6(x)
        x = self.Conv_12_1(x)
        x = self.Conv_12_2(x)
        x = self.Conv_12_3(x)
        x = self.Conv_12_4(x)  
        x = self.Conv_12_5(x)
        x = self.Conv_12_6(x)
        x = self.Conv_7_1(x)
        x = self.Conv_7_2(x)
        x = self.Conv_Res_1(x)
        x = self.Conv_Res_2(x)
        x = self.Conv_Res_3(x)
        x_fl_output = self.Conv_Res_4(x)
        x = x_fl_output.permute(0, 2, 3, 1)
        
        bbox_preds = self.identity_1(x[:, :, :, :10])  
        class_preds = self.identity_2(x[:, :, :, 10:]) 
        
        return bbox_preds, class_preds, x_fl_output

def ai85yolo96(pretrained=False, **kwargs):
    assert not pretrained
    return Yolov1_net(**kwargs)

def ai85yolo96_save():
    model = Yolov1_net()
    torch.save(model.state_dict(), 'model/yolo/yolov1.pth')

if __name__ == "__main__":
    ai85yolo96_save()
