import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  
sys.path.insert(0, os.path.join(PROJECT_ROOT, "../../", "ai8x-training"))

import torch.nn as nn
import torch

import ai8x
from ai8x import Conv2d
from ai8x import FusedConv2dReLU, FusedMaxPoolConv2dReLU, FusedConv2dBNReLU, FusedMaxPoolConv2dBNReLU

ai8x.set_device(85, 0, True)

# class Yolov1_net(nn.Module):

#     def __init__(self, B=2, num_classes=2, bias=False, **kwargs):
#         super().__init__()
#         print("YOLO v1 Model_Z {} class (96 input), {} bounding boxes.".format(num_classes, B))
#         self.B = B
#         self.Classes_Num = num_classes

#         self.Conv_96 = FusedConv2dReLU(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
#         self.Conv_48 = FusedMaxPoolConv2dReLU(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
       
#         self.Conv_24_1 = FusedMaxPoolConv2dReLU(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
#         self.Conv_24_2 = FusedConv2dBNReLU(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
#         self.Conv_24_3 = FusedConv2dBNReLU(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
#         self.Conv_24_4 = FusedConv2dBNReLU(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
#         self.Conv_24_5 = FusedConv2dBNReLU(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
#         self.Conv_24_6 = FusedConv2dBNReLU(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        
#         self.Conv_12_1 = FusedMaxPoolConv2dBNReLU(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
#         self.Conv_12_2 = FusedConv2dBNReLU(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
#         self.Conv_12_3 = FusedConv2dBNReLU(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
#         self.Conv_12_4 = FusedConv2dBNReLU(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)    
#         self.Conv_12_5 = FusedConv2dBNReLU(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)  
#         self.Conv_12_6 = FusedConv2dBNReLU(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        
#         self.Conv_7_1 = FusedConv2dBNReLU(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
#         self.Conv_7_2 = FusedConv2dBNReLU(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        
#         self.Conv_Res_1 = FusedConv2dBNReLU(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
#         self.Conv_Res_2 = FusedConv2dBNReLU(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
#         self.Conv_Res_3 = FusedConv2dBNReLU(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
#         self.Conv_Res_4 = Conv2d(in_channels=16, out_channels=self.B * 5 + self.Classes_Num, kernel_size=1, stride=1, padding=0, bias=True, wide=True, **kwargs)

#     def forward(self, x):
#         x = self.Conv_96(x)
#         x = self.Conv_48(x)
#         x = self.Conv_24_1(x)
#         x = self.Conv_24_2(x)
#         x = self.Conv_24_3(x)
#         x = self.Conv_24_4(x)
#         x = self.Conv_24_5(x)
#         x = self.Conv_24_6(x)
#         x = self.Conv_12_1(x)
#         x = self.Conv_12_2(x)
#         x = self.Conv_12_3(x)
#         x = self.Conv_12_4(x)  
#         x = self.Conv_12_5(x)
#         x = self.Conv_12_6(x)
#         x = self.Conv_7_1(x)
#         x = self.Conv_7_2(x)
#         x = self.Conv_Res_1(x)
#         x = self.Conv_Res_2(x)
#         x = self.Conv_Res_3(x)
#         x_fl_output = self.Conv_Res_4(x)
#         x = x_fl_output.permute(0, 2, 3, 1)
#         class_possible = torch.softmax(x[:, :, :, 10:], dim=3)
#         x = torch.cat((torch.sigmoid(x[:,:,:,:10]), class_possible), dim=3)
#         return x, x_fl_output


class Yolov1_net(nn.Module):

    def __init__(self, B=2, num_classes=2, bias=True, **kwargs):
        super().__init__()
        print("YOLO v1 Model_Z {} class (96 input), {} bounding boxes.".format(num_classes, B))
        self.B = B
        self.Classes_Num = num_classes

        self.Conv_96 = FusedConv2dReLU(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.Conv_48 = FusedMaxPoolConv2dReLU(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
       
        self.Conv_24_1 = FusedMaxPoolConv2dReLU(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False, **kwargs)
        self.Conv_24_2 = FusedConv2dBNReLU(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_24_3 = FusedConv2dBNReLU(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_24_4 = FusedConv2dBNReLU(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_24_5 = FusedConv2dBNReLU(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_24_6 = FusedConv2dBNReLU(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        
        self.Conv_12_1 = FusedMaxPoolConv2dBNReLU(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_12_2 = FusedConv2dBNReLU(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_12_3 = FusedConv2dBNReLU(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_12_4 = FusedConv2dBNReLU(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)    
        self.Conv_12_5 = FusedConv2dBNReLU(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)  
        self.Conv_12_6 = FusedConv2dBNReLU(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        
        self.Conv_7_1 = FusedConv2dBNReLU(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_7_2 = FusedConv2dBNReLU(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=bias, batchnorm='NoAffine', **kwargs)
        
        self.Conv_Res_1 = FusedConv2dBNReLU(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_Res_2 = FusedConv2dBNReLU(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
        self.Conv_Res_3 = FusedConv2dBNReLU(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0, bias=bias, batchnorm='NoAffine', **kwargs)
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
        
        # Separate the components
        bbox_preds = self.identity_1(x[:, :, :, :10])  # Bounding box predictions
        class_preds = self.identity_2(x[:, :, :, 10:])  # Class predictions
        
        return bbox_preds, class_preds, x_fl_output

def ai85yolo96(pretrained=False, **kwargs):
    assert not pretrained
    return Yolov1_net(**kwargs)

models = [
    {
        'name': 'ai85yolo96',
        'min_input': 1,
        'dim': 2,
    },
]

def ai85yolo96_save():
    model = Yolov1_net()
    torch.save(model.state_dict(), 'model/yolo/yolov1.pth')

if __name__ == "__main__":
    ai85yolo96_save()
