import math

import torch
import torch.nn as nn

class Yolov1_Loss(nn.Module):

    def __init__(self, S=12, B=2, Classes=2, l_coord=5, l_noobj=0.5): 
        super(Yolov1_Loss, self).__init__()
        self.S = S
        self.B = B
        self.Classes = Classes
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def iou(self, bounding_box, ground_box, gridX, gridY, img_size=96, grid_size=8):
        predict_box = list([0,0,0,0])
        predict_box[0] = (int)(gridX + bounding_box[0] * grid_size)
        predict_box[1] = (int)(gridY + bounding_box[1] * grid_size)
        predict_box[2] = (int)(bounding_box[2] * img_size)
        predict_box[3] = (int)(bounding_box[3] * img_size)

        predict_coord = list([max(0, predict_box[0] - predict_box[2] / 2), max(0, predict_box[1] - predict_box[3] / 2),min(img_size - 1, predict_box[0] + predict_box[2] / 2), min(img_size - 1, predict_box[1] + predict_box[3] / 2)])
        predict_Area = (predict_coord[2] - predict_coord[0]) * (predict_coord[3] - predict_coord[1])

        ground_coord = list([ground_box[5],ground_box[6],ground_box[7],ground_box[8]])
        ground_Area = (ground_coord[2] - ground_coord[0]) * (ground_coord[3] - ground_coord[1])

        CrossLX = max(predict_coord[0], ground_coord[0])
        CrossRX = min(predict_coord[2], ground_coord[2])
        CrossUY = max(predict_coord[1], ground_coord[1])
        CrossDY = min(predict_coord[3], ground_coord[3])

        if CrossRX < CrossLX or CrossDY < CrossUY:
            return 0

        interSection = (CrossRX - CrossLX) * (CrossDY - CrossUY)
            
        return interSection / (predict_Area + ground_Area - interSection)

    def forward(self, bounding_boxes, ground_truth, batch_size=32, grid_size=8, img_size=96):  
        loss = 0
        loss_coord = 0
        loss_confidence = 0
        loss_classes = 0
        iou_sum = 0
        object_num = 0
        mseLoss = nn.MSELoss()

        for batch in range(len(bounding_boxes)):
            for i in range(self.S): 
                for j in range(self.S):
                    if bounding_boxes[batch][i][j][4] < bounding_boxes[batch][i][j][9]:
                        predict_box = bounding_boxes[batch][i][j][5:]
                        loss = loss + self.l_noobj * torch.pow(bounding_boxes[batch][i][j][4], 2)
                        loss_confidence += self.l_noobj * torch.pow(bounding_boxes[batch][i][j][4], 2).item()
                    else:
                        predict_box = bounding_boxes[batch][i][j][0:5]
                        predict_box = torch.cat((predict_box, bounding_boxes[batch][i][j][10:]), dim=0)
                        loss = loss + self.l_noobj * torch.pow(bounding_boxes[batch][i][j][9], 2)
                        loss_confidence += self.l_noobj * torch.pow(bounding_boxes[batch][i][j][9], 2).item()

                    ground_box_data = ground_truth[batch][i][j][0]
                    
                    if ground_box_data[9] == 0: 
                        loss = loss + self.l_noobj * torch.pow(predict_box[4], 2)
                        loss_confidence += self.l_noobj * torch.pow(predict_box[4], 2).item()
                    else: 
                        object_num += 1
                        iou = self.iou(predict_box, ground_box_data, j * grid_size, i * grid_size,
                                    img_size=img_size, grid_size=grid_size)
                        iou_sum += iou

                        sqrt_gt_w = torch.sqrt(torch.clamp(ground_box_data[2], min=0.0) + 1e-8)
                        sqrt_pr_w = torch.sqrt(torch.clamp(predict_box[2], min=0.0) + 1e-8)
                        sqrt_gt_h = torch.sqrt(torch.clamp(ground_box_data[3], min=0.0) + 1e-8)
                        sqrt_pr_h = torch.sqrt(torch.clamp(predict_box[3], min=0.0) + 1e-8)

                        coord_loss = self.l_coord * (
                            torch.pow(ground_box_data[0] - predict_box[0], 2) +
                            torch.pow(ground_box_data[1] - predict_box[1], 2) +
                            torch.pow(sqrt_gt_w - sqrt_pr_w, 2) +
                            torch.pow(sqrt_gt_h - sqrt_pr_h, 2)
                        )
                        loss += coord_loss
                        loss_coord += coord_loss.item()

                        conf_loss = torch.pow(ground_box_data[4] - predict_box[4], 2)
                        loss += conf_loss
                        loss_confidence += conf_loss.item()

                        ground_class = ground_box_data[10:]  
                        predict_class = bounding_boxes[batch][i][j][self.B * 5:] 

                        if ground_class.numel() > 0 and predict_class.numel() > 0:
                            class_loss = mseLoss(ground_class, predict_class) * self.Classes
                            loss += class_loss
                            loss_classes += class_loss.item()

        return loss, loss_coord, loss_confidence, loss_classes, iou_sum, object_num