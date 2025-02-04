import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  
sys.path.insert(0, os.path.join(PROJECT_ROOT, "../../", "ai8x-training"))

import importlib

import random
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

from yolo_dataset import YoloV1DataSet
from yolov1_loss_function import Yolov1_Loss

mod = importlib.import_module("yolov1_96_max78000")

import ai8x

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', type=int, default=400, help='Maximum training epoch.')
parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=16, help='Minibatch size.')
parser.add_argument('--img_train', type=str, default="300", help='Image number per class for training.')
parser.add_argument('--gpu', type=int, default=0, help='Use which gpu to train the model.')
parser.add_argument('--exp', type=str, default="NOQAT_gridsize12", help='Experiment name.')
parser.add_argument('--seed', type=int, default=7, help='Random seed.')
args = parser.parse_args()

torch.set_default_dtype(torch.float32)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def log_init():
    import log_utils, time, glob, logging, sys

    fdir0 = os.path.join("log", args.exp + '-{}'.format(time.strftime("%Y%m%d-%H%M%S")))
    log_utils.create_exp_dir(fdir0, scripts_to_save=glob.glob('*.py'))
    args.output_dir = fdir0

    logger = log_utils.get_logger(tag=(args.exp), log_level=logging.INFO)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(fdir0, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logger.info("hyperparam = %s", args)

    return logger
    
def dataset_init(logger):
    dataset_root = "./model/yolo/data/VOC2007"
    
    dataSet = YoloV1DataSet(
        imgs_dir=f"{dataset_root}/JPEGImages",
        annotations_dir=f"{dataset_root}/Annotations",
        ClassesFile=f"{dataset_root}/VOC_person.data",
        train_root=f"{dataset_root}/ImageSets/Main/",
        ms_logger=logger,
        img_size=96
    )
    
    dataLoader = DataLoader(dataSet, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return dataSet, dataLoader

def train(logger):

    dataSet, dataLoader = dataset_init(logger)

    ai8x.set_device(device=85, simulate=False, round_avg=False)
    
    Yolo = mod.Yolov1_net(num_classes=dataSet.Classes, bias=True)
    Yolo = Yolo.to(args.device)

    loss_function = Yolov1_Loss().to(args.device)
    optimizer = torch.optim.SGD(Yolo.parameters(),lr=args.lr,momentum=0.9,weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50, 100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000,40000],gamma=0.8)

    num_epochs = args.max_epoch

    for epoch in tqdm(range(0, num_epochs)):

        loss_sum = 0
        loss_coord = 0
        loss_confidence = 0
        loss_classes = 0
        epoch_iou = 0
        epoch_object_num = 0
        grid_size, img_size = dataSet.grid_cell_size, dataSet.img_size

        for _, batch_train in tqdm(enumerate(dataLoader)):
            optimizer.zero_grad()
            train_data = batch_train[0].float().to(args.device)
            train_data.requires_grad = True
            label_data = batch_train[1].float().to(args.device)
            bb_pred, _ = Yolo(train_data)
            loss = loss_function(bounding_boxes=bb_pred, ground_truth=label_data, grid_size=grid_size, img_size=img_size)
            batch_loss = loss[0]
            loss_coord = loss_coord + loss[1]
            loss_confidence = loss_confidence + loss[2]
            loss_classes = loss_classes + loss[3]
            epoch_iou = epoch_iou + loss[4]
            epoch_object_num = epoch_object_num + loss[5]
            batch_loss.backward()
            optimizer.step()
            batch_loss = batch_loss.item()
            loss_sum = loss_sum + batch_loss

        scheduler.step()

        if epoch % 1 == 0:
            torch.save(Yolo.state_dict(), os.path.join(args.output_dir, "hpd_noaffine_shift_maxim_yolo_ep{}.pth".format(epoch)))
        batch_num = len(dataLoader)
        avg_loss= loss_sum/batch_num
        logger.info("epoch : {} ; loss : {} ; avg_loss: {}; IOU: {}; class SSE: {}".format(epoch, loss_sum, avg_loss, epoch_iou, loss_classes))
        epoch = epoch + 1
    torch.save(Yolo.state_dict(), os.path.join(args.output_dir, "hpd_noaffine_shift_maxim_yolo_ep{}.pth".format(epoch)))


def main():
    logger = log_init()
    args.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    logger.info('Running on device: {}'.format(args.device))
    train(logger)

if __name__ == "__main__":
    main()

