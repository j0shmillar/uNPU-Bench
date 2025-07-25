import os 
import sys

train_path = os.environ.get("AI8X_TRAIN_PATH")
if not train_path:
    raise EnvironmentError("AI8X_TRAIN_PATH is not set.")

sys.path.append(train_path)

import ai8x

import cv2
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class YoloV1DataSet(Dataset):

    def __init__(self, imgs_dir="./VOC2007/Train/JPEGImages",
                 annotations_dir="./VOC2007/Train/Annotations", img_size=224, S=12, B=2,
                 ClassesFile="./VOC2007/Train/VOC_remain_class.data", img_per_class=None,
                 train_root="./VOC2007/Train/ImageSets/Main/"):
        
        self.transfrom = transforms.Compose([
            #transforms.ToPILImage(),
            #transforms.Resize((224,224)),
            transforms.ToTensor(), # height * width * channel -> channel * height * width
            transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])

        self.img_size = img_size
        self.S = S
        self.B = B
        self.grid_cell_size = self.img_size / self.S
        self.img_per_class = img_per_class

        self.img_selection(ClassesFile, train_root)
        self.generate_img_path(self.five, imgs_dir)
        self.generate_annotation_path(self.annot, annotations_dir)
        self.generate_ClassNameToInt(ClassesFile)
        self.getGroundTruth()

    def generate_img_path(self, img_names, imgs_dir):
        img_names.sort() 
        self.img_path = []
        for img_name in img_names:
            self.img_path.append(os.path.join(imgs_dir, img_name))

    def generate_annotation_path(self, annotation_names, annotations_dir):
        annotation_names.sort() 
        self.annotation_path = []
        for annotation_name in annotation_names:
            self.annotation_path.append(os.path.join(annotations_dir, annotation_name))

    def generate_ClassNameToInt(self, ClassesFile):
        self.ClassNameToInt = {'background': 0}
        # self.instance_counting = {}
        self.IntToClassName = {0: 'background'}
        classIndex = 1
        with open(ClassesFile, 'r') as f:
            for line in f:
                line = line.replace('\n', '')
                self.ClassNameToInt[line] = classIndex
                self.IntToClassName[classIndex] = line
                classIndex = classIndex + 1
        self.Classes = classIndex 

    def img_selection(self, ClassesFile, train_root):

        def generate_class_index_fname(train_root, class_name):
            class_index_fname_buf = os.listdir(train_root)

            if class_name + "_test.txt" in class_index_fname_buf:
                return os.path.join(train_root, class_name + "_test.txt")
            else:
                return os.path.join(train_root, class_name + "_trainval.txt") 

        def one_class_img(img_index_fname):
            img_num = 0
            with open(img_index_fname, 'r') as f:
                for line in f:

                    img_valid_flag = line.strip().split(" ")[-1]
                    if img_valid_flag == "1":
                        img_No = line.strip().split(" ")[0]
                        img_num += 1
                        if img_No + ".jpg" not in self.five:
                            self.five.append(img_No + ".jpg")
                            self.annot.append(img_No + ".xml")
                            if self.img_per_class is not None and img_num == self.img_per_class:
                                return img_num
            return img_num

        self.img_index_fpaths = {}
        f =  open(ClassesFile, 'r')
        lines = f.readlines()
        for l in lines:
            class_name = l.strip()
            self.img_index_fpaths[class_name] = generate_class_index_fname(train_root, class_name)

        self.five = []
        self.annot = []
        self.Class_img_num = {}
        for class_name, img_index_fname in self.img_index_fpaths.items():
            img_num_class = one_class_img(img_index_fname)
            self.Class_img_num[class_name] = img_num_class


    def getGroundTruth(self):
        self.ground_truth = [[[list() for i in range(self.S)] for j in range(self.S)] for k in
                             range(len(self.img_path))]  
        ground_truth_index = 0
        for annotation_file in self.annotation_path:
            ground_truth = [[list() for i in range(self.S)] for j in range(self.S)]
            tree = ET.parse(annotation_file)
            annotation_xml = tree.getroot()
            width = (int)(annotation_xml.find("size").find("width").text)
            scaleX = self.img_size / width
            height = (int)(annotation_xml.find("size").find("height").text)
            scaleY = self.img_size / height
            objects_xml = annotation_xml.findall("object")
            for object_xml in objects_xml:
                class_name = object_xml.find("name").text
                if class_name not in self.ClassNameToInt: 
                    continue
                bnd_xml = object_xml.find("bndbox")
                xmin = (int)((int)(bnd_xml.find("xmin").text) * scaleX)
                ymin = (int)((int)(bnd_xml.find("ymin").text) * scaleY)
                xmax = (int)((int)(bnd_xml.find("xmax").text) * scaleX)
                ymax = (int)((int)(bnd_xml.find("ymax").text) * scaleY)
                centerX = (xmin + xmax) / 2
                centerY = (ymin + ymax) / 2
                indexI = (int)(centerY / self.grid_cell_size)
                indexJ = (int)(centerX / self.grid_cell_size)
                ClassIndex = self.ClassNameToInt[class_name]
                ClassList = [0 for i in range(self.Classes)]
                ClassList[ClassIndex] = 1
                ground_box = list([centerX / self.grid_cell_size - indexJ,centerY / self.grid_cell_size - indexI,(xmax-xmin)/self.img_size,(ymax-ymin)/self.img_size,1,xmin,ymin,xmax,ymax,(xmax-xmin)*(ymax-ymin)])
                ground_box.extend(ClassList)
                ground_truth[indexI][indexJ].append(ground_box)

            for i in range(self.S):
                for j in range(self.S):
                    if len(ground_truth[i][j]) == 0:
                        ClassList = [0 for i in range(self.Classes)]
                        ClassList[0] = 1
                        dummy_arr = [0 for _ in range(10)] + ClassList
                        self.ground_truth[ground_truth_index][i][j].append(dummy_arr)
                    else:
                        ground_truth[i][j].sort(key = lambda box: box[9], reverse=True)
                        self.ground_truth[ground_truth_index][i][j].append(ground_truth[i][j][0])

            ground_truth_index = ground_truth_index + 1
        self.ground_truth = torch.Tensor(self.ground_truth).float()

    def __getitem__(self, item):
        img_data = self.read_img(item)
        img_data = self.transfrom(img_data)
        return img_data, self.ground_truth[item]

    def __len__(self):
        return len(self.img_path)

    def read_img(self, item):
        img_data = cv2.imread(self.img_path[item])
        img_data = cv2.resize(img_data, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return img_data

def YoloV1_getdataset(data, load_train=True, load_test=True):

    (data_dir, args) = data

    if load_train:
        train_dataset = YoloV1DataSet() 
        print(f'Train dataset length: {len(train_dataset)}\n')
    else:
        train_dataset = None

    if load_test:
        train_dataset = YoloV1DataSet() 
        print(f'Test dataset length: {len(test_dataset)}\n')
    else:
        test_dataset = None

    return train_dataset, test_dataset

def YoloV1_224_getdataset(data, load_train=True, load_test=True):
    return YoloV1_getdataset(data, load_train, load_test, im_size=(224, 224), use_memory=False)

def YoloV1_128_getdataset(data, load_train=True, load_test=True):
    return YoloV1_getdataset(data, load_train, load_test, im_size=(128, 128), use_memory=False)

def YoloV1_96_getdataset(data, load_train=True, load_test=True):
    return YoloV1_getdataset(data, load_train, load_test, im_size=(96, 96), use_memory=False)

datasets = [
    {
        'name': 'yolov1dataset_256',
        'input': (3,224,224),
        'output': ('person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'),
        'loader': YoloV1_224_getdataset
    },
    {
        'name': 'yolov1dataset_128',
        'input': (3,128,128),
        'output': ('person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'),
        'loader': YoloV1_128_getdataset
    },
    {
        'name': 'yolov1dataset_96',
        'input': (3,96,96),
        'output': ('person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'),
        'loader': YoloV1_96_getdataset
    },
]