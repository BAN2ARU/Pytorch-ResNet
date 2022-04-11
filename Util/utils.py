'''
Author : Na Rae Baek
Author email : naris27@dgu.ac.kr
Github : https://github.com/ban2aru
License : MIT license
Modified Year-Month : 2022-04

Description : utils.py
The main code for utils using Pytorch
'''
import sys
import albumentations as A
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import cv2
import natsort
import glob
import os
import torch
from torch.optim.lr_scheduler import _LRScheduler
import math
import logging
import logging.handlers

def get_network(args):
    if (args.net == 'ResNet18'):
        from Model.ResNet import ResNet18
        net = ResNet18(pretrained_model=args.pretrained_model, num_classes=args.num_classes)
    elif (args.net == 'ResNet34'):
        from Model.ResNet import ResNet34
        net = ResNet34(pretrained_model=args.pretrained_model, num_classes=args.num_classes)
    elif (args.net == 'ResNet50'):
        from Model.ResNet import ResNet50
        net = ResNet50(pretrained_model=args.pretrained_model, num_classes=args.num_classes)
    elif (args.net == 'ResNet101'):
        from Model.ResNet import ResNet101
        net = ResNet101(pretrained_model=args.pretrained_model, num_classes=args.num_classes)
    elif (args.net == 'ResNet152'):
        from Model.ResNet import ResNet152
        net = ResNet152(pretrained_model=args.pretrained_model, num_classes=args.num_classes)
    
    else :
        print('Error : Please check supported network')
        sys.exit()

    if args.gpu :
        net = net.cuda()

    return net

class AlbumentDataset(Dataset) :

    def __init__(self, dataset_path, transform=None) :
        self.dataset_path = dataset_path
        self.img_list = natsort.natsorted(glob.glob(self.dataset_path+'/**/*.png')+glob.glob(self.dataset_path+'/**/*.jpg')+glob.glob(self.dataset_path+'/**/*.bmp'))
        self.label_list = [int(os.path.basename(os.path.dirname(img))) for img in self.img_list]
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        label = self.label_list[idx]

        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None :
            augment = self.transform(image=img)
            img = augment['image']
    
        return img, label

def get_mean_std(args):
    
    train_dataset = AlbumentDataset(dataset_path=args.dataset_path+'/train', transform=A.Compose([A.Resize(args.height, args.width, p=1.0), ToTensorV2()])) 
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=False, num_workers=0)
    mean, mean_squared, std  = 0, 0, 0

    for img, _ in train_dataloader :
        img = img*1.0
        batch_sample = img.size(0)
        img = img.view(batch_sample, img.size(1), -1)
        mean += img.mean(2).sum(0)
        mean_squared += (img**2).mean(2).sum(0)
        

    mean /= len(train_dataloader.dataset)
    mean_squared /= len(train_dataloader.dataset)

    std = torch.sqrt(mean_squared - mean**2)

    mean /= 255
    std /= 255

    return mean, std

def get_train_dataloader(args):
    temp = []
    if (args.RandomCrop==True):
        temp.extend([A.Resize(round(args.height*1.5), round(args.width*1.5))])
        temp.extend([A.RandomCrop(args.height, args.width)])
    if (args.ColorJitter==True):
        temp.extend([A.ColorJitter(brightness=(0.2, 3), contrast=(0.2, 3), saturation=(0.2,3), hue=(-0.5,0.5), p=0.5)])
    if (args.HorizonFlip==True):
        temp.extend([A.HorizontalFlip(p=0.5)])
    if (args.VerticalFlip==True):
        temp.extend([A.VerticalFlip(p=0.5)])
    if (args.Affine==True):
        temp.extend([A.ShiftScaleRotate(p=0.5)])
    if (args.Blur==True):
        temp.extend([A.Blur(p=0.5)])
    if (args.GaussianNoise == True):
        temp.extend([A.GaussNoise(p=0.5)])   
    if (args.CutOut==True):
        temp.extend([A.Cutout(num_holes = 8, max_h_size=round(args.height*0.1), max_w_size=round(args.width*0.1), p=0.5)])
    
    mean, std = get_mean_std(args)
    temp.extend([A.Resize(args.height, args.width, p=1.0), A.Normalize(mean=mean, std=std, p=1.0), ToTensorV2(p=1.0)])
    train_transform = A.Compose(temp)

    train_dataset = AlbumentDataset(dataset_path=args.dataset_path+'/train', transform=train_transform)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=args.shuffle, num_workers=args.num_workers, batch_size=args.batch_size)

    return train_dataloader

def get_test_dataloader(args):

    mean, std = get_mean_std(args)
    test_transform = A.Compose([A.Resize(args.height, args.width, p=1.0), A.Normalize(mean=mean, std=std, p=1.0), ToTensorV2(p=1.0)])

    test_dataset = AlbumentDataset(dataset_path=args.dataset_path+'/test', transform=test_transform)
    test_dataloader = DataLoader(test_dataset, shuffle=args.shuffle, num_workers=args.num_workers, batch_size=args.batch_size)

    return test_dataloader

def get_valid_dataloader(args):

    mean, std = get_mean_std(args)
    valid_transform = A.Compose([A.Resize(args.height, args.width, p=1.0), A.Normalize(mean=mean, std=std, p=1.0), ToTensorV2(p=1.0)])

    valid_dataset = AlbumentDataset(dataset_path=args.dataset_path+'/valid', transform=valid_transform)
    valid_dataloader = DataLoader(valid_dataset, shuffle=args.shuffle, num_workers=args.num_workers, batch_size=args.batch_size)

    return valid_dataloader

class CosineAnnealingWarmUpSchedule(_LRScheduler) :

    def __init__(self, optimizer, cycle_T0, cycle_Tmult=1., max_lr=0.1, min_lr=1e-7, warmup_epoch=5, gamma=0.5, last_epoch=-1) :
        assert warmup_epoch < cycle_T0

        self.cycle_T0 = cycle_T0
        self.cycle_Tmult = cycle_Tmult
        self.max_lr_base = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_epoch = warmup_epoch
        self.gamma = gamma

        self.cycle_Tcur = cycle_T0
        self.cycle = 0
        self.cycle_step = last_epoch

        super(CosineAnnealingWarmUpSchedule, self).__init__(optimizer, last_epoch)

        self.lr_init()

    def lr_init(self) :
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self) :
        if self.cycle_step == -1:
            return self.base_lrs
        elif self.cycle_step < self.warmup_epoch : 
            return [(self.max_lr - lr_base)*self.cycle_step / self.warmup_epoch + lr_base for lr_base in self.base_lrs]
        else :
            return [lr_base + (self.max_lr - lr_base) * (1 + math.cos(math.pi * (self.cycle_step-self.warmup_epoch)/ (self.cycle_Tcur - self.warmup_epoch))) / 2 for lr_base in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None :
            epoch = self.last_epoch + 1
            self.cycle_step = self.cycle_step + 1
            if self.cycle_step >= self.cycle_Tcur :
                self.cycle += 1
                self.cycle_step = self.cycle_step - self.cycle_Tcur
                self.cycle_Tcur = int((self.cycle_Tcur - self.warmup_epoch) * self.cycle_Tmult) + self.warmup_epoch

        else :
            if epoch >= self.cycle_T0 :
                if self.cycle_mult == 1. :
                    self.cycle_step = epoch % self.cycle_T0
                    self.cycle = epoch // self.cycle_T0

                else :
                    n = int(math.log((epoch / self.cycle_T0 * (self.cycle_Tmult - 1) + 1), self.cycle_Tmult))
                    self.cycle = n
                    self.cycle_step = epoch - int(self.cycle_T0 * (self.cycle_Tmult ** n -1) / (self.cycle_Tmult - 1))
                    self.cycle_Tcur = self.cycle_T0 * self.cycle_Tmult ** (n)

            else :
                self.cycle_Tcur = self.cycle_T0
                self.cycle_step = epoch

        self.max_lr = self.max_lr_base * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        
def make_logger(logger_name, log_file) :
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s||%(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger