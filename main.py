'''
Author : Na Rae Baek
Author email : naris27@dgu.ac.kr
Github : https://github.com/ban2aru
License : MIT license
Modified Year-Month : 2022-04
Version : 1.0

Description : main.py
The main code for training/testing ResNet using Pytorch
'''

import argparse
import torch.nn as nn
import torch.optim as optim

from Util.utils import get_network, CosineAnnealingWarmUpSchedule
from Util.train import train, train_resume
from Util.test import test
import time
import os

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Pytorch Training')

parser.add_argument('--pretrained_model', default=False, type=bool, help='whether or not use ImageNet-pretrained model')
parser.add_argument('--resume', default=False, type=bool, help='wheter or not resume training (If true, check chk_dir)')

parser.add_argument('--gpu', default=False, type=bool, help='whether or not use gpu')
parser.add_argument('--shuffle', default=True, type=bool, help='wheter or not shuffle when training')
parser.add_argument('--num_workers', default=2, type=int, help='number of workers(for data load multi-processing)')

# parser.add_argument('--net', default='ResNet18', required=True, help='type of network')
parser.add_argument('--net', default='ResNet18', help='type of network')
parser.add_argument('--phase', default='Train', type=str, choices=['Train', 'Test'], help='choose train phase or test phase')
parser.add_argument('--log_dir', default = './log', type=str, help='tensorboard log directory')
parser.add_argument('--txt_dir', default= './txt', type=str, help='train log txtfile directory')
parser.add_argument('--num_classes', default=10, type=int, help='number of class')
parser.add_argument('--dataset_path', default='./data', type=str, help='trianing/test directory(parent folder)')
parser.add_argument('--batch_size', default=16, type=int, help='number of batch_size')
parser.add_argument('--height', default=28, type=int, help='image height')
parser.add_argument('--width', default=28, type=int, help='image width')

# argument for training hyperparameter
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help='training epochs')

# argument for image_augmentation 
parser.add_argument('--RandomCrop', default=False, type=bool, help='whether or not use RandomCrop (for image augmentation)')
parser.add_argument('--ColorJitter', default=False, type=bool, help='whether or not use ColorJitter (for image augmentation)')
parser.add_argument('--HorizonFlip', default=False, type=bool, help='whether or not use HorizonFlip (for image augmentation)')
parser.add_argument('--VerticalFlip', default=False, type=bool, help='whether or not use VerticalFlip (for image augmentation)')
parser.add_argument('--Affine', default=False, type=bool, help='whether or not use Affine (for image augmentation)')
parser.add_argument('--Blur', default=False, type=bool, help='whether or not use Blur (for image augmentation)')
parser.add_argument('--GaussianNoise', default=False, type=bool, help='whether or not use GaussianNoise (for image augmentation)')
parser.add_argument('--CutOut', default=False, type=bool, help='whether or not use CutOut (for image augmentation)')

# argument for test
parser.add_argument('--chk_dir', default='./checkpoint/20220407183239/ResNet18-4.pth', type=str, help='checkpoint directory')
args = parser.parse_args()


if __name__ == '__main__' :
    model = get_network(args)

    start_epoch, best_loss = 0, 100.0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = CosineAnnealingWarmUpSchedule(optimizer=optimizer, cycle_T0=50, cycle_Tmult=1., max_lr=0.1, min_lr=1e-7, warmup_epoch=10, gamma=0.5, last_epoch=-1)

    if args.resume==True and args.phase=='Train' :
        assert args.phase=='Train', 'Error : args.phase is not train'
        model, best_loss, start_epoch = train_resume(args, model) 
        
        if os.path.isdir(args.log_dir) :
            start_strf = time.strftime('%Y%m%d%H%M%S')
            writer = SummaryWriter(args.log_dir)
        else :
            print('Error : log_dir is not exist. Tensorboard will start anew')
            start_strf = time.strftime('%Y%m%d%H%M%S')
            log_dir = args.log_dir+'/'+start_strf
            writer = SummaryWriter(log_dir)            
        
        train(args, start_epoch=start_epoch, criterion=criterion, optimizer=optimizer, train_scheduler=train_scheduler, best_loss=best_loss, writer=writer, model=model, start_strf=start_strf)
        writer.close()

    elif args.resume==False and args.phase=='Train':
        start_strf = time.strftime('%Y%m%d%H%M%S')
        log_dir = args.log_dir+'/'+start_strf
        writer = SummaryWriter(log_dir)
        train(args, start_epoch=start_epoch, criterion=criterion, optimizer=optimizer, train_scheduler=train_scheduler, best_loss=best_loss, writer=writer, model=model, start_strf=start_strf)
        writer.close()

    elif args.phase=='Test':
        test(args, model=model)
    

