'''
Author : Na Rae Baek
Author email : naris27@dgu.ac.kr
Github : https://github.com/ban2aru
License : MIT license
Modified Year-Month : 2022-04

Description : test.py
The main code for testing ResNet using Pytorch
'''
from .utils import get_test_dataloader
import torch
import time

def test(args, model):
    test_loader = get_test_dataloader(args)

    checkpoint = torch.load(args.chk_dir)
    model = checkpoint['model']

    if args.gpu :
        model = model.cuda()

    model.eval()

    test_correct_top1, test_correct_top5 = 0.0, 0.0
    
    with torch.no_grad():
        for batch_index, (imgs, labels) in enumerate(test_loader):

            if args.gpu :
                imgs = imgs.cuda()
                labels = labels.cuda()
            
            outputs = model(imgs)
            _, predicted = outputs.topk(5, 1, largest=True, sorted=True)

            labels = labels.view(labels.size(0), -1).expand_as(predicted)
            correct = predicted.eq(labels).float()

            if args.num_classes > 5 :
                test_correct_top5 += correct[:, :5].sum()
            
            test_correct_top1 += correct[:, :1].sum()

        print('||'+time.strftime('%Y-%m-%d-%H:%M:%S')+'||Test Step')
        print('||'+time.strftime('%Y-%m-%d-%H:%M:%S')+'||Test Acc@1 %.4f' %(test_correct_top1/len(test_loader.dataset)))
        if args.num_classes > 5 :
            print('||'+time.strftime('%Y-%m-%d-%H:%M:%S')+'||Test Acc@5 %.4f' %(test_correct_top5/len(test_loader.dataset)))
        print('||'+time.strftime('%Y-%m-%d-%H:%M:%S')+'||Parameter numbers %d' %(sum(p.numel() for p in model.parameters())))
