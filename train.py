'''
Author : Na Rae Baek
Author email : naris27@dgu.ac.kr
Github : https://github.com/ban2aru
License : MIT license
Modified Year-Month : 2022-04
Version : 1.0

Description : train.py
The main code for training ResNet using Pytorch
'''
from .utils import get_train_dataloader, get_valid_dataloader
import time
import torch
import os
import copy
import sys
from torchvision.utils import make_grid


def train(args, start_epoch, criterion, optimizer, train_scheduler, best_loss, writer, model, start_strf) :
    training_loader = get_train_dataloader(args)
    valid_loader = get_valid_dataloader(args)
    start = time.time()
    save_point = './checkpoint/'+ start_strf
    train_txt = args.txt_dir + '/' + start_strf
    if args.gpu :
        model = model.cuda()

    best_model = model
    if not os.path.isdir(args.txt_dir) :
        os.mkdir(args.txt_dir)
    if not os.path.isdir(train_txt):
        os.mkdir(train_txt)
    
    sys.stdout = open(train_txt+'/train.txt', 'w')

    for epoch in range(start_epoch+1 , start_epoch + args.epochs+1) :

        model.train()

        running_train_loss, running_correct, total = 0.0, 0.0, 0

        for batch_index, (imgs, labels) in enumerate(training_loader) :
            if args.gpu :
                imgs = imgs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()

            #Tensorboard
            img_grid = make_grid(imgs)
            writer.add_image('Train_image', img_grid)
            writer.add_graph(model, imgs)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
        
            _, predicted = torch.max(outputs.data, 1)

            loss.backward()
                
            optimizer.step()
            running_train_loss += loss.item()
            running_correct += predicted.eq(labels.data).cpu().sum()
            total += labels.size(0)
            n_iter = (epoch-1) * len(training_loader) + batch_index + 1

            #Tensorboard
            writer.add_scalar('Train/Train_loss_iteration', loss.item(), n_iter)
        
        print('||'+time.strftime('%Y-%m-%d-%H:%M:%S')+'||Training Step')
        print('||'+time.strftime('%Y-%m-%d-%H:%M:%S')+'||Epoch[{epoch}/{epochs}] Train_Loss : {:0.4f} Train_Accuracy : {:0.4f} LearningRate : {:0.7f}'.format(
            running_train_loss/ len(training_loader.dataset),
            running_correct.float() / len(training_loader.dataset) * 100,
            optimizer.param_groups[0]['lr'],
            epoch = epoch,
            epochs = args.epochs,              
        ))

        writer.add_scalar('Train/Train_loss_epoch', running_train_loss/ len(training_loader.dataset), epoch)
        writer.add_scalar('Train/Train_accuracy', running_correct.float() / len(training_loader.dataset) * 100, epoch)
        writer.add_scalar('Train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
            
           
        train_scheduler.step()  

        # Valid step (@torch.no_grad())
        model.eval()
        
        with torch.no_grad():
            valid_loss, valid_correct = 0.0, 0.0

            for batch_index, (imgs, labels) in enumerate(valid_loader) :
                if args.gpu :
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                valid_correct += predicted.eq(labels).cpu().sum()

            print('||'+time.strftime('%Y-%m-%d-%H:%M:%S')+'||Validation Step')
            print('||'+time.strftime('%Y-%m-%d-%H:%M:%S')+'||Epoch[{epoch}/{epochs}] Validation_Loss : {:0.4f} Validation_Accuracy : {:0.4f}'.format(
                valid_loss / len(valid_loader.dataset),
                valid_correct.float() / len(valid_loader.dataset) * 100,
                epoch = epoch,
                epochs = args.epochs,             
            ))

            writer.add_scalar('Valid/Valid_loss', valid_loss / len(valid_loader.dataset), epoch)
            writer.add_scalar('Valid/Valid_accuracy', valid_correct.float() / len(valid_loader.dataset) * 100, epoch) 

            epoch_valid_loss = valid_loss / len(valid_loader.dataset)

            if epoch_valid_loss < best_loss :
                print('||'+time.strftime('%Y-%m-%d-%H:%M:%S')+'||Saving Model Epoch %s' %(epoch))
                best_loss = epoch_valid_loss
                best_model = copy.deepcopy(model)
                state = {
                    'model' : best_model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'train_loss' : running_train_loss/ len(training_loader.dataset),
                    'valid_loss' : best_loss,
                    'epoch' : epoch
                }

                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')

                if not os.path.isdir(save_point):
                    os.mkdir(save_point)
                torch.save(state, save_point+'/%s-%sepoch.pth' %(args.net,epoch))

        if epoch == args.epochs :
            print('||'+time.strftime('%Y-%m-%d-%H:%M:%S')+'||Saving Model Epoch%d' %(epoch))
            torch.save(state, save_point+'/%s-%s.pth' %(args.net,epoch))
            
    train_time = time.time() - start
    print('||'+time.strftime('%Y-%m-%d-%H:%M:%S')+'||Training Completed : {:.2f}sec'.format(train_time))

    sys.stdout.close()



def train_resume(args, model) :
    print('||'+time.strftime('%Y-%m-%d-%H:%M:%S')+'||Resume training from checkpoint')
    assert os.path.isfile(args.chk_dir), 'Error : chekpoint directory does not exist' 

    checkpoint = torch.load(args.chk_dir)
    model.load_state_dict(checkpoint['model'])
    best_loss = checkpoint['valid_loss']
    start_epoch = checkpoint['epoch']

    return model, best_loss, start_epoch