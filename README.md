# Pytorch-ResNet
Pytorch implementation of ResNet network using custom dataset.

## Requriements
This is my environments

- python 3.9.9
- torch 1.11.0
- torchvision 0.12.0
- albumentations 1.1.0
- natsort 8.1.0

You can install using pip like this.
```bash
pip install albumentations
```

## Custom Dataset
Your custom dataset need to be structured like this
```bash
[dataset folder name]
  |-train
    |-[0]
    |-[1]
    ...
    |-[n]
  |-val
    |-[0]
    |-[1]
    ...
    |-[n]
  |-test
    |-[0]
    |-[1]
    ...
    |-[n] 
```

## Training
```bash
# You can use more option, check argument

# train from scratch
python main.py --net='ResNet18' --phase='Train' --num_classes=10 --lr=0.1 --epochs=100

# resume training
python main.py --resume=True --net='ResNet18' --phase='Train' --num_classes=10 --lr=0.1 --epochs=100

# fine-tuning (ImageNet)
python main.py --pretrained_model=True --net='ResNet18' --phase='Train' --num_classes=10 --lr=0.1 --epochs=100
```
Train loss, accuracy per epoch and Validation loss, accuracy per epoch will save "txt_dir".
Default directory is "./txt/####(timestamp)".

### (Optional) Tensorboard
You need to install tensorboard.
Default log directory is "./log/####(timestamp)".
```bash
tensorbaord --logdir='log_dir'
```

## Testing
```bash
python main.py --phase='Test' --chk_dir='checkpoint_dir'
```

