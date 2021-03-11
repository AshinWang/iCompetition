###########################################################################
# Step 1: Inport Package
###########################################################################

import albumentations as A
import os.path as osp
import numpy as np
import pandas as pd
import pathlib, sys, os, random, time, glob
import numba, cv2, gc
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torchvision
from torchvision import transforms as T
from albumentations.pytorch import ToTensorV2
import shutil
from shutil import copyfile
from sklearn.model_selection import train_test_split
import PIL
from PIL import Image, ImageChops, ImageEnhance
import matplotlib.pyplot as plt
from pytorch_toolbelt import losses as L
import pytest
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')


###########################################################################
# Step 2: Data Augmentation
###########################################################################

# Read image path in dataset
def MakeData(PATH):
    df = pd.DataFrame()
    X = []
    y = []
    for i in (list({i.split('.')[0] for i in os.listdir(PATH)})):
        i = i.split('.')[0]
        X.append(PATH + '{}.tif'.format(i))
        y.append(PATH + '{}.png'.format(i))
        
    df['X']  =  X
    df['y']  =  y 
    return df

# origin dataset path, and new dataset path
data_path = "../tcdata/suichang_round1_train_210120/"
train_path = '../user_data/tmp_data/dataset/train/'
val_path = '../user_data/tmp_data/dataset/val/'

data = MakeData(data_path)
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

print(len(train_data['X']),len(val_data['X']))
print(len(train_data['y']),len(val_data['y']))

# trainset
for i in tqdm(range(0, len(train_data['X']))):
    shutil.copy(train_data['X'].iloc[i], train_path)
    shutil.copy(train_data['y'].iloc[i], train_path)
print(len(os.listdir(train_path))/2)

# valset
for i in tqdm(range(0, len(val_data['X']))):
    shutil.copy(val_data['X'].iloc[i], val_path)
    shutil.copy(val_data['y'].iloc[i], val_path)
print(len(os.listdir(val_path))/2)


IMAGE_SIZE =256
class DatasetAug(D.Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

        self.len = len(paths)
        self.as_tensor = T.Compose([
            T.ToPILImage(),
            T.Resize(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

    # get data operation
    def __getitem__(self, index):
        img = cv2.imread(self.paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.paths[index].replace('.tif', '.png'),cv2.IMREAD_UNCHANGED)
        transformed = self.transform(image=img, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]
        return image, mask


train_transform = A.Compose(
    [
        A.Resize(256, 256),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        # A.RandomRotate90(p=0.5),
        # A.Transpose(p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        # A.RandomSizedCrop([96, 192],256,256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

dataset_path = '../user_data/tmp_data'
train_dataset = DatasetAug(glob.glob(osp.join(dataset_path, 'dataset/train', '*.tif')),train_transform)
print(len(train_dataset))

def save_augmentations(dataset, path, transform, idx=0, samples=2):
    dataset = copy.deepcopy(dataset)
    transform = transform
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    for i in range(1, samples):
        image, mask = dataset[idx]
        image=Image.fromarray(image)
        mask=Image.fromarray(mask)
        image.save(path+'{}_{}.tif'.format(i,idx))
        mask.save(path+'{}_{}.png'.format(i,idx))
    
for id in tqdm(range(len(train_dataset))):
    save_augmentations(train_dataset, '../user_data/tmp_data/dataset/train/', train_transform, idx=id, samples=3)


image_file_number=glob.glob('../user_data/tmp_data/dataset/train/*tif')
mask_file_number=glob.glob('../user_data/tmp_data/dataset/train/*png')
print(len(image_file_number),len(mask_file_number))

val_image_file_number=glob.glob('../user_data/tmp_data/dataset/val/*tif')
val_mask_file_number=glob.glob('../user_data/tmp_data/dataset/val/*png')
print(len(val_image_file_number),len(val_mask_file_number))


###########################################################################
# Step 3: Module Training
###########################################################################

EPOCHES = 100
BATCH_SIZE = 16
IMAGE_SIZE = 256

torch.backends.cudnn.enabled = True
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


class Trainset(D.Dataset):

    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
       

        self.len = len(paths)
        self.as_tensor = T.Compose([
            T.ToPILImage(),
            T.Resize(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

    def __getitem__(self, index):
        img = cv2.imread(self.paths[index], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
        mask = cv2.imread(self.paths[index].replace('.tif', '.png'))  - 1 
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
        augments = self.transform(image=img, mask=mask)
        return self.as_tensor(augments['image']), augments['mask'][:, :, 0].astype(np.int64)       


train_dataset = Trainset(glob.glob(osp.join(dataset_path, 'dataset/train', '*.tif')),transform)
print("---train_dataset Size---")
print(len(train_dataset))

val_dataset = Trainset(glob.glob(osp.join(dataset_path, 'dataset/val', '*.tif')),transform)
print("---val_dataset Size---")
print(len(val_dataset))

val_idx, train_idx = [], []
for i in range(len(train_dataset)):
    train_idx.append(i) 
for i in range(len(val_dataset)):
    val_idx.append(i)
           
print("---idx Size---")
print(len(train_idx),len(val_idx))

train_ds = D.Subset(train_dataset, train_idx)
valid_ds = D.Subset(val_dataset, val_idx)

# define training and validation data loaders
loader = D.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
vloader = D.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
print("---DataLoader Size---")
print(int(len(loader)), int(len(vloader)))

def get_iou(pred, mask, c=10):
    iou_result = []
    for idx in range(c):
        p = (mask == idx).int().reshape(-1)
        t = (pred == idx).int().reshape(-1)

        uion = p.sum() + t.sum()
        overlap = (p * t).sum()

        iou = 2 * overlap / (uion + 0.001)
        iou_result.append(iou.abs().data.cpu().numpy())
    return np.stack(iou_result)

def validation(model, loader, loss_fn):
    model.eval()
    with torch.no_grad():
        val_iou = []
        for image, target in loader:
            image, target = image.to(DEVICE), target.to(DEVICE)
            output = model(image)
            output = output.argmax(1)
            iou = get_iou(output, target)
            val_iou.append(iou)
    return val_iou


# repo: https://github.com/qubvel/segmentation_models.pytorch
# doc: https://smp.readthedocs.io/en/latest/

header = r'''Epoch |  Loss |  Score | Time(min)'''
raw_line = '{:8d}' + '\u2502{:8f}' * 2 + '\u2502{:8f}'
class_name = ['farm', 'land', 'forest', 'grass', 'road', 'urban_area','countryside', 'industrial_land', 'construction', 'water', 'bareland']

model = smp.UnetPlusPlus(encoder_name="efficientnet-b6", encoder_weights='imagenet', in_channels=3, classes=10)
model.train()
model.to(DEVICE)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(),lr=0.0005)

# loss 
# DiceLoss, JaccardLoss, SoftBCEWithLogitsLoss, SoftCrossEntropyLoss
DiceLoss_fn = DiceLoss(mode='multiclass')
SoftCrossEntropy_fn=SoftCrossEntropyLoss(smooth_factor=0.1)
loss_fn = L.JointLoss(first=DiceLoss_fn, second=SoftCrossEntropy_fn,first_weight=0.5, second_weight=0.5).to(DEVICE)

best_iou = 0

for epoch in (range(1, EPOCHES + 1)):   
    losses = []
    start_time = time.time()

    for image, target in tqdm(loader):
        image, target = image.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    viou = validation(model, vloader, loss_fn)

    print(header)
    print(raw_line.format(epoch, np.array(losses).mean(), np.mean(viou),
                            (time.time() - start_time) / 60 ** 1))
    print('\n')    
    print('  '.join(class_name))
    print('\t'.join(np.stack(viou).mean(0).round(3).astype(str)))
   
    if best_iou < np.stack(viou).mean(0).mean():
        best_iou = np.stack(viou).mean(0).mean()
        model_path = '../user_data/model_data/'
        torch.save(model.state_dict(), model_path + 'upp_efb6_{}.pth'.format(epoch))