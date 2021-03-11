import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import time
from io import BytesIO
import base64
from PIL import Image
from tqdm import tqdm
import glob
import os
from scipy.io import loadmat
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from torch.cuda.amp import autocast
import ttach as tta
from tqdm import tqdm


Image.MAX_IMAGE_PIXELS = 1000000000000000
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def get_infer_transform():
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return transform

def inference(model, img_dir):
    transform=get_infer_transform()
    image = cv2.imread(img_dir, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = transform(image=image)['image']
    img=img.unsqueeze(0)
 
    with torch.no_grad():
        img=img.cuda()
        model = model
        output = model(img)
    
    pred = output.squeeze().cpu().data.numpy()
    pred = np.argmax(pred,axis=0)
    return pred+1

model_name = 'efficientnet-b6'
n_class=10
model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b6", 
    encoder_weights='imagenet', 
    in_channels=3,  
    classes=10)

model.to(DEVICE)
model.load_state_dict(torch.load("../user_data/model_data/upp_efb6_aug_20_3821.pth"))
model.eval()
tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')

assert_list=[1,2,3,4,5,6,7,8,9,10]

out_dir='../prediction_result/'
if not os.path.exists(out_dir):os.makedirs(out_dir)
test_paths=glob.glob('../tcdata/suichang_round2_test_partB_210316/*')
for per_path in tqdm_notebook(test_paths):
    result=inference(tta_model, per_path)
    img=Image.fromarray(np.uint8(result))
    img=img.convert('L')
    out_path=os.path.join(out_dir,per_path.split('/')[-1][:-4]+'.png')
    img.save(out_path)