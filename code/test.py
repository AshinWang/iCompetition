import tensorflow as tf
from keras import backend as K
import segmentation_models
SM_FRAMEWORK = tf.keras
from segmentation_models import Unet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from PIL import Image, ImageChops, ImageEnhance
import os
import itertools
import shutil
from imageio import imread
import imageio
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import datetime
from keras.callbacks import TensorBoard
import cv2
from skimage.transform import resize
import PIL
from keras.optimizers import Adam,SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Input, Add, Dropout,Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, concatenate,Conv2DTranspose,GlobalMaxPool2D,GlobalAveragePooling2D,UpSampling2D
import gc
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
import time
from sklearn.utils import shuffle
from keras.models import load_model


def Test2Npy(TEST_PATH):
    test_df = pd.DataFrame()
    X = []
    for i in (list({i.split('.')[0] for i in os.listdir(TEST_PATH)})):
        i = i.split('.')[0]
        X.append(TEST_PATH + '{}.jpg'.format(i))
    test_df['X']  =  X

    if test_df['X'].values[0].split('.')[0] == '':
        test_df = test_df.drop(index=0)

    X_test = []
    for filename in tqdm(test_df['X'].values, position=0, leave=True):
        temp = Image.open(filename)
        w,h = temp.size
        temp = resize(imread(filename),(512, 512, 3))
        temp = np.array(temp)
        X_test.append(temp)
    gc.collect()
    X_test = np.array(X_test)
    return X_test

TEST_PATH = '../s2_data/data/test/'
X_test = Test2Npy(TEST_PATH)



backbone_name = 'efficientnetb3'
weight = '20201122-170215_Unet_efficientnetb3_model.h5'
model = Unet(backbone_name, classes=1, activation='sigmoid')

model_path ='../user_data/model/'  + weight
model.load_weights(model_path)

# model summary
print(model.summary(line_length=120))



TEST_MASK_PATH = '../prediction_result/images/'
predicted_test = model.predict(X_test)




# Save test mask
print('Get img name and path')
each_test_name = []
each_test_path = []

for i in tqdm(list({i.split('.')[0] for i in os.listdir(TEST_PATH)})):
    i = i.split('.')[0]
    each_test_path.append(TEST_PATH + '{}.jpg'.format(i))
    each_test_name.append(i) 

each_test_path.remove(each_test_path[0])
each_test_name.remove(each_test_name[0])
print(each_test_path[0])
print(each_test_name[0])

# Resize mask
print('Get img height and width')
h = []
w = []
for filename in tqdm(each_test_path):
    temp = Image.open(filename)
    h.append(temp.height)
    w.append(temp.width)

print('Resize and save img')
for index in tqdm(range(0,1500)):
    pred = np.squeeze(predicted_test[index])
    plt.imsave('pred_mask.png',pred)
    im_gray = cv2.imread('pred_mask.png', cv2.IMREAD_GRAYSCALE)
    (thresh, im_bw) = cv2.threshold(im_gray, 220, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    im_bw_t = Image.fromarray(im_bw) 
    im_bw_t.save(TEST_MASK_PATH+'{}.png'.format(each_test_name[index]))
        
    im_bw_t_n = Image.open(TEST_MASK_PATH+'{}.png'.format(each_test_name[index]))
    im_bw_t_nn = im_bw_t_n.resize(( w[index], h[index]),Image.ANTIALIAS)
    im_bw_t_nn.save(TEST_MASK_PATH+'{}.png'.format(each_test_name[index]))
print('DONE! Save mask img！')
