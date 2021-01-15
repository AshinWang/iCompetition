import tensorflow as tf
from keras import backend as K
SM_FRAMEWORK=tf.keras
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



def Data2Npy(PATH, MASK_PATH):
    data_df = pd.DataFrame()
    X = []
    y = []
    for i in (list({i.split('.')[0] for i in os.listdir(PATH)})):
        i = i.split('.')[0]
        X.append(PATH + '{}.jpg'.format(i))
        y.append(MASK_PATH + '{}.png'.format(i))
        
    data_df['X']  = X
    data_df['y']  =  y 

    if data_df['X'].values[0].split('.')[0] == '':
        data_df = data_df.drop(index=0)

    X = []
    for filename in tqdm(data_df['X'].values, position=0, leave=True):
        temp = resize(imread(filename),(512,512,3)) # resize img(512*512*3)
        X.append(temp)
    gc.collect()
    X = np.array(X)

    y = []
    for filename in tqdm(data_df['y'].values, position=0, leave=True):
        temp = resize(imread(filename),(512,512,1)) # resize mask(512*512*1)
        y.append(temp)
    gc.collect()
    y = np.array(y)
    return X, y

# img2np
TRAIN_PATH = '../s2_data/data/train/'
TRAIN_MASK_PATH = '../s2_data/data/train_mask/'

X, y = Data2Npy(TRAIN_PATH, TRAIN_MASK_PATH)

X_train, y_train, X_valid, y_valid = train_test_split(X, y, test_size = .1, random_state=2020)
print('X_train: {}'.format(X_train.shape), 'y_train: {}'.format(y_train.shape))
print('X_valid: {}'.format(X_valid.shape), 'y_valid: {}'.format(y_valid.shape))


# Dice_Coeff or F1 score
def metric(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)


#  segmentation model
# repo: https://github.com/qubvel/segmentation_models

# | Backbones    | Names                                                                                                                                 |
# |--------------|---------------------------------------------------------------------------------------------------------------------------------------|
# | VGG          | 'vgg16' 'vgg19'                                                                                                                       |
# | ResNet       | 'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152'                                                                              |
# | SE-ResNet    | 'seresnet18' 'seresnet34' 'seresnet50' 'seresnet101' 'seresnet152'                                                                    |
# | ResNeXt      | 'resnext50' 'resnext101'                                                                                                              |
# | SE-ResNeXt   | 'seresnext50' 'seresnext101'                                                                                                          |
# | SENet154     | 'senet154'                                                                                                                            |
# | DenseNet     | 'densenet121' 'densenet169' 'densenet201'                                                                                             |
# | Inception    | 'inceptionv3' 'inceptionresnetv2'                                                                                                     |
# | MobileNet    | 'mobilenet' 'mobilenetv2'                                                                                                             |
# | EfficientNet | 'efficientnetb0' 'efficientnetb1' 'efficientnetb2' 'efficientnetb3' 'efficientnetb4' 'efficientnetb5' efficientnetb6' efficientnetb7' |


# Unet + efficientnetb3
model_name = 'Unet'
backbone_name = 'efficientnetb3'
model = Unet(backbone_name, classes=1, activation='sigmoid')
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=metric)


CKPT_PATH = "../user_data/model/"

t= time.strftime("%Y%m%d-%H%M%S", time.localtime())
### MODELCHECKPOINT CALLBACK
save = ModelCheckpoint(CKPT_PATH+'{}_{}_{}_model.h5'.format(t,model_name,backbone_name), verbose=1, save_best_only=True, save_weights_only=True)

### REDUCES LR WHEN METRTIC IS NOT IMPROVING
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_delta =1e-4, min_lr=0.00001, mode='min')

### LEARNING RATE SCHEDULER
def scheduler(epoch):
  if epoch < 10:
     return 0.001
  else:
     return float(0.001 * tf.math.exp(0.1 * (10 - epoch)))
lr_schedule = LearningRateScheduler(scheduler) 

### EARLY STOPPING
early_stopping = EarlyStopping(patience=10, verbose=1)

# Data Generator and augmentation
data_gen_args = dict(horizontal_flip=True,vertical_flip=True)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 42
bs = 32 # BacthSize
nb_epochs = 100 # epoch

image_generator = image_datagen.flow(X_train, seed=seed, batch_size=bs, shuffle=True)
mask_generator = mask_datagen.flow(y_train, seed=seed, batch_size=bs, shuffle=True)

# Just zip the two generators to get a generator that provides augmented images and masks at the same time
train_generator = zip(image_generator, mask_generator)

results = model.fit_generator(train_generator, steps_per_epoch=(spe), epochs=nb_epochs,validation_data=(X_valid, y_valid),callbacks=[save,lr_schedule,reduce_lr])

# save final model
model.save_weights(CKPT_PATH+'{}_{}_{}_model.h5'.format(t,model_name,backbone_name))

# predicte valid data
predicted = model.predict(X_valid)