import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Dense, Flatten, Activation, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU

from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from skimage import transform

import numpy as np
import random
import math
import cv2
import matplotlib.pyplot as plt
import os
import csv


#----All methods and functions-------------------------------------------------------------------------
#def random_rotate(image, angles = (-15,15)):
    
#    rand_angle = random.randint(angles[0],angles[1])
#    rot = transform.rotate(image,rand_angle,mode='edge')
#    return np.array(rot).reshape(160,320,3)
def random_brightness(img,factor_range = 0.4):
    
    alpha = random.uniform(factor_range,1+factor_range)
    img = cv2.cvtColor(img,cv2.COLOR_YUV2BGR)
   
    new_image = np.zeros(img.shape, img.dtype)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                new_image[y,x,c] = np.clip(alpha*img[y,x,c], 0, 255)
    
    return cv2.cvtColor(new_image,cv2.COLOR_BGR2YUV)

def random_shadow(img, w_low=0.4, w_high=0.6):
    
    height, width = (img.shape[0], img.shape[1])
    
    x1_top = np.random.random() * width/2 
    x2_top = np.random.uniform(x1_top,width)
    
    x1_bot = np.random.random() * width
    x2_bot = np.random.uniform(x1_top,width)
    
    poly = np.asarray([[ [x1_top,0], [x1_bot,height], [x2_bot,height], [x2_top,0]]], dtype=np.int32)
        
    mask_weight = np.random.uniform(w_low, w_high)
    origin_weight = 1 - mask_weight
    
    mask = np.copy(img).astype(np.int32)
    cv2.fillPoly(mask, poly, (0, 0, 0))
    #masked_image = cv2.bitwise_and(img, mask)
    
    return cv2.addWeighted(img.astype(np.int32), origin_weight, mask, mask_weight, 0).astype(np.uint8)

def random_translation(img,translation = (-10,10)):
    
    x = np.random.randint(translation[0],translation[1])
    y = np.random.randint(translation[0],translation[1])
    
    tform = transform.AffineTransform(scale = None,rotation = None, shear = None, translation = (x,y))
    new_image = transform.warp(img,tform,output_shape = (160,320),mode = 'edge')
    """
    correction = 0.05
    if y<0:
        correction *= 1
    else:
        correction *= -1
        
    correction_angle = angle + correction
    """
    return np.array(new_image).reshape(160,320,3)

def generate_data(samples,batch_size = 128, flip = True, translation = True):
    
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            
            for batch_sample in batch_samples:
                for i in range(3):
                    local_path = batch_sample[i]
                    image = cv2.imread(local_path)
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
                    images.append(image)
        
                correction = 0.2
                angle = float(batch_sample[3])
                
                angles.append(angle)
                angles.append(angle+correction)
                angles.append(angle-correction)
            
            aug_images = []
            aug_angles = []
            
            for image, angle in zip(images,angles):
                aug_images.append(image)
                aug_angles.append(angle)
                
                transformed_image = cv2.flip(image,1)
                transformed_angle = float(angle)*-1.0

                transformed_image = random_brightness(transformed_image)
                transformed_image = random_translation(transformed_image)
                transformed_image = random_shadow(transformed_image)
                
                aug_images.append(transformed_image)
                aug_angles.append(transformed_angle)
                
            X_train = np.array(aug_images)
            y_train = np.array(aug_angles)
            
            yield shuffle(X_train,y_train)
    
#----Pre-Processing Data----------------------------------------------------------------------------
info = []
path = "/home/workspace/data/"
destination = np.array([#"data/",
                        "own_data1/",
                        "own_data2/",
                        #"own_data3/",
                        #"own_data4/",
                        #"own_data5/"
                       ])
csv_file = "driving_log.csv"

for dest in destination:
    with open(path+dest+csv_file,"r") as file:
        reader = csv.reader(file)
        for line in reader:
            for i in range(3):
                line[i] = line[i].replace("\\","/").split("/")
                line[i] = path + dest + "IMG/" +line[i][-1]
            
            info.append(line)

train_samples,valid_samples = train_test_split(info,test_size = 0.2)
            
train_generator = generate_data(train_samples)
valid_generator = generate_data(valid_samples,flip = False,translation = False)

batch= 128

#----Deep-Learning Model----------------------------------------------------------------------------
model = Sequential()
model.add(Lambda(lambda x:x/255.0 -0.5,input_shape = (160,320,3)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))

model.add(Convolution2D(24,5,5, kernel_regularizer = l2(0.001), subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(36,5,5, kernel_regularizer = l2(0.001), subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(48,5,5, subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(Convolution2D(64,3,3, activation = 'relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer=Adam(lr = 0.0001),loss = 'mse')
#model.fit(X_train,y_train, epochs=4,validation_split = 0.2, shuffle = True)
model.fit_generator(train_generator,
                    steps_per_epoch = math.ceil(len(train_samples)/batch), 
                    validation_data = valid_generator, 
                    validation_steps = math.ceil(len(valid_samples)/batch), 
                    epochs = 5, verbose =1)
                   
model.save('model3.h5')



