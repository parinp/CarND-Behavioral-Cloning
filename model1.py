import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Dense, Flatten, Activation, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU

import keras
from keras.layers import GlobalAveragePooling2D , Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from skimage import transform
from skimage import exposure

import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import os
import csv
import random
import pickle

lines = []
images = []
load_data = False

def balance_data(X, y, bins):

    new_y = []
    new_X = []

    bal = np.linspace(-1,1,bins)
    hist = plt.hist(y,bins = bal)
    total = sum(hist[0])
    dist = hist[0]
    mean = dist[dist!=0]

    size = ((total*0.1) / len(mean))
    sample_size = np.repeat(size, len(dist)).astype(np.int)
    
    for i in range(0,len(bal)-1):
        y_inrange = [y for X, y in zip(X, y) if y >= bal[i] and y <= bal[i + 1]]
        x_inrange = [X for X, y in zip(X, y) if y >= bal[i] and y <= bal[i + 1]]

        if len(y_inrange)!=0:
            rand = random.choice(range(len(y_inrange)), sample_size[i], replace = True)
            new_y.append([y_inrange[j] for j in rand])
            new_X.append([x_inrange[j] for j in rand])

    return new_X, new_y

def generate_data(samples, batch_size = 64,Augment=True):
    
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
                    image = image[65:140,:,:]
                    image = cv2.resize(image,(200, 66))
                    
                    """
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
                    image, u, v = cv2.split(image)
                    image = norm_exposure(image)
                    """
                    images.append(image)
        
                correction = 0.20
                angle = float(batch_sample[3])
                
                angles.append(angle)
                angles.append(angle+correction)
                angles.append(angle-correction)
            
            aug_images = []
            aug_angles = []
            
            
            
            
            for image, angle in zip(images,angles):
                aug_images.append(image)
                aug_angles.append(angle)
                
                if Augment == True:
                    
                    transformed_image = cv2.flip(image,1)
                    transformed_angle = float(angle)*-1.0
                    
                    #aug_images.append(transformed_image)
                    #aug_angles.append(transformed_angle)
                    
                    """
                    rand = bool(random.getrandbits(1))
                    
                    if(rand==True):
                        transformed_image, side = random_translation(image)

                        if(side>0):
                            angle += angle*0.15/10
                        else:
                            angle += angle*-0.15/10
                    """
                    
                    aug_images.append(transformed_image)    
                    aug_angles.append(transformed_angle)
                    
                    #shadowed_image = random_shadow(image)
                    #aug_images.append(shadowed_image)
                    #aug_angles.append(angle)
                    
                    #contrasted_image = random_contrast(image)
                    #aug_images.append(contrasted_image)
                    #aug_angles.append(angle)
            
            X_train = aug_images
            y_train = aug_angles
            
            if Augment == True:
                m = np.mean(aug_angles)
                s = np.std(aug_angles)
                values, counts = np.unique(aug_angles,return_counts = True)
                
                max_num = max(counts)
                
                for val_i in range(len(values)//2):
                    num = max_num-counts[val_i]
                    if num>0:
                        value = values[val_i]
        
                        if value<(m-2*s):
                            indx = np.where(aug_angles==value)[0]
                            for i in range(num):
                                ind = np.random.choice(indx)
                                y_train.append(aug_angles[ind])
                                X_train.append(aug_images[ind])
                        elif value>(m+2*s):
                            indx = np.where(aug_angles==value)[0]
                            for i in range(num):
                                ind = np.random.choice(indx)
                                y_train.append(aug_angles[ind])
                                X_train.append(aug_images[ind])
                        """
                        elif value>=(m-2*s) and value<=(m-s):
                            indx = np.where(aug_angles==value)[0]
                            for i in range(num):
                                ind = np.random.choice(indx)
                                y_train.append(aug_angles[ind])
                                X_train.append(aug_images[ind])
                        elif value<=(m+2*s) and value > (m+s):
                            indx = np.where(aug_angles==value)[0]
                            for i in range(num):
                                ind = np.random.choice(indx)
                                y_train.append(aug_angles[ind])
                                X_train.append(aug_images[ind])
                        """
            yield shuffle(np.array(X_train),np.array(y_train))

batch = 128
if load_data == False:
    info = []

    """
    path = "../Data/"
    destination = np.array([#"Udacity_data/",
                        "track1_1/",
                        "track1_2/",
                        "track1_3/",
                        "track1_4/",
                        "track1_5/",
                        #"Track1_a/",
                        #"track2_1/",
                        "track1_cor1/",
                        ])
    """
    path = "../Data2/"
    destination = np.array(["Udacity_data/",
                            "track1_l1/",
                            "track1_l2/",
                            "track1_l3/",
                            #"track1_l4/",
                            #"track1_l5/"
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

    with open('train_sample.p', 'wb') as f:
        pickle.dump(train_samples, f)
    
    with open('valid_sample.p','wb') as f:
        pickle.dump(valid_samples, f)
else:
    with open('train_sample.p','rb') as f:
        train_samples = pickle.load(f)

    with open('valid_sample.p','rb') as f:
        valid_samples = pickle.load(f)

train_gen = generate_data(train_samples,batch,Augment = True)
valid_gen = generate_data(valid_samples,batch,Augment = False)

#----Deep-Learning Model
model = Sequential()
model.add(Lambda(lambda x:x/255.0 -0.5,input_shape = (66,200,3)))
#model.add(Cropping2D(cropping = ((70,25),(0,0))))
model.add(Convolution2D(6,(5,5),activation = 'elu', kernel_regularizer = l2(0.001)))
model.add(MaxPooling2D())
model.add(Convolution2D(16,(5,5),activation = 'elu', kernel_regularizer = l2(0.001)))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120, kernel_regularizer = l2(0.001)))
model.add(Dense(84, kernel_regularizer = l2(0.001)))
model.add(Dense(1, kernel_regularizer = l2(0.001)))

model.compile(optimizer=Adam(lr = 1e-4),loss = 'mse')

model.fit_generator(train_gen,
                    steps_per_epoch = 5000, 
                    epochs=30,
                    validation_data = valid_gen, 
                    validation_steps = 800,
                    verbose=1,
                    callbacks = [EarlyStopping(patience = 2)])

model.save('model10.h5')