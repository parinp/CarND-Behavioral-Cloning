import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Dense, Flatten, Activation, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
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
import numpy.random as random
import pickle

import matplotlib.pyplot as plt
import pylab

load_data = True
batch = 128
def random_brightness(image):

    rand = random.uniform(0.3,1.0)
    new_image = rand*image
    
    return new_image

if(load_data == False):
    info = []

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
        
    # from Udacity class material, reads in X
    images = []
    angles = []

    for lines in info:
        for i in range(3):
            local_path = lines[i]
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
        angle = float(lines[3])
        
        angles.append(angle)
        angles.append(angle+correction)
        angles.append(angle-correction)

    def balance_data(X, y, bins):

        new_y = []
        new_X = []

        bal = np.linspace(-1,1,bins)
        hist = plt.hist(y,bins = bal)
        total = sum(hist[0])
        dist = hist[0]
        non_z = dist[dist!=0]

        size = ((total*0.8) / len(non_z))
        sample_size = np.repeat(size, len(dist)).astype(np.int)
        
        for i in range(0,len(bal)-1):
            y_inrange = [y for X, y in zip(X, y) if y >= bal[i] and y <= bal[i + 1]]
            x_inrange = [X for X, y in zip(X, y) if y >= bal[i] and y <= bal[i + 1]]

            if len(y_inrange)!=0:
                rand = random.choice(range(len(y_inrange)), sample_size[i], replace = True)
                new_y += [y_inrange[j] for j in rand]
                new_X += [x_inrange[j] for j in rand]

        return new_X, new_y

    # run the balance
    
    X_sample, y_sample = balance_data(images,angles,25)

    with open('D:/data/sample1.p', 'wb') as f:
            pickle.dump([X_sample, y_sample], f)


    """
    bal = np.linspace(-1, 1, 50)

    plt.hist(y_sample, bins = bal) 
    plt.show()

    """
    # end vis

else:
    with open('D:/data/sample1.p', 'rb') as f:
        X_sample, y_sample = pickle.load(f)
    

X_train = np.array(X_sample)
y_train = np.array(y_sample)

X_valid, X_train, y_valid, y_train = train_test_split(X_train, y_train, test_size=0.8, shuffle = True)

#  Generator
datagen = ImageDataGenerator(
    rotation_range = 5,
    width_shift_range=0,
    channel_shift_range = 20,
    shear_range = 0, 
    preprocessing_function = random_brightness,
    fill_mode='nearest')

# Model
model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape = (66,200,3)))
model.add(Conv2D(24, (5, 5), strides = (2,2), activation = "elu", kernel_regularizer = l2(.0001)))
model.add(Conv2D(36, (5, 5), strides = (2,2), activation = "elu", kernel_regularizer = l2(.0001)))
model.add(Conv2D(48, (3, 3), activation = "elu", kernel_regularizer = l2(.0001)))
model.add(Conv2D(64, (3, 3), activation = "elu", kernel_regularizer = l2(.0001)))
model.add(Conv2D(64, (3, 3), activation = "elu", kernel_regularizer = l2(.0001)))
model.add(Flatten())
model.add(Dense(100, kernel_regularizer = l2(.0001), activation = 'elu'))
model.add(Dense(25,  kernel_regularizer = l2(.0001), activation = 'elu'))
model.add(Dense(10, kernel_regularizer = l2(.0001), activation = "elu"))
model.add(Dense(1))

model.compile(optimizer = Adam(lr = 0.0001), loss='mse')
model.fit_generator(datagen.flow(X_train,y_train, batch_size = batch, shuffle = True),
                    steps_per_epoch = 2000, 
                    validation_data = (X_valid,y_valid), 
                    validation_steps = 800, 
                    epochs = 30, verbose =1,
                    callbacks = [EarlyStopping(patience = 2)])

model.save('model.h5')
