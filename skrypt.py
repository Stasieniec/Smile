# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 08:58:56 2020

@author: Stanisław Wasilewski
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import  RMSprop, Adam
from tensorflow.keras.layers import  Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import  matplotlib.pyplot as plt
import cv2
import os
import numpy as np

sciezka = "PATH"

#%%

train = ImageDataGenerator(rescale  =  1/255)

train_dataset  = train.flow_from_directory((sciezka + '/train/'), 
                                           target_size  = (224, 224),
                                           batch_size = 16,
                                           class_mode = 'binary')

validation_dataset  = train.flow_from_directory((sciezka + '/test/'), 
                                           target_size  = (224, 224),
                                           batch_size = 16,
                                           class_mode = 'binary')

#%%
train_dataset.class_indices
#%%
model  = Sequential([Conv2D(32, (3,3), activation = 'relu', input_shape = (224, 224, 3)),
                     MaxPool2D(2,2),
                     
                     Conv2D(32,(3,3), activation = 'relu'),
                     MaxPool2D(2,2),
                     Conv2D(64,(3,3), activation = 'relu'),
                     MaxPool2D(2,2),
                     
                     Flatten(),
                     
                     Dense(64, activation  = 'relu'),
                     Dropout(0.5),
                     Dense(64, activation  = 'relu'),
                     Dropout(0.5),
                     
                     Dense(1, activation  = 'sigmoid')
    ]
    )


model.compile(loss = 'binary_crossentropy',
              optimizer  = 'rmsprop',
              metrics = ['accuracy'])

model_fit = model.fit(train_dataset,
                      epochs = 30 ,
                      steps_per_epoch = 10,
                      validation_data  = validation_dataset)


#%%

dir_path = (sciezka + 'TEST YES')

for  i in  os.listdir(dir_path):
    img = image.load_img(dir_path + '/' +  i, target_size = (224,224))
    plt.imshow(img)
    plt.show()
        
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis =0)
    images = np.vstack([X])
    val =  model.predict(images)
    if  val  == 0:
        print('not a smile')
    else:
        print('smile')
        
        
#%%
img = image.load_img('TEST PHOTO', target_size = (224,224))
plt.imshow(img)
plt.show()  
X = image.img_to_array(img)
X = np.expand_dims(X, axis =0)
images = np.vstack([X])
val =  model.predict(images)
if val  == 0:
    print('not a smile')
else:
    print('smile')
    
