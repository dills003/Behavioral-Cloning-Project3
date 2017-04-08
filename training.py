#Udacity Behavioral Cloning - Project #3
#This is my method to train on data collected by driving around
#the test course to create a self-driving car on course #1

import csv
import cv2
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split


#array to hold the csv excel files created in 'Training' mode
lines = [] 

#open the data that was collected and store in lines
with open("C:\\Users\\dills\\Desktop\\Data\\driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
#used for the data generator, split into test and validation
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
train_samplesLen = len(train_samples) * 3 * 2 #i augmented the data by using three cameras and a flip

print(train_samplesLen) #sanity check/debug

#data generator idea presented in lecture
#i augmented the data by using three cameras and flipping
def generator(samples, batch_size=100):
    num_samples = len(samples)
    while 1:
        for offset in range(0, num_samples, batch_size): #loop through the samples
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            steeringCorrection = .625 #for camera correction, fake like center of road

            for line in batch_samples: #picks out all three images/steering angle at a given time
                for i in range(3): 
                    source_path = line[i]
                    image = cv2.imread(source_path)
                    images.append(image)
                    measurement = float(line[3])
                    if i == 0:
                        measurements.append(measurement) #front camera
                    elif i == 1:
                        measurements.append(measurement + steeringCorrection) #left camera
                    elif i == 2:
                        measurements.append(measurement - steeringCorrection) #right camera


            aug_images = []
            aug_measurements = []
            for image, measurement in zip(images, measurements): #more data by flipping
                aug_images.append(image)
                aug_measurements.append(measurement)
                aug_images.append(cv2.flip(image, 1))
                aug_measurements.append(measurement*(-1.0))

            X_train = np.array(aug_images) #x training data
            y_train = np.array(aug_measurements) #y training data
            yield sklearn.utils.shuffle(X_train, y_train) #sweet new way to return

#for the keras fit model
train_generator = generator(train_samples, batch_size=100)
validation_generator = generator(validation_samples, batch_size=100)

#my model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


model = Sequential()
model.add(Cropping2D(cropping=((80,25),(0,0)),input_shape=(160,320,3))) #crop the image first
model.add(Lambda(lambda x: x / (255.0 - 0.5))) #normalize the data
model.add(Convolution2D(12,5,5,subsample=(2,2),activation="relu")) #convolution with 12 layers, 2x2 stride, 5x5 pattern
model.add(MaxPooling2D()) #pool the data, canned it 2x2
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu")) #convolution again 36, 2x2, and 5x5
model.add(MaxPooling2D()) #pool again
model.add(Flatten()) #flatten into 1 dimension
model.add(Dense(200,activation="sigmoid")) #200 on output, sigmoid(didn't seem to help much) but...
model.add(Dense(100))#100, none is linear activation
model.add(Dense(50))#100
model.add(Dense(1))#1
model.compile(loss='mse', optimizer='adam') #mean squared error
model.fit_generator(train_generator,samples_per_epoch=train_samplesLen,validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=10) #does everything

model.save('model.h5')


