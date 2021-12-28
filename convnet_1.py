# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 15:52:35 2021

@author: Kert PC
"""
import os
import numpy as np
import tensorflow as tf
from scipy import ndimage, signal


from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers
import tensorflow_addons as tfa
from scipy import ndimage
from tensorflow.keras.layers import Rescaling, Flatten, BatchNormalization, Conv2D, MaxPooling2D, DepthwiseConv2D, Dropout, UpSampling2D, concatenate
import pandas as pd
from focal_loss import SparseCategoricalFocalLoss
from skimage.color import rgb2hsv, hsv2rgb
import scipy

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Add
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, Reshape, Dense, LeakyReLU, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
import cv2

from tensorflow.keras.utils import image_dataset_from_directory

import PIL
from PIL import ImageOps, Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import math

from scipy import signal
from scipy.io import wavfile
import random
from aenum import Enum, MultiValue

class Genre(Enum) :
    _init_ = 'value fullname'
    _settings_ = MultiValue
    
    blues = 0, "blues"
    classical = 1, "classical"
    country = 2, "country"
    disco = 3, "disco"
    hiphop = 4, "hiphop"
    jazz = 5, "jazz"
    metal = 6, "metal"
    pop = 7, "pop"
    reggae = 8, "reggae"
    rock = 9, "rock"
    
    def __int__(self):
        return self.value

class SpectImages(keras.utils.Sequence):

    def __init__(self, images, batch_size=8, img_size=(360,480)):
        self.images = images
        
        self.batch_size = batch_size
        self.img_size = img_size
        

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        
        if i >= len(self.images) :
            i = i % len(self.images)
            print('Taking pictures from the beginning')
            
        
        batch_images = self.images[i : i + self.batch_size]
        
        return self.getData(batch_images)
    
    def __len__(self):
        ret = len(self.images) // self.batch_size
        
        return ret
    
    def getData(self, batch_images, aug_round=0) :
        
        x = []
        y = []
        
        for j, path in enumerate(batch_images):
            #img = np.array(load_img(path, color_mode="rgb"))
            
            sample_rate, samples = wavfile.read(path)
            frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

            m_class = path.split('/')[-1].split('\\')[0]
            class_vect = np.zeros((10, 10))
            class_vect[Genre(m_class).value][Genre(m_class).value] = 1
            
            
            spectrogram = np.expand_dims(spectrogram[:, 1000:1900], axis=-1)
            
            
            #print(np.shape(spectrogram))
            #print(type(spectrogram))
            
            
            """
            img = img - np.min(img)
            img = ((img / np.max(img)) * 2) - 1
            """
            #img = img / 255

            spectrogram = spectrogram - np.min(spectrogram)
            spectrogram = ((spectrogram / np.max(spectrogram)) * 2) - 1
             
            x.append(spectrogram.astype('float64'))
            y.append(class_vect)
            
        return np.array(x), np.array(y)
    


class Classifier:  
    def __init__(self) :
        self.input_dir = './dataset/Data/genres_original/'
        self.target_dir = './dataset/Data/genres_original/'
        self.model_dir = './models/convnet_1/'


    def segment(self, img) :
        result = self.model.predict(img)
        return result


    def segment_list(self, img_list) :
        val_images = sorted(
            [   os.path.join(self.input_dir, fname)
                for fname in img_list ] )
        
        val_masks = sorted(
            [   os.path.join(self.target_dir, fname)
                for fname in img_list ] )
        
        keras.backend.clear_session()
    
        val_gen = SpectImages(val_images, val_masks, batch_size=1)
        
        return self.model.predict(val_gen)

    
    def train_model(self, num_classes = 2, batch_size = 8, num_epochs = 50, block_number = 2, filter_number = 16) :
        
        train_images = sorted(
            [   os.path.join(self.input_dir, dname, fname)
                for dname in os.listdir(self.input_dir)
                for fname in os.listdir(self.input_dir + dname)
                if fname.endswith(".wav") ] )
        
        random.shuffle(train_images)
        
        val_images = train_images[-100:]
        train_images = train_images[:-100]

        
        img_height = 288
        img_width = 432
        """
        train_ds = image_dataset_from_directory(
              self.input_dir,
              validation_split=0.2,
              subset="training",
              seed=123,
              image_size=(img_height, img_width),
              batch_size=batch_size)

        val_ds = image_dataset_from_directory(
              self.input_dir,
              validation_split=0.2,
              subset="validation",
              seed=123,
              image_size=(img_height, img_width),
              batch_size=batch_size)
        """
        keras.backend.clear_session()
            
        train_gen = SpectImages(train_images, batch_size=batch_size)
        val_gen = SpectImages(val_images, batch_size=batch_size)
        
        
        #inputs, outputs, self.model = self.unet_model_blocks(block_number=block_number, filter_number=filter_number)
        #self.model = self.get_model()
        
        inputs, outputs, self.model = self.cnn_model(num_classes=10)
        
        self.model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=["sparse_categorical_accuracy"])
            
        self.model.summary()
        
        epochs = num_epochs
    
        callbacks = [
            keras.callbacks.ModelCheckpoint("ear_segmentation", save_best_only=True)
        ]
            
        history = self.model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
        #history = self.model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks)
        
        model_names = [ mod_name for mod_name in os.listdir(self.model_dir) ]
        
        model_type_num = str(len(model_names) + 1)
        model_path = self.model_dir + 'model' + '_' + model_type_num
        
        self.model.save(model_path)
        
        hist_df = pd.DataFrame(history.history) 
        
        hist_json_file = self.model_dir + 'history_' + model_type_num + '.json' 
        with open(hist_json_file, mode='w') as f:
            hist_df.to_json(f)
        

    def cnn_model(self, inputs=None, num_classes=10):
        if inputs is None:
            num_of_channels = 1
            
            inputs = layers.Input((129, 900) + (num_of_channels, ))
            
        x = inputs
        
        x = Rescaling(1./255)(x)

        conv1 = Conv2D(16, (3, 3), activation="relu", padding="same")(x)
        
        conv1 = Conv2D(24, (3, 3), activation="relu", padding="same")(conv1)
        pool1 = MaxPooling2D(pool_size=(3, 3))(conv1)
        pool1 = Dropout(0.25)(pool1)

        conv2 = Conv2D(32, (3, 3), activation="relu", padding="same")(pool1)
        pool2 = MaxPooling2D(pool_size=(3, 3))(conv2)
        pool2 = Dropout(0.25)(pool2)
        
        conv3 = Conv2D(48, (3, 3), activation="relu", padding="same")(pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        pool3 = Dropout(0.25)(pool3)
        
        
        flat = Flatten()(pool3)
        
        dense1 = Dense(1000, activation='relu')(flat)
        dense1 = Dropout(0.4)(dense1)
        
        dense2 = Dense(500, activation='relu')(dense1)
        dense2 = Dropout(0.35)(dense2)
        
        dense3 = Dense(125, activation='relu')(dense2)
        dense3 = Dropout(0.3)(dense3)
        
        dense4 = Dense(60, activation='relu')(dense3)
        dense4 = Dropout(0.25)(dense4)


        dense5 = Dense(num_classes, activation='softmax')(dense2)
        
        model = keras.Model(inputs, dense5)

        return inputs, dense3, model

            

if __name__ == '__main__':
    classifier = Classifier()
    classifier.train_model(batch_size=3, num_epochs=60)

    print('blah')