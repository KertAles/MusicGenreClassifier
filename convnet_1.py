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


from loadData import load_data, get_genres

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

class SpectGen(keras.utils.Sequence):

    def __init__(self, images, labels, batch_size=8, img_size=(129, 600)):
        self.images = images
        self.labels = labels
        
        self.batch_size = batch_size
        self.img_size = img_size
        

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        
        if i >= len(self.images) :
            i = i % len(self.images)
            print('Taking pictures from the beginning')
            
        
        batch_images = self.images[i : i + self.batch_size]
        batch_labels = self.labels[i : i + self.batch_size]
        
        return self.getData(batch_images, batch_labels)
    
    def __len__(self):
        ret = len(self.images) // self.batch_size
        
        return ret
    
    def getData(self, batch_images, batch_labels, aug_round=0) :
        
        x = []
        y = []
        
        for j, img in enumerate(batch_images):

            x.append(img)
            
        for j, lab in enumerate(batch_labels):
            y.append(lab)
            
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
    
        val_gen = SpectGen(val_images, val_masks, batch_size=1)
        
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
            
        #train_gen = SpectImages(train_images, batch_size=batch_size)
        #val_gen = SpectImages(val_images, batch_size=batch_size)
        
        
        #inputs, outputs, self.model = self.unet_model_blocks(block_number=block_number, filter_number=filter_number)
        #self.model = self.get_model()
        
        inputs, outputs, self.model = self.cnn_model(num_classes=10)
        
        self.model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=["sparse_categorical_accuracy"])
            
        self.model.summary()
        
        epochs = num_epochs
    
        callbacks = [
            keras.callbacks.ModelCheckpoint("ear_segmentation", save_best_only=True)
        ]
            
        train_gen = None
        val_gen = None
        
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
        

    def cnn_model(self, inputs=None, num_classes=10, num_of_channels=1):
        if inputs is None:
            
            inputs = layers.Input((128, 512) + (num_of_channels, ))
            
        x = inputs
        
        #x = Rescaling(1./255)(x)

        conv1 = Conv2D(16, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.2), padding="same")(x)
        conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv1 = Conv2D(20, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.2), padding="same")(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        pool1 = Dropout(0.05)(pool1)

        conv2 = Conv2D(24, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.2), padding="same")(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        pool2 = BatchNormalization()(pool2)
        
        conv3 = Conv2D(28, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.2), padding="same")(pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        pool3 = Dropout(0.05)(pool3)
        
        conv4 = Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.2), padding="same")(pool3)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        pool4 = Dropout(0.05)(pool4)
        
        
        flat = Flatten()(pool4)
        
        dense1 = Dense(40, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(flat)
        dense1 = Dropout(0.1)(dense1)
        
        dense2 = Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(dense1)
        
        dense3 = Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(dense2)

        dense4 = Dense(num_classes, activation='softmax')(dense3)
        
        model = keras.Model(inputs, dense4)

        return inputs, dense4, model



from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from evaluation import evaluate_predictions


def train_eval_conv(data, labels, data_test, labels_test, val_split=0.8, num_of_channels=1) :
    classifier = Classifier()
    
    genres = set(list(labels))
    inputs, outputs, model = classifier.cnn_model(num_classes=len(genres), num_of_channels=num_of_channels)
        
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                
    model.summary()
            
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_Y = encoder.transform(labels)
    
    #dummy_y = np_utils.to_categorical(encoded_Y)
    
    #val_split = 0.75
    data_len = len(data)
    n_split = round(data_len * val_split) + 1
    epochs = 50
    batch_size = 4   
    
    callbacks = [
         keras.callbacks.ModelCheckpoint("conv_classification", save_best_only=True)
    ]
    
    tr_dat = []
    tr_lab = []
    va_dat = []
    va_lab = []
    
    for i in range(len(data)) : 
        if i < n_split-1 :
            tr_dat.append(tf.convert_to_tensor(data[i]))
            tr_lab.append(tf.convert_to_tensor(encoded_Y[i]))
        else :
            va_dat.append(tf.convert_to_tensor(data[i]))
            va_lab.append(tf.convert_to_tensor(encoded_Y[i]))
            
    train_gen = SpectGen(tr_dat, tr_lab, batch_size=batch_size)
    val_gen = SpectGen(va_dat, va_lab, batch_size=batch_size)
    
    tf.config.run_functions_eagerly(True)
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, batch_size=batch_size)
    
    model_dir = './models/convnet1/'
    
    model_names = [ mod_name for mod_name in os.listdir(model_dir) ]
            
    model_type_num = str(len(model_names) + 1)
    #model_path = model_dir + 'model_' + feat + '_' + gen + '_' + prep + '_' + model_type_num
    model_path = model_dir + 'model_' + model_type_num
     
    model.save(model_path)
            
    hist_df = pd.DataFrame(history.history) 
            
    #hist_json_file = model_dir + 'history_' + feat + '_' + gen + '_' + prep + '_' + model_type_num + '.json'
    hist_json_file = model_dir + 'history_' + model_type_num + '.json'
    
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)
    
    test_gen = SpectGen(data_test, labels_test, batch_size=1)
    
    predictions_raw = model.predict(test_gen)
    predictions_raw = np.argmax(predictions_raw, axis=1)
    
    predictions = encoder.inverse_transform(predictions_raw)
    
    return evaluate_predictions(predictions, labels_test)


def load_eval_conv(data_test, labels_test, model_name) :
    model_dir = './models/convnet1/'
    batch_size = 4
            
    model = keras.models.load_model(model_dir + model_name)
    
    
    encoder = LabelEncoder()
    encoder.fit(labels_test)
    
    test_gen = SpectGen(data_test, labels_test, batch_size=1)
    
    predictions_raw = model.predict(test_gen)
    predictions_raw = np.argmax(predictions_raw, axis=1)
    
    predictions = encoder.inverse_transform(predictions_raw)
    
    return evaluate_predictions(predictions, labels_test)


if __name__ == '__main__':
    feat = 'mc'
    gen = 'ps'
    prep = 'mfccspectconv'
    
    if gen == 'all' :
        genres = get_genres('all')
    if gen == 'ps' :
        genres = get_genres('paper_split')
    if gen == 'cs1' :
        genres = get_genres('custom_split')
     
    
    data, labels, data_test, labels_test = load_data(split=0.8, shuffle=True, prep=prep, take_middle=True, genres=genres)
    
    evaluation = train_eval_conv(data, labels, data_test, labels_test)


    
"""  
feat = 'mc'
gen = 'ps'
prep = 'mfccspectconv'

if gen == 'all' :
    genres = get_genres('all')
if gen == 'ps' :
    genres = get_genres('paper_split')
if gen == 'cs1' :
    genres = get_genres('custom_split')
 
#features = get_features(feat)

data_train, label_train, data_test, label_test = load_data(split=0.8, shuffle=True, prep=prep, take_middle=True, genres=genres)


classifier = Classifier()

inputs, outputs, model = classifier.cnn_model(num_classes=len(genres))
        
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
model.summary()
        
encoder = LabelEncoder()
encoder.fit(label_train)
encoded_Y = encoder.transform(label_train)

#dummy_y = np_utils.to_categorical(encoded_Y)

val_split = 0.75
data_len = len(data_train)
n_split = round(data_len * val_split) + 1
epochs = 50
batch_size = 3   

callbacks = [
     keras.callbacks.ModelCheckpoint("ear_segmentation", save_best_only=True)
]

tr_dat = []
tr_lab = []
va_dat = []
va_lab = []

for i in range(len(data_train)) : 
    if i < n_split-1 :
        tr_dat.append(tf.convert_to_tensor(data_train[i]))
        tr_lab.append(tf.convert_to_tensor(encoded_Y[i]))
    else :
        va_dat.append(tf.convert_to_tensor(data_train[i]))
        va_lab.append(tf.convert_to_tensor(encoded_Y[i]))
        
train_gen = SpectGen(tr_dat, tr_lab, batch_size=batch_size)
val_gen = SpectGen(va_dat, va_lab, batch_size=batch_size)

tf.config.run_functions_eagerly(True)
history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, batch_size=batch_size)

model_dir = './models/convnet1/'

model_names = [ mod_name for mod_name in os.listdir(model_dir) ]
        
model_type_num = str(len(model_names) + 1)
model_path = model_dir + 'model_' + feat + '_' + gen + '_' + prep + '_' + model_type_num
        
model.save(model_path)
        
hist_df = pd.DataFrame(history.history) 
        
hist_json_file = model_dir + 'history_' + feat + '_' + gen + '_' + prep + '_' + model_type_num + '.json'
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

test_gen = SpectGen(data_test, label_test, batch_size=batch_size)

predictions_raw = model.predict(test_gen)
predictions_raw = np.argmax(predictions_raw, axis=1)

predictions = encoder.inverse_transform(predictions_raw)

overall, classwise, conf, n_classes = evaluate_predictions(predictions, label_test)

print(overall)
print(classwise)
print(conf)
print(n_classes)

"""
