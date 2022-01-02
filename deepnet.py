# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 15:37:48 2021

@author: Kert PC
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dropout, concatenate, Dense
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import numpy as np
import os
import pandas as pd
import tensorflow as tf

"""
model = Sequential()
model.add(Dense(500, input_dim=57, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(data_train, label_train, validation_split=0.1, epochs=100, batch_size=20)
"""

from loadData import load_data, get_genres, get_features
from evaluation import evaluate_predictions

def baseline_model(n_classes, input_dim=57):
    model = Sequential()
        
    model.add(Dense(40, input_dim=input_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.1))
        
    model.add(Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=0.2)))
    model.add(BatchNormalization())
        
    model.add(Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.05))
        
    model.add(Dense(n_classes, activation='softmax'))
        
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    return model

def train_eval_deep(data, labels, data_test, labels_test, val_split=0.8) :
    genres = set(list(labels))
    
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_Y = encoder.transform(labels)
    
    dummy_y = np_utils.to_categorical(encoded_Y)
    
    model = baseline_model(len(genres), data.shape[1])
    
    data_len = data.shape[0]
    n_split = round(data_len * val_split)
    
    n_epochs = 150
    batch_size = 50

    
    history = model.fit(data[:n_split], dummy_y[:n_split], verbose=0, validation_data=(data[n_split:], dummy_y[n_split:]), epochs=n_epochs, batch_size=batch_size)
    
    model_dir = './models/deepnet/'
    
    model_names = [ mod_name for mod_name in os.listdir(model_dir) ]
            
    model_type_num = str(len(model_names) + 1)
    model_path = model_dir + 'model_' + model_type_num
            
    model.save(model_path)
            
    hist_df = pd.DataFrame(history.history) 
            
    hist_json_file = model_dir + 'history_' + model_type_num + '.json' 
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)
    
    
    predictions_raw = model.predict(data_test)
    predictions_raw = np.argmax(predictions_raw, axis=1)
    
    predictions = encoder.inverse_transform(predictions_raw)
    
    return evaluate_predictions(predictions, labels_test)



def load_eval_deep(data_test, labels_test, model_name) :
    model_dir = './models/deepnet/'
    batch_size = 40
            
    model = keras.models.load_model(model_dir + model_name)
    
    encoder = LabelEncoder()
    encoder.fit(labels_test)
    
    predictions_raw = model.predict(data_test)
    predictions_raw = np.argmax(predictions_raw, axis=1)
    
    predictions = encoder.inverse_transform(predictions_raw)
    
    return evaluate_predictions(predictions, labels_test)


if __name__ == '__main__':
    feat = 'mc'
    gen = 'ps'
    prep = 'mfcc15'
    
    if gen == 'all' :
        genres = get_genres('all')
    if gen == 'ps' :
        genres = get_genres('paper_split')
    if gen == 'cs1' :
        genres = get_genres('custom_split')
     
    
    data, labels, data_test, labels_test = load_data(split=0.8, shuffle=True, prep=prep, take_middle=True, genres=genres)
    
    evaluation = train_eval_deep(data, labels, data_test, labels_test)


"""

feat = 'mc'
gen = 'ps'
prep = 'mfcc15'

if gen == 'all' :
    genres = get_genres('all')
if gen == 'ps' :
    genres = get_genres('paper_split')
if gen == 'cs1' :
    genres = get_genres('custom_split')
 
features = get_features(feat)

data_train, label_train, data_test, label_test = load_data(split=0.8, shuffle=True, prep=prep, take_middle=True, genres=genres, features=features)

encoder = LabelEncoder()
encoder.fit(label_train)
encoded_Y = encoder.transform(label_train)

dummy_y = np_utils.to_categorical(encoded_Y)
 
def baseline_model(n_classes, input_dim=57):
    model = Sequential()
    
    model.add(Dense(40, input_dim=input_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.1))
    
    model.add(Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=0.2)))
    model.add(BatchNormalization())
    
    model.add(Dense(10, activation=tf.keras.layers.LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.05))
    
    model.add(Dense(n_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

model = baseline_model(len(genres), data_train.shape[1])

val_split = 0.75
data_len = data_train.shape[0]
n_split = round(data_len * val_split)

n_epochs = 200
batch_size = 40


history = model.fit(data_train[:n_split], dummy_y[:n_split], validation_data=(data_train[n_split:], dummy_y[n_split:]), epochs=n_epochs, batch_size=batch_size)

model_dir = './models/deepnet/'

model_names = [ mod_name for mod_name in os.listdir(model_dir) ]
        
model_type_num = str(len(model_names) + 1)
model_path = model_dir + 'model_' + feat + '_' + gen + '_' + prep + '_' + model_type_num
        
model.save(model_path)
        
hist_df = pd.DataFrame(history.history) 
        
hist_json_file = model_dir + 'history_' + feat + '_' + gen + '_' + prep + '_' + model_type_num + '.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)


predictions_raw = model.predict(data_test)
predictions_raw = np.argmax(predictions_raw, axis=1)

predictions = encoder.inverse_transform(predictions_raw)

overall, classwise, conf, n_classes = evaluate_predictions(predictions, label_test)

print(overall)
print(classwise)
print(conf)
print(n_classes)



for model in model_names :
    if 'history' in model:
        continue
    
    test_m = tf.keras.models.load_model(model_dir + model)
    
    print(model)
    
    if model == 'model_mc_ps_mfcc15_7' or model == 'model_mc_ps_mfcc15_13' or model == 'model_mc_ps_mfcc15_1' :
        test_m.summary()
    
    predictions_raw = test_m.predict(data_test)
    predictions_raw = np.argmax(predictions_raw, axis=1)
    
    predictions = encoder.inverse_transform(predictions_raw)
    
    overall, classwise, conf, n_classes = evaluate_predictions(predictions, label_test)
    
    print(overall)
    print(classwise)
    print(conf)
    print(n_classes)
"""


