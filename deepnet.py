# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 15:37:48 2021

@author: Kert PC
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dropout, concatenate, Dense
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


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

from loadData import load_split_data, load_split_data_spect, load_split_data_mfcc

data_train, label_train, data_test, label_test = load_split_data_mfcc(split=1.0, shuffle=True)

encoder = LabelEncoder()
encoder.fit(label_train)
encoded_Y = encoder.transform(label_train)

dummy_y = np_utils.to_categorical(encoded_Y)
 
def baseline_model():
    model = Sequential()
    model.add(Dense(400, input_dim=230, activation='relu'))
    #model.add(Dropout(0.05))
    #model.add(Dense(300, input_dim=57, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(400, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

model = baseline_model()

model.fit(data_train[800:], dummy_y[800:], validation_data=(data_train[:800], dummy_y[:800]), epochs=150, batch_size=20)


#estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=20)
#kfold = KFold(n_splits=5, shuffle=True)
#results = cross_val_score(estimator, data_train, dummy_y, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#predictions = estimator.predict(data_test)

