# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 15:44:15 2021

@author: Kert PC
"""

from sklearn import preprocessing
import pandas as pd
import numpy as np


from scipy import signal
from scipy.io import wavfile

import librosa


def load_data(shuffle=False) :
    data = pd.read_csv('./dataset/Data/features_30_sec.csv')
    
    if shuffle :
        data = data.sample(frac = 1)
        
    data = data.drop(columns=data.columns[0:2])
    labels = data['label']
    data = data.drop(columns=data.columns[-1])
    
    data = norm_data(data)
    
    return data, labels
    
    
def norm_data(data) :
    data_norm = (data-data.mean())
    data_norm = data_norm / data_norm.std()
    
    return data_norm


def load_split_data(split=0.8, shuffle=False) :
    data, labels = load_data(shuffle)

    n_split = round(len(data) * split)
    
    data_train = data[:n_split]
    data_test = data[n_split:]
    
    label_train = labels[:n_split]
    label_test = labels[n_split:]
    
    return data_train, label_train, data_test, label_test


def load_data_spect(shuffle=False) :
    data = pd.read_csv('./dataset/Data/features_30_sec.csv')
    
    #data = data[:][:50]
    
    
    if shuffle :
        data = data.sample(frac = 1)
        
    labels = data['label']
    
    data_spect = np.zeros((1000, 6215))
    
    i = 0
    for index, row in data.iterrows():
        filename = row['filename']
        if filename != 'jazz.00054.wav':
            path = './dataset/Data/genres_original/' + filename.split('.')[0] + '/' + filename
            
            sample_rate, samples = wavfile.read(path)
            frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
            
            spectrogram = spectrogram[10:120, :]
            
            meanvect = np.mean(spectrogram, axis=1)
            covmat = np.cov(spectrogram)
            
            vect = norm_data(meanvect)
            covvect = []
            
            for idx, row in enumerate(covmat):
                covvect = np.concatenate((covvect, row[idx:]))
        
            covvect = norm_data(covvect)
        
            data_spect[i, :] = np.concatenate((vect, covvect))
        i += 1


        
    #data_spect = norm_data(data_spect)
    
    return data_spect, labels


def load_split_data_spect(split=0.8, shuffle=False) :
    data, labels = load_data_spect(shuffle)

    n_split = round(len(data) * split)
    
    data_train = data[:n_split]
    data_test = data[n_split:]
    
    label_train = labels[:n_split]
    label_test = labels[n_split:]
    
    return data_train, label_train, data_test, label_test



def load_data_mfcc(shuffle=False) :
    data = pd.read_csv('./dataset/Data/features_30_sec.csv')
    
    #data = data[:][:500]
    
    if shuffle :
        data = data.sample(frac = 1)
        
    labels = data['label']
    
    data_spect = np.zeros((1000, 230))
    
    i = 0
    for index, row in data.iterrows():
        filename = row['filename']
        if filename != 'jazz.00054.wav':
            path = './dataset/Data/genres_original/' + filename.split('.')[0] + '/' + filename
            
            sample_rate, samples = wavfile.read(path)
            mfcc = librosa.feature.mfcc(samples.astype('float64'), sample_rate)
            
            meanvect = np.mean(mfcc, axis=1)
            covmat = np.cov(mfcc)
            
            vect = norm_data(meanvect)
            covvect = []
            
            for idx, row in enumerate(covmat):
                covvect = np.concatenate((covvect, row[idx:]))
        
            covvect = norm_data(covvect)
        
            data_spect[i, :] = np.concatenate((vect, covvect))
        i += 1


        
    #data_spect = norm_data(data_spect)
    
    return data_spect, labels


def load_split_data_mfcc(split=0.8, shuffle=False) :
    data, labels = load_data_mfcc(shuffle)

    n_split = round(len(data) * split)
    
    data_train = data[:n_split]
    data_test = data[n_split:]
    
    label_train = labels[:n_split]
    label_test = labels[n_split:]
    
    return data_train, label_train, data_test, label_test



