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

    
def norm_data(data, mode='std') :
    
    if mode == 'std' :
        data_norm = (data-data.mean())
        data_norm = data_norm / data_norm.std()
    elif mode == 'minmax' :
        data_norm = data - np.min(data)
        data_norm = (data_norm / np.max(data_norm)) * 2
        data_norm = data_norm - 1
        
        
    return data_norm

def get_genres(mode='all') :
    genres = []
    if mode == 'all' :
        genres = ['jazz', 'classical', 'blues', 'rock', 'country', 'hiphop', 'metal', 'reggae', 'pop', 'disco']
    elif mode == 'paper_split' :
        genres = ['jazz', 'classical', 'metal', 'pop']
    elif mode == 'custom_split' :
        genres = ['jazz', 'classical', 'metal', 'pop', 'disco']
        
    return genres

def get_features(mode='all') :
    features = []
    
    if mode == 'all' :
        features = ['mean', 'var', 'cov']
    else:
        if 'm' in mode :
            features.append('mean')
        if 'v' in mode :
            features.append('var')
        if 'c' in mode :
            features.append('cov')
            
    return features


def extract_features(data, features) :
    feat_vect = []
    
    if 'mean' in features :
        meanvect = np.mean(data, axis=1)
        meanvect = norm_data(meanvect)
        feat_vect = np.concatenate((feat_vect, meanvect))
        
    if 'var' in features :
        varvect = np.var(data, axis=1)
        varvect = norm_data(varvect)
        feat_vect = np.concatenate((feat_vect, varvect))
        
    if 'cov' in features :
        covmat = np.cov(data)
        covvect = []
        for idx, row in enumerate(covmat):
            covvect = np.concatenate((covvect, row[idx:]))
        covvect = norm_data(covvect)
        feat_vect = np.concatenate((feat_vect, covvect))
    
    return feat_vect


def load_data(split=0.8, shuffle=False, prep='instant', take_middle=False, genres=[], features=[]) :   
    if len(genres) == 0 :
        genres = get_genres()
    
    data, labels = load_data_csv(shuffle, prep, genres, features)

    if len(data) > 0 :
        n_split = round(len(data) * split)
        
        data_train = data[:n_split]
        data_test = data[n_split:]
        
        label_train = labels[:n_split]
        label_test = labels[n_split:]
        
        return data_train, label_train, data_test, label_test
    else :
        return [], [], [], []
    

def load_data_csv(shuffle=False, prep='instant', genres = [], features=[], take_middle=False) :
    if len(genres) == 0 :
        genres = get_genres()
        
    if len(features) == 0 :
        features = get_features()
    
    data_raw = pd.read_csv('./dataset/Data/features_30_sec.csv')
    
    data = data_raw.loc[data_raw['label'] == genres[0]]    
    
    for genre in genres[1:] :
        data = data.append(data_raw.loc[data_raw['label'] == genre], ignore_index=True)
    
    if shuffle :
        data = data.sample(frac = 1)

    labels = data['label']
    
    if prep == 'mfcc' :
        data = prep_data_mfcc(data, features=features, take_middle=take_middle)
    elif prep == 'mfcc15' :
        data = prep_data_mfcc(data, features=features, mfcc_cutoff=15, take_middle=take_middle)
    elif prep == 'spect' :
        data = prep_data_spect(data, features=features, take_middle=take_middle)
    elif prep == 'stft' :
        data = prep_data_stft(data, features=features, take_middle=take_middle)
    elif prep == 'instant' :
        data = prep_data_instant(data)
    elif prep == 'spectconv' :
        data = prep_data_spect_conv(data, take_middle=take_middle)
    elif prep == 'stftconv' :
        data = prep_data_stft_conv(data, take_middle=take_middle)
    elif prep == 'mfccconv' :
        data = prep_data_mfcc_conv(data, take_middle=take_middle)
    elif prep == 'mfccstftconv' :
        data = prep_data_mfcc_stft_conv(data, take_middle=take_middle)
    elif prep == 'mfccspectconv' :
        data = prep_data_mfcc_spect_conv(data, take_middle=take_middle)
    else :
        return  [], []
    
    return data, labels

def prep_data_instant(data) :
    data = data.drop(columns=data.columns[0:2])
    data = data.drop(columns=data.columns[-1])
    
    data = norm_data(data)
    
    return data

def prep_data_mfcc(data, features=[], mfcc_cutoff=20, take_middle=False) :
    if len(features) == 0:
        feature = get_features()
    
    if mfcc_cutoff > 20 :
        mfcc_cutoff = 20
    
    n_rows = data.shape[0]
    n_features = 0
    
    if 'mean' in features :
        n_features += mfcc_cutoff
    if 'var' in features :
        n_features += mfcc_cutoff
    if 'cov' in features :
        n_features += int((mfcc_cutoff + 1) * mfcc_cutoff / 2)

    data_mfcc = np.zeros((n_rows, n_features))

    i = 0
    for index, row in data.iterrows():
        filename = row['filename']
        if filename != 'jazz.00054.wav':
            path = './dataset/Data/genres_original/' + filename.split('.')[0] + '/' + filename
            
            sample_rate, samples = wavfile.read(path)
            if take_middle :
                sample_len = len(samples)
                #samples_2 = samples[: sample_len // 2 - sample_len // 4] + samples[sample_len // 2 + sample_len // 4:]
                samples = samples[sample_len // 2 - sample_len // 4 : sample_len // 2 + sample_len // 4 ]
            
            mfcc = librosa.feature.mfcc(samples.astype('float64'), sample_rate)
            
            if mfcc_cutoff < mfcc.shape[0] :
                mfcc = mfcc[:mfcc_cutoff, :]
            
            data_mfcc[i, :] = extract_features(mfcc, features)
        i += 1
    
    return data_mfcc


def prep_data_spect(data, features=[], take_middle=False) :
    
    n_rows = data.shape[0]
    n_features = 0
    
    spect_height = 129
    
    if 'mean' in features :
        n_features += spect_height
    if 'var' in features :
        n_features += spect_height
    if 'cov' in features :
        n_features += int((spect_height + 1) * spect_height / 2)
    
    data_spect = np.zeros((n_rows, n_features))
    
    i = 0
    for index, row in data.iterrows():
        filename = row['filename']
        if filename != 'jazz.00054.wav':
            path = './dataset/Data/genres_original/' + filename.split('.')[0] + '/' + filename
            
            sample_rate, samples = wavfile.read(path)
            
            if take_middle :
                sample_len = len(samples)
                samples = samples[sample_len // 2 - sample_len // 4 : sample_len // 2 + sample_len // 4 ]
            
            frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

            data_spect[i, :] = extract_features(spectrogram, features)
        i += 1

    return data_spect


def prep_data_stft(data, features=[], take_middle=False) :
    
    n_rows = data.shape[0]
    n_features = 0
    
    spect_height = 129
    
    if 'mean' in features :
        n_features += spect_height
    if 'var' in features :
        n_features += spect_height
    if 'cov' in features :
        n_features += int((spect_height + 1) * spect_height / 2)
    
    data_spect = np.zeros((n_rows, n_features), dtype=np.complex_)
    
    i = 0
    for index, row in data.iterrows():
        filename = row['filename']
        if filename != 'jazz.00054.wav':
            path = './dataset/Data/genres_original/' + filename.split('.')[0] + '/' + filename
            
            sample_rate, samples = wavfile.read(path)
            
            if take_middle :
                sample_len = len(samples)
                samples = samples[sample_len // 2 - sample_len // 4 : sample_len // 2 + sample_len // 4 ]
            
            frequencies, times, stft = signal.stft(samples, sample_rate)

            data_spect[i, :] = extract_features(stft, features)
        i += 1

    return data_spect



def prep_data_spect_conv(data, take_middle=False) :
    n_rows = data.shape[0]
    
    n_features = 512
    spect_height = 128
    
    data_spect = []
    
    i = 0
    for index, row in data.iterrows():
        filename = row['filename']
        if filename != 'jazz.00054.wav':
            path = './dataset/Data/genres_original/' + filename.split('.')[0] + '/' + filename
            
            sample_rate, samples = wavfile.read(path)
            
            if take_middle :
                sample_len = len(samples)
                samples = samples[sample_len // 2 - sample_len // 4 : sample_len // 2 + sample_len // 4 ]
            
            frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
            spectrogram = spectrogram [0:spect_height, :]
            
            window_size = spectrogram.shape[1] // n_features
            
            spect_avg = np.zeros((spect_height, n_features, 1))
            for i in range(n_features) :
                spect_avg[:, i, :] = np.expand_dims(np.mean(spectrogram[:, i*window_size : (i+1) * window_size], axis=1), axis=-1)
            
            spect_avg = norm_data(spect_avg)
            
            data_spect.append(spect_avg)
        else : 
            data_spect.append(np.zeros((spect_height, n_features, 1)))
        i += 1

    return data_spect


def prep_data_stft_conv(data, take_middle=False) :
    n_rows = data.shape[0]
    
    n_features = 512
    spect_height = 128
    
    data_stft = []
    
    i = 0
    for index, row in data.iterrows():
        filename = row['filename']
        if filename != 'jazz.00054.wav':
            path = './dataset/Data/genres_original/' + filename.split('.')[0] + '/' + filename
            
            sample_rate, samples = wavfile.read(path)
            
            if take_middle :
                sample_len = len(samples)
                samples = samples[sample_len // 2 - sample_len // 4 : sample_len // 2 + sample_len // 4 ]
            
            frequencies, times, stft = signal.stft(samples, sample_rate)
            stft = stft [0:spect_height, :]
            
            window_size = stft.shape[1] // n_features
            
            
            stft_avg = np.zeros((spect_height, n_features, 1), dtype=np.complex_)
            for i in range(n_features) :
                stft_avg[:, i, :] = np.expand_dims(np.mean(stft[:, i*window_size : (i+1) * window_size], axis=1), axis=-1)
            
            stft_avg = norm_data(stft_avg)
            
            data_stft.append(stft_avg)
        else : 
            data_stft.append(np.zeros((spect_height, n_features, 1), dtype=np.complex_))
        i += 1

    return data_stft


def prep_data_mfcc_conv(data, features=[], mfcc_bins=128, take_middle=False) :
    if len(features) == 0:
        feature = get_features()
    
    n_features = 512
    data_mfcc = []

    i = 0
    for index, row in data.iterrows():
        filename = row['filename']
        if filename != 'jazz.00054.wav':
            path = './dataset/Data/genres_original/' + filename.split('.')[0] + '/' + filename
            
            sample_rate, samples = wavfile.read(path)
            if take_middle :
                sample_len = len(samples)
                #samples_2 = samples[: sample_len // 2 - sample_len // 4] + samples[sample_len // 2 + sample_len // 4:]
                samples = samples[sample_len // 2 - sample_len // 4 : sample_len // 2 + sample_len // 4 ]
            
            mfcc = librosa.feature.mfcc(samples.astype('float64'), sample_rate, n_mfcc=mfcc_bins)
            
            window_size = mfcc.shape[1] // n_features
            
            mfcc_avg = np.zeros((mfcc_bins, n_features, 1))
            for i in range(n_features) :
                mfcc_avg[:, i, :] = np.expand_dims(np.mean(mfcc[:, i*window_size : (i+1) * window_size], axis=1), axis=-1)
            
            mfcc_avg = norm_data(mfcc_avg)
            
            data_mfcc.append(mfcc_avg)
        else : 
            data_mfcc.append(np.zeros((mfcc_bins, n_features, 1)))
        i += 1
    
    return data_mfcc


def prep_data_mfcc_stft_conv(data, features=[], mfcc_bins=128, take_middle=False) :
    if len(features) == 0:
        feature = get_features()
    
    n_features = 512
    data_mfcc = []

    i = 0
    for index, row in data.iterrows():
        filename = row['filename']
        if filename != 'jazz.00054.wav':
            path = './dataset/Data/genres_original/' + filename.split('.')[0] + '/' + filename
            
            sample_rate, samples = wavfile.read(path)
            if take_middle :
                sample_len = len(samples)
                #samples_2 = samples[: sample_len // 2 - sample_len // 4] + samples[sample_len // 2 + sample_len // 4:]
                samples = samples[sample_len // 2 - sample_len // 4 : sample_len // 2 + sample_len // 4 ]
            
            mfcc = librosa.feature.mfcc(samples.astype('float64'), sample_rate, n_mfcc=mfcc_bins)
            
            window_size = mfcc.shape[1] // n_features
            
            mfcc_avg = np.zeros((mfcc_bins, n_features, 1))
            for i in range(n_features) :
                mfcc_avg[:, i, :] = np.expand_dims(np.mean(mfcc[:, i*window_size : (i+1) * window_size], axis=1), axis=-1)
            
            mfcc_avg = norm_data(mfcc_avg)
            
            frequencies, times, stft = signal.stft(samples, sample_rate)
            
            stft = stft [0:mfcc_bins, :]
            
            window_size = stft.shape[1] // n_features
            
            stft_avg = np.zeros((mfcc_bins, n_features, 1), dtype=np.complex_)
            for i in range(n_features) :
                stft_avg[:, i, :] = np.expand_dims(np.mean(stft[:, i*window_size : (i+1) * window_size], axis=1), axis=-1)
            
            stft_avg = norm_data(stft_avg)
            
            combined = np.zeros((mfcc_bins, n_features, 2), dtype=np.complex_)
            
            combined[:, :, 0] = mfcc_avg[:, :, 0]
            combined[:, :, 1] = stft_avg[:, :, 0]
            
            data_mfcc.append(combined)
        else : 
            data_mfcc.append(np.zeros((mfcc_bins, n_features, 2), dtype=np.complex_))
        i += 1
    
    return data_mfcc


def prep_data_mfcc_spect_conv(data, features=[], mfcc_bins=128, take_middle=False) :
    if len(features) == 0:
        feature = get_features()
    
    n_features = 512
    data_mfcc = []

    i = 0
    for index, row in data.iterrows():
        filename = row['filename']
        if filename != 'jazz.00054.wav':
            path = './dataset/Data/genres_original/' + filename.split('.')[0] + '/' + filename
            
            sample_rate, samples = wavfile.read(path)
            if take_middle :
                sample_len = len(samples)
                #samples_2 = samples[: sample_len // 2 - sample_len // 4] + samples[sample_len // 2 + sample_len // 4:]
                samples = samples[sample_len // 2 - sample_len // 4 : sample_len // 2 + sample_len // 4 ]
            
            mfcc = librosa.feature.mfcc(samples.astype('float64'), sample_rate, n_mfcc=mfcc_bins)
            
            window_size = mfcc.shape[1] // n_features
            
            mfcc_avg = np.zeros((mfcc_bins, n_features, 1))
            for i in range(n_features) :
                mfcc_avg[:, i, :] = np.expand_dims(np.mean(mfcc[:, i*window_size : (i+1) * window_size], axis=1), axis=-1)
            
            mfcc_avg = norm_data(mfcc_avg)
            
            frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
            stft = spectrogram
            stft = stft [0:mfcc_bins, :]
            
            window_size = stft.shape[1] // n_features
            
            stft_avg = np.zeros((mfcc_bins, n_features, 1))
            for i in range(n_features) :
                stft_avg[:, i, :] = np.expand_dims(np.mean(stft[:, i*window_size : (i+1) * window_size], axis=1), axis=-1)
            
            stft_avg = norm_data(stft_avg)
            
            combined = np.zeros((mfcc_bins, n_features, 2))
            
            combined[:, :, 0] = mfcc_avg[:, :, 0]
            combined[:, :, 1] = stft_avg[:, :, 0]
            
            data_mfcc.append(combined)
        else : 
            data_mfcc.append(np.zeros((mfcc_bins, n_features, 2)))
        i += 1
    
    return data_mfcc







