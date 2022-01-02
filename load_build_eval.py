# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 16:14:11 2022

@author: Kert PC
"""

import numpy as np

from loadData import load_data
from evaluation import evaluate_predictions

from clustering import train_eval_clustering
from svm import train_eval_svm
from deepnet import train_eval_deep, load_eval_deep
from convnet_1 import train_eval_conv, load_eval_conv

prep_mode = 'mfccconv'
is_conv = 'conv' in prep_mode
genres = ['jazz', 'classical', 'metal', 'pop', 'country', 'disco', 'rock', 'reggae', 'hiphop', 'blues'] #10
#genres = ['jazz', 'classical', 'metal', 'pop', 'country', 'disco', 'rock', 'reggae', 'hiphop'] #9
#genres = ['jazz', 'classical', 'metal', 'pop', 'country', 'disco', 'rock', 'reggae'] #8
#genres = ['jazz', 'classical', 'metal', 'pop', 'country', 'disco', 'rock'] #7
#genres = ['jazz', 'classical', 'metal', 'pop', 'country', 'disco'] #6
#genres = ['jazz', 'classical', 'metal', 'pop', 'country'] #5
#genres = ['classical', 'metal', 'pop'] #3
#genres = ['classical',  'pop'] #2

#genres = ['jazz', 'classical', 'metal', 'pop']
features = ['mean', 'cov']

#load_data(split=0.8, shuffle=False, prep='instant', take_middle=False, genres=[], features=[]) :   
data, labels, blah, blah2 = load_data(split=1.0, prep=prep_mode, shuffle=True, take_middle='mfcc' in prep_mode, genres=genres, features=features)

with open('./data_splits/' + prep_mode + '_' + str(len(genres)) + '.txt', 'w') as f:
    for idx in labels.index :
        f.write(str(idx) + ' ')

val_split = 70/90

if is_conv :
    n_split = 200
    val_split = 60/80
    
    accuracies = {}
    accuracies['conv1'] = 0 
   
    print("convnet1: ")
    
    data_test = data[:n_split]
    labels_test = labels[:n_split]
    
    data_train = data[n_split:]
    labels_train = labels[n_split:]
  
    conv1_acc = train_eval_conv(data_train, labels_train, data_test, labels_test, val_split, 1)
    accuracies['conv1'] += conv1_acc[0][0]
        
    print("Accuracy : " + str(conv1_acc[0][0]))

    with open('./eval_results/' + prep_mode + '_' + str(len(genres)) + '.txt', 'w') as f:
        f.write('Convnet1: ' + ' ' + str(accuracies['conv1']) + '\n')
    
    
    
else :
    k_fold = len(genres)
    
    k_fold_accuracies = {}
    if prep_mode != 'stft' :
        k_fold_accuracies['clust'] = 0
        k_fold_accuracies['svm'] = 0
    k_fold_accuracies['deep'] = 0
    
    if prep_mode != 'stft' :
        print("N of fold | cluster | svm | deepnet")
    else :
        print("N of fold | deepnet")
        
        
    for i in range(k_fold) :
        data_test = data[i*100: (i+1)*100]
        labels_test = labels[i*100: (i+1)*100]
        
        if i == 0 :
            data_train = data[100:]
            labels_train = labels[100:]
        elif i == 9 :
            data_train = data[:900]
            labels_train = labels[:900]
        else :
            data_train = data[: i*100]
            data_train = np.concatenate((data_train, data[(i+1) * 100:]), axis=0)
            labels_train = labels[: i*100]
            labels_train = np.concatenate((labels_train, labels[(i+1) * 100:]), axis=0)
        
        if prep_mode != 'stft':
            clust_acc = train_eval_clustering(data_train, labels_train, data_test, labels_test, n_clusters=len(genres)*2)
            k_fold_accuracies['clust'] += clust_acc[0]
            
            svm_acc = train_eval_svm(data_train, labels_train, data_test, labels_test)
            k_fold_accuracies['svm'] += svm_acc[0][0]
        
        deep_acc = train_eval_deep(data_train, labels_train, data_test, labels_test, val_split)
        k_fold_accuracies['deep'] += deep_acc[0][0]
        
        if prep_mode != 'stft' :
            print(str(i+1) + "tzh fold : " + str(clust_acc[0]) + ';  ' + str(svm_acc[0][0]) + ';  ' + str(deep_acc[0][0]))
        else :
            print(str(i+1) + "tzh fold : " + str(deep_acc[0][0]))
    if prep_mode != 'stft' :
        k_fold_accuracies['clust'] /= k_fold
        k_fold_accuracies['svm'] /= k_fold
    k_fold_accuracies['deep'] /= k_fold  
    

    with open('./eval_results/' + prep_mode + '_' + str(len(genres)) + '.txt', 'w') as f:
        if prep_mode != 'stft' :
            f.write('Clustering: ' + ' ' + str(k_fold_accuracies['clust']) + '\n')
            f.write('SVM: ' + ' ' + str(k_fold_accuracies['svm']) + '\n')
        f.write('Deepnet: ' + ' ' + str(k_fold_accuracies['deep']) + '\n')
