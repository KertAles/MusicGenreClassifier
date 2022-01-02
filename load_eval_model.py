# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 11:02:49 2022

@author: Kert PC
"""

import numpy as np

from loadData import load_data
from evaluation import evaluate_predictions

from clustering import train_eval_clustering
from svm import train_eval_svm
from deepnet import train_eval_deep, load_eval_deep
from convnet_1 import train_eval_conv, load_eval_conv

prep_mode = 'mfcc'
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
data_unord, labels_unord, blah, blah2 = load_data(split=1.0, prep=prep_mode, shuffle=False, take_middle='mfcc' in prep_mode, genres=genres, features=features)


with open('./data_splits/' + prep_mode + '_' + str(len(genres)) + '.txt', 'r') as f:
    idxs = f.readline().split(' ')
    idxs = idxs[:-1]

data = [data_unord[int(idxs[0])]]
labels = [labels_unord[int(idxs[0])]]

for idx in idxs[1:] :
    data = np.concatenate((data, [data_unord[int(idx)]]), axis=0)
    labels = np.concatenate((labels, [labels_unord[int(idx)]]), axis=0)

val_split = 70/90

if is_conv :
    n_split = 200
    val_split = 60/80
    """
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
    """
    
else :
    k_fold = len(genres)
    
    k_fold_conf = {}
    k_fold_conf['deep'] = {}
    
    for gen in genres :
        k_fold_conf[gen] = {}
        for gen2 in genres :
            k_fold_conf[gen][gen2] = 0
    
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
        
        model_name = 'model_' + str(289 + i * 2)
        
        deep_acc = load_eval_deep(data_test, labels_test, model_name)
        
        for gen in genres :
            for gen2 in genres :
                k_fold_conf[gen][gen2] += deep_acc[2][gen][gen2]
        
        
        

    #k_fold_conf['deep'] /= k_fold

with open('./latex_table.txt', 'w') as f:
    f.write('/ & ' + ' & '.join(genres) + ' \\\\ \hline \n')
    
    for gen in genres :
        f.write(gen)
        for gen2 in genres :
            f.write(' & ' + str(k_fold_conf[gen][gen2]))
        f.write(' \\\\ \n')
        
        
stri = 'l|'      
for gen in genres :
    stri = stri + ':c@{\hskip 0.03in}'
