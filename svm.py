# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 19:36:09 2021

@author: Kert PC
"""

from sklearn import svm
from loadData import load_data, get_genres, get_features
from evaluation import evaluate_predictions


def train_eval_svm(data, labels, test_data, test_labels, decision_function='ovr'): 
        
    clf = svm.SVC(decision_function_shape=decision_function)
    clf.fit(data, labels)
    
    classes = set(labels)
    
    confusion = {}
    
    for clas in classes :
        confusion[clas] = {}
        for clas2 in classes :
            confusion[clas][clas2] = 0
    
    predictions = clf.predict(test_data)
    
    return evaluate_predictions(predictions, test_labels)

    

if __name__ == '__main__':
    data, labels, test_data, test_labels = load_data(shuffle=True)
    
    evaluation = train_eval_svm(data, labels, test_data, test_labels)
    
    
"""
feat = 'mvc'
gen = 'all'
prep = 'instant'

if gen == 'all' :
    genres = get_genres('all')
if gen == 'ps' :
    genres = get_genres('paper_split')
if gen == 'cs1' :
    genres = get_genres('custom_split')
 
features = get_features(feat)

data_train, label_train, data_test, label_test = load_data(split=0.8, shuffle=True, prep=prep, take_middle=False, genres=genres, features=features)
"""