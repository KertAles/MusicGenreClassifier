# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 14:48:58 2021

@author: Kert PC
"""

def evaluate_predictions(predictions, ground_truth, classes=[]) :
    ground_truth = list(ground_truth)
    confusion = {}
    acc_vect = {}
    num_in_class = {}
    
    if len(classes) == 0 :
        classes = set(ground_truth)
    
    for clas in classes :
        confusion[clas] = {}
        for clas2 in classes :
            confusion[clas][clas2] = 0

    tru = 0
    fal = 0
    
    for idx, pred in enumerate(predictions) :
        confusion[pred][ground_truth[idx]] += 1
        
        if pred == ground_truth[idx] :
            tru += 1
        else:
            fal += 1
        
    for clas in classes :
        summ = 0
        acc_vect[clas] = 0
        
        for clas2 in classes :
            summ += confusion[clas2][clas]
            if clas == clas2 :
                acc_vect[clas] += confusion[clas][clas2]
        acc_vect[clas] /= summ
        num_in_class[clas] = summ
        
        
      
    acc = tru / (tru + fal)

    return (acc, tru, fal), acc_vect, confusion, num_in_class

