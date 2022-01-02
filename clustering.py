# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 18:55:57 2021

@author: Kert PC
"""

from sklearn.cluster import KMeans
from loadData import load_data
from collections import Counter
from evaluation import evaluate_predictions

def most_common(lst):
    return max(set(lst), key=lst.count)

def train_eval_clustering(points, labels, test_points, test_labels, n_clusters=20) :
    
    labels = list(labels)
    test_labels = list(test_labels)
    
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(points)
    y_km = kmeans.fit_predict(points)
    
    tru = 0
    fal = 0
    
    classes = []
    
    for i in range(0,n_clusters) :
        indices = [j for j, x in enumerate(y_km) if x == i]
        
        c = Counter([labels[ix] for ix in indices])
        tup = c.most_common(1)
        
        classes.append(tup[0][0])
        
        tru += tup[0][1]
        fal += len(indices) - tup[0][1]
        
        
    acc = tru / (tru + fal)
    
    
    
    preds = kmeans.predict(test_points)
    
    tr = 0
    fa = 0
    
    for idx, pred in enumerate(preds) :
        
        if classes[pred] == test_labels[idx] :
            tr += 1
        else :
            fa += 1
        
    ac = tr / (tr + fa)
    
    return (ac, tr, fa)
    """
    classes = []

    for i in range(0,n_clusters) :
        indices = [j for j, x in enumerate(y_km) if x == i]
        
        c = Counter([labels[ix] for ix in indices])
        tup = c.most_common(1)
        
        classes.append(tup[0][0])
        
    #print(classes)
    predictions = kmeans.predict(test_points)
    
    evaluation = evaluate_predictions(predictions, test_labels, classes)
    
    return evaluation
    """


if __name__ == '__main__':
    points, labels, test_points, test_labels = load_data(shuffle=True)
    
    evaluation = train_eval_clustering(points, labels, test_points, test_labels)