# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 18:55:57 2021

@author: Kert PC
"""

from sklearn.cluster import KMeans
from loadData import load_split_data, load_data
from collections import Counter


points, labels, t_p, t_l = load_split_data(shuffle=True)

labels = list(labels)
t_l = list(t_l)

n_clusters = 30


kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(points)
#print(kmeans.cluster_centers_)
y_km = kmeans.fit_predict(points)

def most_common(lst):
    return max(set(lst), key=lst.count)

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
print(acc)



preds = kmeans.predict(t_p)

tr = 0
fa = 0

for idx, pred in enumerate(preds) :
    
    if classes[pred] == t_l[idx] :
        tr += 1
    else :
        fa += 1
    
ac = tr / (tr + fa)

print(ac)

