# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 19:36:09 2021

@author: Kert PC
"""

from sklearn import svm
from loadData import load_split_data, load_data, load_split_data_mfcc


def superclassify(label) :
    ret = ''
    if label == 'blues' or label == 'jazz' or label == 'metal' or label == 'rock' or label == 'classical' :
        ret = 'group1'
    else :
        ret = 'group2'
        
    return ret


points, labels, t_p, t_l = load_split_data_mfcc(shuffle=True)

nu_labels = labels
nu_t_labels = t_l

nu_labels.apply(superclassify)
nu_t_labels.apply(superclassify)

"""
nu_labels = ['group1' for i in range(len(nu_labels)) if nu_labels[i] == 'blues' or nu_labels[i] == 'country' or nu_labels[i] == 'metal' or nu_labels[i] == 'rock' or nu_labels[i] == 'reggae']
nu_labels = ['group2' for i in range(len(nu_labels)) if nu_labels[i] == 'classical' or nu_labels[i] == 'disco' or nu_labels[i] == 'hiphop' or nu_labels[i] == 'jazz' or nu_labels[i] == 'pop']

nu_t_labels = ['group1' for i in range(len(nu_labels)) if nu_labels[i] == 'blues' or nu_labels[i] == 'country' or nu_labels[i] == 'metal' or nu_labels[i] == 'rock' or nu_labels[i] == 'reggae']
nu_t_labels = ['group2' for i in range(len(nu_labels)) if nu_labels[i] == 'classical' or nu_labels[i] == 'disco' or nu_labels[i] == 'hiphop' or nu_labels[i] == 'jazz' or nu_labels[i] == 'pop']
"""

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(points, labels)


preds = clf.predict(t_p)

tru = 0
fal = 0

labels = list(labels)
t_l = list(t_l)

nu_labels = list(nu_labels)
nu_t_labels = list(nu_t_labels)

for idx, pred in enumerate(preds):
    if pred == t_l[idx] :
        tru += 1
    else:
        fal += 1
        
        
acc = tru / (tru + fal)
print(acc)