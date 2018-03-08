# coding: utf-8
import sys
import os.path
from tqdm import tqdm
import numpy as np
import pandas as pd
import multivariate_os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn import over_sampling
from sklearn.preprocessing import normalize
from numpy.random import multivariate_normal
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import roc_curve, auc
import re
from io import StringIO

# Load dataset
data = pd.read_csv(sys.argv[1])
basename = os.path.basename(sys.argv[1])
name = os.path.splitext(basename)
svm_out = 'output/svm_{}.csv'.format(name[0])
knn_out = 'output/knn_{}.csv'.format(name[0])
auc_out = 'output/auc_{}.csv'.format(name[0])

X = data.drop('Label', axis=1)
y = data.Label
pos = data[data.Label == 1]
pos = pos.drop('Label', axis=1)

# Split the data
RANDOM_STATE = 6
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=RANDOM_STATE)
cnt = Counter(y_train)
num_minority = int((cnt[0] - cnt[1]))
print('y_train: {}'.format(Counter(y_train)))
print('y_test: {}'.format(Counter(y_test)))
print(num_minority)

# Multivariate over-sampling
pos, zero_std = multivariate_os.find_zerostd(pos, num_minority)
pos, no_corr = multivariate_os.no_corr(pos, num_minority)
pos = multivariate_os.mnd_os(pos, num_minority)
df = multivariate_os.append_data(pos, zero_std, no_corr)


## Preprocessing
 # df + train data
X_mlpd = df.drop('Label', axis=1)
y_mlpd = df.Label
X_mlpd = pd.concat([X_mlpd, X_train])
y_mlpd = pd.concat([y_mlpd, y_train])
print('y_mlpd: {}'.format(Counter(y_mlpd)))

for i in tqdm(range(100), desc="Preprocessing"):
    #Apply over-sampling
    sm_reg = over_sampling.SMOTE(kind='regular', random_state=RANDOM_STATE)
    ada = over_sampling.ADASYN(random_state=RANDOM_STATE)
    rand = over_sampling.RandomOverSampler(random_state=RANDOM_STATE)
    X_reg, y_reg = sm_reg.fit_sample(X_train, y_train)
    X_ada, y_ada = ada.fit_sample(X_train, y_train)
    X_rand, y_rand = rand.fit_sample(X_train, y_train)

    # normalize
    X_train = normalize(X_train, norm='l2')
    X_reg = normalize(X_reg, norm='l2')
    X_ada = normalize(X_ada, norm='l2')
    X_rand = normalize(X_rand, norm='l2')
    X_mlpd = normalize(X_mlpd, norm='l2')
    os_list = [[X_train, y_train], [X_reg, y_reg],
               [X_ada, y_ada], [X_rand, y_rand], [X_mlpd, y_mlpd]]

# convert classification report to dataframe
def report_to_df(report):
    report = re.sub(r" +", " ", report).replace("avg / total", "avg/total").replace("\n ", "\n")
    report_df = pd.read_csv(StringIO("Classes" + report), sep=' ', index_col=0)        
    return(report_df)

# start learning
for i in tqdm(range(100), desc="Learning"):
    svm_clf = []
    svm_df = pd.DataFrame(index=[], columns=[])
    auc_list = []

    # svm
    for i in range(len(os_list)):
        svm_clf.append(svm.SVC(random_state=RANDOM_STATE, probability=True, class_weight='balanced').fit(os_list[i][0], os_list[i][1]))
        
    for i in range(len(svm_clf)):
        pred = classification_report_imbalanced(y_test, svm_clf[i].predict(X_test))
        svm_df = svm_df.append(report_to_df(pred))
        #calc auc
        prob = svm_clf[i].predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(y_test, prob)
        roc_auc_area = auc(fpr, tpr)
        #print('AUC={}'.format(roc_auc_area))
        auc_list.append(roc_auc_area)

    #k-NN
    k=3
    knn_clf = []
    knn_df = pd.DataFrame(index=[], columns=[])

    for i in range(len(os_list)):
        knn_clf.append(KNeighborsClassifier(n_neighbors=k).fit(os_list[i][0], os_list[i][1]))
        
    for i in range(len(knn_clf)):
        pred = classification_report_imbalanced(y_test, knn_clf[i].predict(X_test))
        knn_df = knn_df.append(report_to_df(pred))
        #calc auc
        prob = knn_clf[i].predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(y_test, prob)
        roc_auc_area = auc(fpr, tpr)
        #print('AUC={}'.format(roc_auc_area))
        auc_list.append(roc_auc_area)

    auc_df = pd.DataFrame(auc_list)

    # export resualt
    svm_df.to_csv(svm_out)
    knn_df.to_csv(knn_out)
    auc_df.to_csv(auc_out)
