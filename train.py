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
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import roc_curve, auc
import re
from io import StringIO

# set save path
def set_path(basename):
    name = os.path.splitext(basename)
    pred_out = 'output/pred_{}.csv'.format(name[0])
    auc_out = 'output/auc_{}.csv'.format(name[0])
    save_path = [pred_out, auc_out]
    return save_path

# Multivariate over-sampling
def mndo(pos, num_minority):
    pos, zero_std = multivariate_os.find_zerostd(pos, num_minority)
    pos, no_corr = multivariate_os.no_corr(pos, num_minority)
    pos = multivariate_os.mnd_os(pos, num_minority)
    mndo_df = multivariate_os.append_data(pos, zero_std, no_corr)
    return mndo_df

# train data + mndo data
def append_mndo(X_train, y_train, df):
    X_mndo = df.drop('Label', axis=1)
    y_mndo = df.Label
    X_mndo = pd.concat([X_mndo, X_train])
    y_mndo = pd.concat([y_mndo, y_train])
    return X_mndo, y_mndo

# convert classification report to dataframe
def report_to_df(report):
    report = re.sub(r" +", " ", report).replace("avg / total", "avg/total").replace("\n ", "\n")
    report_df = pd.read_csv(StringIO("Classes" + report), sep=' ', index_col=0)        
    return(report_df)

if __name__ == '__main__':
    # Load dataset
    data = pd.read_csv(sys.argv[1])
    save_path = set_path(os.path.basename(sys.argv[1]))

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

    #-----------------
    # Preprocessing
    #-----------------
    # Multivariate over-sampling
    mndo_df = mndo(pos, num_minority)
    X_mndo, y_mndo = append_mndo(X_train, y_train, mndo_df)
    print('y_mndo: {}'.format(Counter(y_mndo)))

    for i in tqdm(range(100), desc="Preprocessing", leave=False):
        # Apply over-sampling
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
        X_mndo = normalize(X_mndo, norm='l2')
        os_list = [[X_train, y_train], [X_reg, y_reg], [X_ada, y_ada], [X_rand, y_rand], [X_mndo, y_mndo]]

    #-------------
    # Learning
    #-------------
    for i in tqdm(range(100), desc="Learning", leave=False):
        svm_clf = []
        pred_df = pd.DataFrame(index=[], columns=[])
        auc_list = []
        # svm
        for i in range(len(os_list)):
            svm_clf.append(svm.SVC(random_state=RANDOM_STATE, probability=True).fit(os_list[i][0], os_list[i][1]))
            
        for i in range(len(svm_clf)):
            pred = classification_report_imbalanced(y_test, svm_clf[i].predict(X_test))
            pred_df = pred_df.append(report_to_df(pred))
            # calc auc
            prob = svm_clf[i].predict_proba(X_test)[:,1]
            fpr, tpr, thresholds = roc_curve(y_test, prob, pos_label=1)
            roc_auc_area = auc(fpr, tpr)
            auc_list.append(roc_auc_area)
        delimiter = pd.DataFrame(columns=pred_df.columns, index=['#'])
        delimiter = delimiter.fillna('#')
        pred_df = pred_df.append(delimiter)
        
        #k-NN
        k=3
        knn_clf = []
        #knn_df = pd.DataFrame(index=[], columns=[])
        for i in range(len(os_list)):
            knn_clf.append(KNeighborsClassifier(n_neighbors=k).fit(os_list[i][0], os_list[i][1]))
            
        for i in range(len(knn_clf)):
            pred = classification_report_imbalanced(y_test, knn_clf[i].predict(X_test))
            pred_df = pred_df.append(report_to_df(pred))
            # calc auc
            prob = knn_clf[i].predict_proba(X_test)[:,1]
            fpr, tpr, thresholds = roc_curve(y_test, prob, pos_label=1)
            roc_auc_area = auc(fpr, tpr)
            auc_list.append(roc_auc_area)

        auc_df = pd.DataFrame(auc_list)
    #print(pred_df)
    #print(auc_df)

    # export resualt
    pred_df.to_csv(save_path[0])
    auc_df.to_csv(save_path[1])
