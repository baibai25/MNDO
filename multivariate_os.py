import numpy as np
import pandas as pd

def find_zerostd(pos):
    std = pos.std()
    mean = pos.mean()

    zero_list = []
    zero_mean = []

    for i in range(len(pos.columns)):
        if std[i] == 0:
            print(pos.columns[i])
            zero_list.append(pos.columns[i])
            zero_mean.append(mean[i])
            pos = pos.drop(pos.columns[i], axis=1)
    print(zero_list)
    print(zero_mean)
    return zero_list, zero_mean, pos
    
def mnd_os(zero_list, zero_mean, pos, num_minority):
    #calc correlation and covert absolute value
    corr = abs(pos.corr())
    print(corr)
    
    #find strong correlation attribute
    corr_col = []
    corr_ind = []
    for i in range(len(corr.columns)):
        sort = corr.iloc[:, [i]] #extract one index
        sort = sort.sort_values(by=sort.columns[0], ascending=False) #sort
        corr_col.append(sort.columns[0]) #strong corr coulumns
        corr_ind.append(sort.index[1]) #strong corr index
    print(corr_col)
    print(corr_ind)
    
    #calc mean and covariance
    mean_list = []
    cov_list = []
    for i in range(len(pos.columns)):
        mean_list.append([pos[corr_col[i]].mean(), pos[corr_ind[i]].mean()])
        cov_list.append(pd.concat([pos[corr_col[i]], pos[corr_ind[i]]], axis=1).cov())
        
    # generate new sample
    tmp = []
    np.random.seed(seed=42)
    for mean, cov in zip(mean_list, cov_list):
        mul_x, mul_y = np.random.multivariate_normal(mean, cov, num_minority).T
        tmp.append(mul_x)
        #sns.jointplot(mul_x, mul_y, kind="resid")
    
    #append original data
    df = pd.DataFrame(tmp).T
    df.columns = pos.columns
    
    if len(zero_list) != 0:
        for i in range(len(zero_list)):
            df[zero_list[i]] = zero_mean[i]
            
    df['Label'] = 1
    df.to_csv('/home/yura/Desktop/mlpd_train.csv', index=False)
    return df
    #df
