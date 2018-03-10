import numpy as np
import pandas as pd
from tqdm import tqdm

# Find zero standard deviation
def find_zerostd(pos, num_minority): 
    for i in tqdm(range(100), desc="Searching zero std", leave=False):
        std = pos.std()
        mean = pos.mean()
        zero_list = []
        zero_mean = []

        for i in range(len(pos.columns)):
            if std[i] == 0:
                zero_list.append(pos.columns[i])
                zero_mean.append(mean[i])
        #print(zero_list)
        #print(zero_mean)

    if (len(zero_list) == 0) and (len(zero_mean) == 0): 
        print("Not found zero std.")
        df = None
    else:
        print("Found zero std! {}".format(zero_list))
        df_index = np.zeros(shape=(num_minority, len(zero_list)))
        df = pd.DataFrame(df_index, columns=zero_list)

        for i in range(len(zero_list)): 
            pos.drop(zero_list[i], axis=1, inplace=True)
            df[zero_list[i]] = zero_mean[i]
    #print(pos)
    #print(df)
    return pos, df

# Find no correlation and univariate sampling
def no_corr(pos, num_minority):
    for i in tqdm(range(100), desc="Searching no correlation", leave=False):
        corr = abs(pos.corr())
        nocorr_df = pd.DataFrame(index=[], columns=[]) 
        mean_list = []
        var_list = []
        col_list = []

        # split no corr attribute and calc mean and var.
        for i in range(len(corr.columns)):
            sort = corr.iloc[:, [i]]
            sort = sort.sort_values(by=sort.columns[0], ascending=False) #sort
            
            if sort.values[1] < 0.2:
                #print(sort.values[1])
                mean_list.append(pos[pos.columns[i]].mean())
                var_list.append(pos[pos.columns[i]].var())
                col_list.append(pos.columns[i])
    
    if (len(mean_list)==0) and (len(var_list)==0) and (len(col_list)==0): 
        print("Not found no correlation.")
        df = None
    else:
        print("Found no corr! {}".format(col_list))
        # univariate normal dist over-sampling
        tmp = []
        np.random.seed(seed=6)
        for mean, var in zip(mean_list, var_list):
            uni_x = np.random.normal(mean, var, num_minority)
            tmp.append(uni_x)
    
        # convert to dataframe
        df = pd.DataFrame(tmp).T
        df.columns = col_list
        # drop no correlation attributes
        for  i in range(len(col_list)):
            pos.drop(col_list[i], axis=1, inplace=True)

    #print(mean_list, var_list)
    #print(df)
    #print(pos)
    return pos, df

# Multivariate sampling
def mnd_os(pos, num_minority):
    for i in tqdm(range(100), desc="Multi normal dist over-sampling", leave=False):
        # calc correlation and covert absolute value
        corr = abs(pos.corr())
        #print(corr)
        
        # find strong correlation attribute
        corr_col = []
        corr_ind = []
        for i in range(len(corr.columns)):
            sort = corr.iloc[:, [i]] #extract one index
            sort = sort.sort_values(by=sort.columns[0], ascending=False) #sort
            #print(sort)
            corr_col.append(sort.columns[0]) #strong corr coulumns
            corr_ind.append(sort.index[1]) #strong corr index
        #print(corr_col)
        #print(corr_ind)
        
        # calc mean and covariance
        mean_list = []
        cov_list = []
        for i in range(len(pos.columns)):
            mean_list.append([pos[corr_col[i]].mean(), pos[corr_ind[i]].mean()])
            cov_list.append(pd.concat([pos[corr_col[i]], pos[corr_ind[i]]], axis=1).cov())
            
        # generate new sample
        tmp = []
        np.random.seed(seed=6)
        for mean, cov in zip(mean_list, cov_list):
            mul_x, mul_y = np.random.multivariate_normal(mean, cov, num_minority).T
            tmp.append(mul_x)
        
        # convert to dataframe
        df = pd.DataFrame(tmp).T
        df.columns = pos.columns
        #print(df)
    return df

def append_data(pos, zero_std, no_corr):
    pos = pd.concat([pos, zero_std, no_corr], axis=1)
    pos['Label'] = 1
    #print(pos)
    #pos.to_csv('/home/yura/Desktop/mlpd_train.csv', index=False)
    return pos

