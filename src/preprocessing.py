from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

# Normalize
def normalization(os_list):
    for i in range(len(os_list)):
        os_list[i][0] = normalize(os_list[i][0], norm='l2') 
    return os_list  

# Standardization
def standardization(os_list):
    for i in range(len(os_list)):
        sc = StandardScaler()
        os_list[i][0] = sc.fit_transform(os_list[i][0])
    return os_list
