# coding: utf-8

# In[1]:
import sys
import os.path
import numpy as np
import pandas as pd
import multivariate_os
import matplotlib.pyplot as plt
from collections import Counter
from io import StringIO



# In[2]:
data = pd.read_csv(sys.argv[1])

X = data.drop('Label', axis=1)
y = data.Label
pos = data[data.Label == 1]
pos = pos.drop('Label', axis=1)


# In[3]:
RANDOM_STATE = 6
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=RANDOM_STATE)
cnt = Counter(y_train)
num_minority = int((cnt[0] - cnt[1]))
print('y_train: {}'.format(Counter(y_train)))
print('y_test: {}'.format(Counter(y_test)))
print(num_minority)

# In[4]:

#pos, no_corr = multivariate_os.no_corr(pos, num_minority)
#pos = multivariate_os.mnd_os(pos, num_minority)
#df = multivariate_os.append_data(pos, zero_std, no_corr)
