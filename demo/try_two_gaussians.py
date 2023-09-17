import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time

from sklearn.preprocessing import StandardScaler

import src.data as data
import src.kernel as kernel
import src.tst as tst


# significance level of the test
alpha= 0.01
seed = 47
np.random.seed(seed=seed)

# Mean shift
X = np.random.multivariate_normal([1,0], [[1,0],[0,1]], size=50)
Y = np.random.multivariate_normal([0,0], [[1,0],[0,1]], size=50)

tst_data = data.TSTData(X, Y)

# Split here as we need the train set for the null.
tr, te = tst_data.split_tr_te(tr_proportion=0.5, seed=seed)

# scale the features
scaler = StandardScaler()
len_X_tr = tr.X.shape[0]

xy = scaler.fit_transform(tr.stack_xy())
tr.X = xy[:len_X_tr]
tr.Y = xy[len_X_tr:]

te.X = scaler.transform(te.X)
te.Y = scaler.transform(te.Y)

k = kernel.KGauss(sigma2=tst.QuadMMDTest.median_distance(tr.X, tr.Y))

# Masked test params
op = {
    'max_iter': 1000, # maximum number of gradient ascent iterations
    'lam': 1e-1, # LASSO regularization strength
    'func_var':tst.QuadMMDTest.compute_var_biased, # Use the biased variance estimator
    'ftol': 1e-4, # stop if the objective does not increase more than this.
    'kernel':k, # the kernel to use
    'power_stat':True,
    'verbose':False
}

# optimize on the training set
info = tst.QuadMMDTest.optimize_w(tr, alpha, **op)

print('Optimisation results:')
print(info)

# Perform test
masked_test = tst.QuadMMDTest(kernel=k,alpha=0.01).perform_test(te,w=info['weights'])

print('Masked MMD results:')
print(masked_test)