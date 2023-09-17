import os
import pickle
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time

import src.own_data as data
import jitkrittum.kernel as kernel
import jitkrittum.tst as tst
from src.utils import load_scanpy

from sklearn.preprocessing import StandardScaler,MinMaxScaler

OUT_DIR = os.path.join('experiments','jitkrittum','output')

# significance level of the test
alpha= 0.01
n_trials = 500
seed = 47

file_path = os.path.join('data','scanpy')

print('Experiment: Scanpy, Jitkrittum')

for label in ['H0']+list(range(8)):
    cols = ['alpha','pvalue','pvalue_std','test_stat','h0_rejected','d','sample_size','runtime','runtime_std']
    out_df_baseline = pd.DataFrame(np.zeros((1,len(cols))),columns=cols)

    X,Y,names = load_scanpy(file_path,label=label)

    results_df_baseline = pd.DataFrame(np.zeros((n_trials,4)),columns=['pvalue','test_stat','h0_rejected','runtime'])

    contrast_names_list = []

    for j in tqdm(range(n_trials),disable=False):

        # Set a new seed every time
        seed += 1
        np.random.seed(seed=seed)

        # Subsample to get an equal number of samples
        if X.shape[0]>Y.shape[0]:
            subsample = np.random.choice(range(X.shape[0]),size=(Y.shape[0],),replace=False)
            X_sampled = X[subsample]
            Y_sampled = Y
        elif Y.shape[0]>X.shape[0]:
            subsample = np.random.choice(range(Y.shape[0]),size=(X.shape[0],),replace=False)
            X_sampled = X
            Y_sampled = Y[subsample]
        else:
            X_sampled = X
            Y_sampled = Y

        tst_data = data.TSTData(X_sampled,Y_sampled)

        # Split here as we need the train set for the null.
        tr, te = tst_data.split_tr_te(tr_proportion=0.5, seed=seed)

        # Scale the features
        scaler = MinMaxScaler()
        scaler.fit(tr.stack_xy())
        tr.X = scaler.transform(tr.X)
        tr.Y = scaler.transform(tr.Y)

        te.X = scaler.transform(te.X)
        te.Y = scaler.transform(te.Y)

 
        met_opt_options = {'n_test_locs': 1,
                        #    'max_iter': 200,
                        #    'locs_step_size': 500.0,
                        #    'gwidth_step_size': 0.2,
                           'seed': seed,
                           'tol_fun': 1e-4}

        test_locs, gwidth,_ = tst.MeanEmbeddingTest.optimize_locs_width(tr, alpha, **met_opt_options)
        v = np.abs(test_locs[0])
        contrast_names_list.append((v > np.quantile(v,0.9)).astype(int))

        tic = time()
        baseline = tst.MeanEmbeddingTest(test_locs, gwidth,alpha=0.01).perform_test(te)
        results_df_baseline.loc[j,'runtime'] = time()-tic

        # Save baseline results
        results_df_baseline.loc[j,'pvalue'] = baseline['pvalue']
        results_df_baseline.loc[j,'test_stat'] = baseline['test_stat']
        results_df_baseline.loc[j,'h0_rejected'] = baseline['h0_rejected'].astype(int)
    
    mean_contrast_names = np.nanmean(np.array(contrast_names_list),axis=0)
    np.savetxt(os.path.join(OUT_DIR,f'mean_contrast_names_scanpy_{label}.txt'),mean_contrast_names)

    out_df_baseline.loc[0,'alpha'] = alpha
    out_df_baseline.loc[0,'d'] = len(names)
    out_df_baseline.loc[0,'sample_size'] = 2*tst_data.X.shape[0]
    out_df_baseline.loc[0,['pvalue','test_stat','h0_rejected','runtime']] =  results_df_baseline.mean(axis=0)
    out_df_baseline.loc[0,'pvalue_std'] = results_df_baseline['pvalue'].std(axis=0)
    out_df_baseline.loc[0,'runtime_std'] = results_df_baseline['runtime'].std(axis=0)

    out_df_baseline.to_csv(os.path.join(OUT_DIR,f'scanpy_{label}.csv'),index=False)
