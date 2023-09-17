import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time

from sklearn.preprocessing import StandardScaler

import src.own_data as data
import jitkrittum.kernel as kernel
import jitkrittum.tst as tst

OUT_DIR = os.path.join('experiments','jitkrittum','output')

# significance level of the test
alpha= 0.01
sample_size_list = 100*np.arange(2,11,2)
n_trials = 500
d = 50

seed = 47

cols = ['alpha','pvalue','pvalue_std','test_stat','h0_rejected','d','sample_size','type_I_error','type_II_error','runtime','runtime_std']
out_df_baseline = pd.DataFrame(np.zeros((len(sample_size_list),len(cols))),columns=cols)

print('Experiment: Same gauss vary sample size, Jitkrittum')

for i,sample_size in enumerate(sample_size_list):
    results_df_baseline = pd.DataFrame(np.zeros((n_trials,6)),columns=['pvalue','test_stat','h0_rejected','type_I_error','type_II_error','runtime'])

    for j in tqdm(range(n_trials)):
        # Set a new seed every time
        seed += 1
        np.random.seed(seed=seed)

        # Generate the dataset
        X = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=sample_size)
        Y = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=sample_size)
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

        met_opt_options = {'n_test_locs': 1,
                        #    'max_iter': 200,
                        #    'locs_step_size': 500.0,
                        #    'gwidth_step_size': 0.2,
                           'seed': seed,
                           'tol_fun': 1e-4}

        test_locs, gwidth,_ = tst.MeanEmbeddingTest.optimize_locs_width(tr, alpha, **met_opt_options)
        v = np.abs(test_locs[0])
        if np.argmax(v) != 0:
            results_df_baseline.loc[j,'type_I_error'] = 1
            results_df_baseline.loc[j,'type_II_error'] = 1
        else:
            results_df_baseline.loc[j,'type_I_error'] = 0
            results_df_baseline.loc[j,'type_II_error'] = 0

        tic = time()
        baseline = tst.MeanEmbeddingTest(test_locs, gwidth,alpha=0.01).perform_test(te)
        results_df_baseline.loc[j,'runtime'] = time()-tic
        
        # Save baseline results
        results_df_baseline.loc[j,'pvalue'] = baseline['pvalue']
        results_df_baseline.loc[j,'test_stat'] = baseline['test_stat']
        results_df_baseline.loc[j,'h0_rejected'] = baseline['h0_rejected'].astype(int)

    
    out_df_baseline.loc[i,'alpha'] = alpha
    out_df_baseline.loc[i,'d'] = d
    out_df_baseline.loc[i,'sample_size'] = sample_size
    out_df_baseline.loc[i,['pvalue','test_stat','h0_rejected','type_I_error','type_II_error']] =  results_df_baseline.mean(axis=0)
    out_df_baseline.loc[i,'pvalue_std'] = results_df_baseline['pvalue'].std(axis=0)
    out_df_baseline.loc[i,'runtime'] = results_df_baseline['runtime'].mean(axis=0)
    out_df_baseline.loc[i,'runtime_std'] = results_df_baseline['runtime'].std(axis=0)


out_df_baseline.to_csv(os.path.join(OUT_DIR,'same_gaussian_vary_sample_size.csv'),index=False)
