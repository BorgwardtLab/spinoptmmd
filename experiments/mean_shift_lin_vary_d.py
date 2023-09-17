import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time

from sklearn.preprocessing import StandardScaler

import src.own_data as data
import src.kernel_torch as kernel
import src.own_tst as tst

OUT_DIR = os.path.join('experiments','output')

# significance level of the test
alpha= 0.01
sample_size = 1000
n_trials = 500
d_list = [5,300,600,900,1200,1500]
seed = 47

cols = ['alpha','pvalue','pvalue_std','test_stat','h0_rejected','lambda','d','sample_size','type_I_error','type_II_error','runtime','runtime_std','h0_rejected_pruned']
cols_baseline = [c for c in cols if c not in ['lambda','type_I_error','type_II_error']]
out_df = pd.DataFrame(np.zeros((len(d_list),len(cols))),columns=cols)
out_df_baseline = pd.DataFrame(np.zeros((len(d_list),len(cols_baseline))),columns=cols_baseline)

print('Experiment: Mean shift vary d, linear')

for i,d in enumerate(d_list):
    results_df = pd.DataFrame(np.zeros((n_trials,7)),columns=['pvalue','test_stat','h0_rejected','type_I_error','type_II_error','runtime','h0_rejected_pruned'])
    results_df_baseline = pd.DataFrame(np.zeros((n_trials,5)),columns=['pvalue','test_stat','h0_rejected','runtime','h0_rejected_pruned'])

    for j in tqdm(range(n_trials)):
        # Set a new seed every time
        seed += 1
        np.random.seed(seed=seed)

        # Generate the dataset
        mean_1 = np.zeros(d)
        mean_1[0] = 1
        X = np.random.multivariate_normal(mean_1, np.eye(d), size=sample_size)
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

        # Define kernels
        median_dist = tst.LinearMMDTest.median_distance(tr.X, tr.Y)
        k_list = [kernel.KGauss(sigma2=s) for s in median_dist*2**np.linspace(-4,4,30)]
            
        # Perform test
        k_idx,_ = tst.LinearMMDTest.grid_search_kernel(tr.X,tr.Y,k_list,alpha)
        k_ours = k_list[k_idx]
        tic = time()
        baseline = tst.LinearMMDTest(kernel=k_ours,alpha=alpha).perform_test(te,w=None)
        results_df_baseline.loc[j,'runtime'] = time()-tic

        # Masked test params
        op = {
            'max_iter': 1000, # maximum number of gradient ascent iterations
            'lam': 1e-1*d, # LASSO regularization strength
            'ftol': 1e-4, # stop if the objective does not increase more than this.
            'kernel':k_ours, # the kernel to use
            'power_stat':True,
            'verbose':False
        }

        # optimize on the training set
        info = tst.LinearMMDTest.optimize_w(tr, alpha, **op)

        # Calculate important features
        feature_ids = tst.LinearMMDTest.select_features(tr,info['weights'],alpha,
                                                            kernel=k_ours, 
                                                            pruning=True,
                                                            feature_percentage=0.1)
        if 0 in feature_ids:
            results_df.loc[j,'type_I_error'] = 0
        else:
            results_df.loc[j,'type_I_error'] = 1
        results_df.loc[j,'type_II_error'] = len(feature_ids[feature_ids!=0])/(0.1*d)
        
        tic = time()
        masked_test = tst.LinearMMDTest(kernel=k_ours,alpha=alpha).perform_test(te,w=info['weights'])
        results_df.loc[j,'runtime'] = time()-tic
        
        # Save baseline results
        results_df_baseline.loc[j,'pvalue'] = baseline['pvalue']
        results_df_baseline.loc[j,'test_stat'] = baseline['test_stat']
        results_df_baseline.loc[j,'h0_rejected'] = baseline['h0_rejected'].astype(int)


        # Save masked results
        results_df.loc[j,'pvalue'] = masked_test['pvalue']
        results_df.loc[j,'test_stat'] = masked_test['test_stat']
        results_df.loc[j,'h0_rejected'] = masked_test['h0_rejected'].astype(int)

    
    out_df_baseline.loc[i,'alpha'] = alpha
    out_df_baseline.loc[i,'d'] = d
    out_df_baseline.loc[i,'sample_size'] = sample_size
    out_df_baseline.loc[i,['pvalue','test_stat','h0_rejected']] =  results_df_baseline.mean(axis=0)
    out_df_baseline.loc[i,'pvalue_std'] = results_df_baseline['pvalue'].std(axis=0)
    out_df_baseline.loc[i,'runtime'] = results_df_baseline['runtime'].mean(axis=0)
    out_df_baseline.loc[i,'runtime_std'] = results_df_baseline['runtime'].std(axis=0)

    out_df.loc[i,'alpha'] = alpha
    out_df.loc[i,'d'] = d
    out_df.loc[i,'sample_size'] = sample_size
    out_df.loc[i,'lambda'] = op['lam']
    out_df.loc[i,['pvalue','test_stat','h0_rejected','type_I_error','type_II_error']] = results_df.mean(axis=0)
    out_df.loc[i,'pvalue_std'] = results_df['pvalue'].std(axis=0)
    out_df.loc[i,'runtime'] = results_df_baseline['runtime'].mean(axis=0)
    out_df.loc[i,'runtime_std'] = results_df_baseline['runtime'].std(axis=0)

out_df_baseline.to_csv(os.path.join(OUT_DIR,'mean_shift_lin_vary_d_baseline.csv'),index=False)
out_df.to_csv(os.path.join(OUT_DIR,'mean_shift_lin_vary_d.csv'),index=False)