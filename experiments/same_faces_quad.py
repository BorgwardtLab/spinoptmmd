import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
import argparse

import src.own_data as data
import src.kernel_torch as kernel
import src.own_tst as tst
from src.utils import load_karolinska

OUT_DIR = os.path.join('experiments','output')

parser = argparse.ArgumentParser()
parser.add_argument('--lam', default=0.1, type=float, help = 'the regularisation parameter lambda.')
args = parser.parse_args()

# significance level of the test
alpha= 0.01
n_trials = 500
lam = args.lam
seed = 47

cols = ['alpha','pvalue','pvalue_std','test_stat','h0_rejected','lambda','d','sample_size','runtime','runtime_std','h0_rejected_pruned']
cols_baseline = [c for c in cols if c not in ['lambda']]
out_df = pd.DataFrame(np.zeros((1,len(cols))),columns=cols)
out_df_baseline = pd.DataFrame(np.zeros((1,len(cols_baseline))),columns=cols_baseline)

file_path = os.path.join('data','gretton')
fname = os.path.join(file_path,'crop48_h0_repickled.p')
data_arr = load_karolinska(fname,label='H0')

x_orig = 34
y_orig = 48
n_pixels = x_orig*y_orig

results_df = pd.DataFrame(np.zeros((n_trials,5)),columns=['pvalue','test_stat','h0_rejected','runtime','h0_rejected_pruned'])
results_df_baseline = pd.DataFrame(np.zeros((n_trials,5)),columns=['pvalue','test_stat','h0_rejected','runtime','h0_rejected_pruned'])

contrast_images = []

print('Experiment: Same faces, quadratic')

for j in tqdm(range(n_trials),disable=False):

    # Set a new seed every time
    seed += 1
    np.random.seed(seed=seed)

    np.random.shuffle(data_arr)
    idx = int(data_arr.shape[0]/2)
    tst_data = data.TSTData(data_arr[:idx], data_arr[idx:])

    # Split the features
    tr, te = tst_data.split_tr_te(tr_proportion=0.5, seed=seed)

    # No need to scale the features as they are all on the same scale already

    # Define kernels
    median_dist = tst.QuadMMDTest.median_distance(tr.X, tr.Y)
    k_list = [kernel.KGauss(sigma2=s) for s in median_dist*2**np.linspace(-4,4,30)]

    # Perform test
    k_idx,_ = tst.QuadMMDTest.grid_search_kernel(tr.X,tr.Y,k_list,alpha,func_var=tst.QuadMMDTest.compute_var_biased)
    k_ours = k_list[k_idx]
    tic = time()
    baseline = tst.QuadMMDTest(kernel=k_ours,alpha=alpha).perform_test(te,w=None)
    results_df_baseline.loc[j,'runtime'] = time()-tic  

    # Masked test
    op = {
        'max_iter': 1000, # maximum number of gradient ascent iterations
        'lam': lam*n_pixels, # LASSO regularization strength
        'func_var':tst.QuadMMDTest.compute_var_biased, # Use the unbiased variance estimator
        'lam_var':1e-8, #regularise lambda just in case
        'ftol': 1e-4, # stop if the objective does not increase more than this.
        'kernel':k_ours, # the kernel to use
        'power_stat':True,
        'power_stat_full':False,
        'verbose':False
    }

    # optimize on the training set
    info = tst.QuadMMDTest.optimize_w(tr, alpha, **op)

    # Calculate important features
    feature_ids = tst.QuadMMDTest.select_features(tr,info['weights'],alpha,
                                                        kernel=k_ours, 
                                                        pruning=True,
                                                        feature_percentage=0.1)

    contrast_image = np.zeros(n_pixels)
    if len(feature_ids)>0:
        contrast_image[feature_ids] = 1
        median_dist_pruned = tst.QuadMMDTest.median_distance(tr.X*contrast_image, tr.Y*contrast_image)
        k_list_pruned = [kernel.KGauss(sigma2=s) for s in median_dist_pruned*2**np.linspace(-4,4,30)]
        k_idx_pruned,_ = tst.QuadMMDTest.grid_search_kernel(tr.X*contrast_image,tr.Y*contrast_image,k_list_pruned,alpha,func_var=tst.QuadMMDTest.compute_var_biased)
        k_ours_pruned = k_list_pruned[k_idx_pruned]
        pruned_test_baseline = tst.QuadMMDTest(kernel=k_ours_pruned,alpha=alpha).perform_test(te,w=contrast_image)
        masked_weights = np.zeros(n_pixels)
        masked_weights[feature_ids] = info['weights'][feature_ids]
        pruned_test = tst.QuadMMDTest(kernel=k_ours_pruned,alpha=alpha).perform_test(te,w=masked_weights)
        results_df_baseline.loc[j,'h0_rejected_pruned'] = pruned_test_baseline['h0_rejected'].astype(int)
        results_df.loc[j,'h0_rejected_pruned'] = pruned_test['h0_rejected'].astype(int)
    else:
        results_df_baseline.loc[j,'h0_rejected_pruned'] = np.nan
        results_df.loc[j,'h0_rejected_pruned'] = np.nan
    contrast_images.append(contrast_image)
    
    tic = time()
    masked_test = tst.QuadMMDTest(kernel=k_ours,alpha=alpha).perform_test(te,w=info['weights'])
    results_df.loc[j,'runtime'] = time()-tic

    # Save baseline results
    results_df_baseline.loc[j,'pvalue'] = baseline['pvalue']
    results_df_baseline.loc[j,'test_stat'] = baseline['test_stat']
    results_df_baseline.loc[j,'h0_rejected'] = baseline['h0_rejected'].astype(int)


    # Save masked results
    results_df.loc[j,'pvalue'] = masked_test['pvalue']
    results_df.loc[j,'test_stat'] = masked_test['test_stat']
    results_df.loc[j,'h0_rejected'] = masked_test['h0_rejected'].astype(int)

mean_contrast_image = np.array(contrast_images).mean(axis=0)
np.savetxt(os.path.join(OUT_DIR,f'mean_contrast_image_same_faces_quad_lam_{lam}.txt'),mean_contrast_image)

out_df_baseline.loc[0,'alpha'] = alpha
out_df_baseline.loc[0,'d'] = n_pixels
out_df_baseline.loc[0,'sample_size'] = 2*tst_data.X.shape[0]
out_df_baseline.loc[0,['pvalue','test_stat','h0_rejected','runtime','h0_rejected_pruned']] =  results_df_baseline.mean(axis=0)
out_df_baseline.loc[0,'pvalue_std'] = results_df_baseline['pvalue'].std(axis=0)
out_df_baseline.loc[0,'runtime_std'] = results_df_baseline['runtime'].std(axis=0)

out_df.loc[0,'alpha'] = alpha
out_df.loc[0,'d'] = n_pixels
out_df.loc[0,'sample_size'] = 2*tst_data.X.shape[0]
out_df.loc[0,'lambda'] = op['lam']
out_df.loc[0,['pvalue','test_stat','h0_rejected','runtime','h0_rejected_pruned']] = results_df.mean(axis=0)
out_df.loc[0,'pvalue_std'] = results_df['pvalue'].std(axis=0)
out_df.loc[0,'runtime_std'] = results_df['runtime'].std(axis=0)

out_df_baseline.to_csv(os.path.join(OUT_DIR,f'same_faces_quad_baseline_lam_{lam}.csv'),index=False)
out_df.to_csv(os.path.join(OUT_DIR,f'same_faces_quad_lam_{lam}.csv'),index=False)