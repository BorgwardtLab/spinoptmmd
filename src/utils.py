import os
import pickle
import functools
from glob import glob,iglob
from math import factorial
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.optimize import minimize
import torch
from tqdm import tqdm


##### Helper functions ######
def m_r(m,r):
    return factorial(m)/factorial(m-r)

def tr_te_indices(n, tr_proportion, seed=9282 ):
    '''
        Get two logical vectors for indexing train/test points.
        Return (tr_ind, te_ind)
        From https://github.com/wittawatj/interpretable-test
    '''
    rand_state = np.random.get_state()
    np.random.seed(seed)

    Itr = np.zeros(n, dtype=bool)
    tr_ind = np.random.choice(n, int(tr_proportion*n), replace=False)
    Itr[tr_ind] = True
    Ite = np.logical_not(Itr)

    np.random.set_state(rand_state)
    return (Itr, Ite)

def subsample_ind(n, k, seed=28):
    '''
    Return a list of indices to choose k out of n without replacement
    From https://github.com/wittawatj/interpretable-test
    '''
    rand_state = np.random.get_state()
    np.random.seed(seed)

    ind = np.random.choice(n, k, replace=False)
    np.random.set_state(rand_state)
    return ind

def multiple_testing_correction(p_values,alpha=0.01,method='Benjamini-Hochberg'):
    '''
    If required. Not needed in the current algorithms.
    '''
    if method=='Bonferroni':
        return np.where(p_values*len(p_values) <= alpha)[0]
    elif method == 'Benjamini-Hochberg':
        ranks = rankdata(p_values)
        correction_factor = (ranks/len(ranks))*alpha
        return np.where(p_values <= correction_factor)[0]
    else:
        raise NotImplementedError("Method not implemented.")

def prune_features(w,feature_percentage):
    cutoff = np.percentile(w.numpy(),100*(1-feature_percentage))
    return np.where(w>cutoff)[0]

def getRadiomicFeatures(data,pats_resp,times_uni = [0,60,120]):

    # Get radiomic pretreatment features for pats
    rad_feat = data.iloc[:, list(data.columns).index('ORIGINAL_SHAPE_ELONGATION_WHOLE'):list(data.columns).index(
        'SERIESDESCRIPTION_CYSTIC')].drop([
        'SERIESDESCRIPTION_CORE', 'ANNOTATION_CORE'], axis=1)
    rad_feat['pnoc_case'] = data['pnoc_case'].astype(str)
    rad_feat['time_days'] = data['time_days']
    rad_feat_pre = rad_feat[rad_feat['time_days']==0]
    inds_2 = [i +1 for i in list(np.where(rad_feat['time_days']==0)[0])]
    rad_feat_t2 = rad_feat.iloc[inds_2,:]


    # Now get pats with labels
    rad_feat_pre.set_index('pnoc_case', inplace = True)
    rad_feat_pre = rad_feat_pre.loc[pats_resp]
    rad_feat_t2.set_index('pnoc_case', inplace = True)
    rad_feat_t2 = rad_feat_t2.loc[pats_resp]


    # Check features
    feats_cat = []
    all_cats = []
    feats_orig = []
    for f in rad_feat_pre.columns:
        this_feat_cat = f.split('_')[0]
        all_cats.append(this_feat_cat)
        if this_feat_cat not in feats_cat:
            if this_feat_cat not in ['time','responder']:
                feats_cat.append(this_feat_cat)

        if this_feat_cat == 'ORIGINAL':
            feats_orig.append(f)
    feats_keep = list(rad_feat_pre.columns[rad_feat_pre.isna().sum()<8])

    # For now keep only "Original" features
    rad_feat_pre = rad_feat_pre[feats_keep]
    rad_feat_t2 = rad_feat_t2[feats_keep]


    # Only 31 (7 positive) pats with label and radiomic features - if thresh = 365
    # thresh = 90: 15 positive, 16 negative
    rad_feat_pre = rad_feat_pre.dropna()
    rad_feat_pre.drop('time_days', axis = 1, inplace = True)


    time_t2 = rad_feat_t2['time_days']
    rad_feat_t2.drop('time_days', axis = 1, inplace = True)
    rad_feat_t2.columns = [c+'_t2' for c in rad_feat_t2.columns]
    rad_feat_t2 = rad_feat_t2.loc[rad_feat_pre.index]
    time_t2 = time_t2.loc[rad_feat_t2.index]



    inds_3 = [i +2 for i in list(np.where(rad_feat['time_days']==0)[0])]
    rad_feat_t3 = rad_feat.iloc[inds_3,:]
    rad_feat_t3.set_index('pnoc_case', inplace = True)
    rad_feat_t3 = rad_feat_t3.loc[rad_feat_t2.index]

    # Check for dublicate patients
    rad_feat_t3.index.duplicated(keep='last')
    rad_feat_t3 = rad_feat_t3[~rad_feat_t3.index.duplicated(keep='last')]
    rad_feat_t3 = rad_feat_t3[feats_keep]
    time_t3 = rad_feat_t3['time_days']
    rad_feat_t3.drop('time_days', axis = 1, inplace = True)
    rad_feat_t3.columns = [c+'_t3' for c in rad_feat_t3.columns]
    rad_feat_t3 = rad_feat_t3.loc[rad_feat_pre.index]

    inds_4 = [i +3 for i in list(np.where(rad_feat['time_days']==0)[0])]
    rad_feat_t4 = rad_feat.iloc[inds_4,:]
    rad_feat_t4.set_index('pnoc_case', inplace = True)
    rad_feat_t4 = rad_feat_t4.loc[rad_feat_t2.index]
    rad_feat_t4.index.duplicated(keep='last')
    rad_feat_t4 = rad_feat_t4[~rad_feat_t4.index.duplicated(keep='last')]
    rad_feat_t4 = rad_feat_t4[feats_keep]
    time_t4 = rad_feat_t4['time_days']
    rad_feat_t4.drop('time_days', axis = 1, inplace = True)
    rad_feat_t4.columns = [c+'_t4' for c in rad_feat_t4.columns]
    rad_feat_t4 = rad_feat_t4.loc[rad_feat_pre.index]

    # Get the times
    time_0 = pd.Series(index = time_t2.index, data = np.zeros((len(time_t2.index),)))
    times = pd.concat([time_0,time_t2,time_t3,time_t4],axis=1)

    # Want uniform time grid - interpolate
    rad_feat_all = pd.concat([rad_feat_pre,rad_feat_t2,rad_feat_t3,rad_feat_t4],axis = 1)


    # Loop over all patients
    rad_feat_all_uni = pd.DataFrame(index = rad_feat_all.index)
    for p in tqdm(times.index):

        # for each patient interpolate the radiomic features
        for c in rad_feat_pre.columns:
            this_cols = [c+k for k in ['','_t2','_t3','_t4']]

            #loop over time points
            for t in times_uni:
                time_interp = times.loc[p].values
                val_interp  = rad_feat_all.loc[p,this_cols].values
                inds_keep = ~np.isnan(val_interp)

                rad_feat_all_uni.loc[p,c+'_'+str(t)] =\
                    np.interp(t,time_interp[inds_keep],val_interp[inds_keep])


    rad_feat_t2uni = rad_feat_all_uni.loc[:,]

    # rad_feat_ratio = pd.DataFrame(index = rad_feat_t2.index, data = rad_feat_t2uni.values/rad_feat_pre.values,
    #                               columns = [c+'_ratio' for c in rad_feat_pre.columns])
    # rad_feat_ratio.fillna(0, inplace=True)
    # inf_cols = rad_feat_ratio.columns[rad_feat_ratio.isin([np.inf, -np.inf]).sum()>0]
    # for c in inf_cols:
    #     this_max =rad_feat_ratio.loc[rad_feat_ratio[c] != np.inf, c].max()
    #     rad_feat_ratio[c].replace(np.inf, this_max, inplace=True)
    #
    # rad_feat_diff = pd.DataFrame(index = rad_feat_t2.index, data = rad_feat_t2.values-rad_feat_pre.values,
    #                               columns = [c+'_dif' for c in rad_feat_pre.columns])
    # rad_feat = pd.concat([rad_feat_pre,rad_feat_t2,rad_feat_ratio,rad_feat_diff],axis = 1)
    return rad_feat_all_uni.copy()

def getUniquePats(df_res_sub,thresh):
    pats_resp = np.unique([np.int(p[:2]) for p in df_res_sub.index])
    ids_keep = []
    resp = []
    for p in pats_resp:
        try:
            resp.append(df_res_sub.loc[str(p)+'_LATE','responder'+str(thresh)])
            ids_keep.append(str(p)+'_LATE')
        except:
            resp.append(df_res_sub.loc[str(p) + '_EARLY', 'responder'+str(thresh)])
            ids_keep.append(str(p) + '_EARLY')
    pats_resp  = [str(p) for p in pats_resp]

    return pats_resp,ids_keep,resp

##### Data loaders ######
def load_lgg_radiomic(file_path,label='ATRX'):
    # Load data
    data = pd.read_csv(os.path.join(file_path,'radiomic_feat.csv'),index_col=0)
    data = data.dropna(axis=1)
    data[data=='#DIV/0!'] = np.nan
    data = data.dropna(axis=0).astype(float)

    # Now do everything else
    if label=='H0':
        XY = data.values
        np.random.shuffle(XY)
        X = XY[:int(np.floor(XY.shape[0]/2))]
        Y = XY[int(np.ceil(XY.shape[0]/2)):]
    else:
        labels = pd.read_csv(os.path.join(file_path,'labels_mut.csv'),index_col=0)
        overlap = list(set(data.index).intersection(set(labels.index)))
        labels_subset = labels.loc[overlap]
        X_idx = labels_subset[labels_subset[label]==1].index
        Y_idx = labels_subset[labels_subset[label]==0].index
        X = data.loc[X_idx].values
        Y = data.loc[Y_idx].values
    feature_names = data.columns.values
    return X,Y,feature_names

def load_lgg_expression(file_path,label='ATRX'):
    # Load data
    data = pd.read_csv(os.path.join(file_path,'expr.csv'),index_col=0)
    data = data.dropna(axis=1)
    data[data=='#DIV/0!'] = np.nan
    data = data.dropna(axis=0).astype(float)

    # Log transform the counts data as usual
    tmp = np.log(data.values+1)
    # Every sample should have mean 0 and std 1 over the genes
    tmp = (tmp-tmp.mean(axis=1,keepdims=True))/tmp.std(axis=1,keepdims=True)
    data = pd.DataFrame(data=tmp,index=data.index,columns=data.columns)

    # Now do everything else
    if label=='H0':
        XY = data.values
        np.random.shuffle(XY)
        X = XY[:int(np.floor(XY.shape[0]/2))]
        Y = XY[int(np.ceil(XY.shape[0]/2)):]
    else:
        labels = pd.read_csv(os.path.join(file_path,'labels_mut.csv'),index_col=0)
        overlap = list(set(data.index).intersection(set(labels.index)))
        labels_subset = labels.loc[overlap]
        X_idx = labels_subset[labels_subset[label]==1].index
        Y_idx = labels_subset[labels_subset[label]==0].index
        X = data.loc[X_idx].values
        Y = data.loc[Y_idx].values
    feature_names = data.columns.values
    return X,Y,feature_names

def load_karolinska(fname):
    with open(fname, 'rb') as f:
        loaded = pickle.load(f)
    return loaded

def load_nips(file_path):
    with open(file_path, 'rb') as f:
        loaded = pickle.load(f)
    X = loaded['P']
    Y = loaded['Q']
    words = loaded['words']
    n_min = min(X.shape[0], Y.shape[0])
    X = X[:n_min, :]
    Y = Y[:n_min, :]
    assert(X.shape[0] == Y.shape[0])
    return X,Y,words

def load_adni_radiomic(file_path,label='test'):
    data = pd.read_csv(os.path.join(file_path,'3DradiomicFeatures_HC_lr.csv'),index_col=0)
    labels = pd.read_csv(os.path.join(file_path,'labels.csv'),index_col=0)
    data['label'] = labels.label
    data_subset = data[~data.label.isna()]
    if label == 'H0':
        XY = data_subset.drop('label',axis=1).values
        np.random.shuffle(XY)
        X = XY[:int(np.floor(XY.shape[0]/2))]
        Y = XY[int(np.ceil(XY.shape[0]/2)):]
    else:
        X = data_subset[data_subset.label==0].drop('label',axis=1).values
        Y = data_subset[data_subset.label==1].drop('label',axis=1).values
    feature_names = data.drop('label',axis=1).columns.values
    return X,Y,feature_names

def load_adni_images(file_path,view='axial',label='test'):
    data_files = glob(os.path.join(file_path,'2Dimages',view,'*.npy'))
    labels = pd.read_csv(os.path.join(file_path,'labels.csv'),index_col=0)
    data_files_filtered = [f for f in data_files if f.split('/')[-1].split('.')[0] in labels.index]
    data = pd.DataFrame({f.split('/')[-1].split('.')[0]:np.load(f,allow_pickle=True).flatten() for f in data_files_filtered}).T
    data_shape = np.load(data_files_filtered[0],allow_pickle=True).shape
    data['label'] = labels.label
    data_subset = data[~data.label.isna()]
    if label == 'H0':
        XY = data_subset.drop('label',axis=1).values
        np.random.shuffle(XY)
        X = XY[:int(np.floor(XY.shape[0]/2))]
        Y = XY[int(np.ceil(XY.shape[0]/2)):]
    else:
        X = data_subset[data_subset.label==0].drop('label',axis=1).values
        Y = data_subset[data_subset.label==1].drop('label',axis=1).values
    return X,Y,data_shape

def pickle_driams(file_path,species='Escherichia coli',drug='Ceftriaxone'):
    if not os.path.exists(os.path.join(file_path,f'driams_{species}_{drug}.pkl')):
        labels = pd.read_csv(os.path.join(file_path,'2018_strat.csv'),index_col=1)
        labels_species = labels[labels.species==species]
        final_labels = labels_species[labels_species[drug].isin(['S','R'])]
        all_files = glob(os.path.join(file_path,'2018','*.txt'))
        relevant_files = [f for f in all_files if f.split('/')[-1].split('.')[0] in final_labels.index.values]
        data = pd.DataFrame(columns=np.arange(6000))
        for f in relevant_files:
            data.loc[f.split('/')[-1].split('.')[0]] = pd.read_csv(f,comment='#',sep=' ',dtype=np.float64,low_memory=False).binned_intensity.values
        data['label'] = final_labels[drug]
        with open(os.path.join(file_path,f'driams_{species}_{drug}.pkl'),'wb') as f:
            pickle.dump(data, f)

def load_driams(file_path,species='Escherichia coli',drug='Ceftriaxone',label='test'):
    with open(os.path.join(file_path,f'driams_{species}_{drug}.pkl'),'rb') as f:
        data = pickle.load(f)
    if label == 'H0':
        XY = data.drop('label',axis=1).values
        np.random.shuffle(XY)
        X = XY[:int(np.floor(XY.shape[0]/2))]
        Y = XY[int(np.ceil(XY.shape[0]/2)):]
    else:
        X = data[data.label=='S'].drop('label',axis=1).values
        Y = data[data.label=='R'].drop('label',axis=1).values
    return X,Y

def load_scanpy(file_path,label='H0'):
    data = pd.read_csv(os.path.join(file_path,'pbmc3k_processed.csv'),index_col=0)
    if label == 'H0':
        XY = data.drop('label',axis=1).values
        np.random.shuffle(XY)
        X = XY[:int(np.floor(XY.shape[0]/2))]
        Y = XY[int(np.ceil(XY.shape[0]/2)):]
    else:
        X = data[data.label==label].drop('label',axis=1).values
        Y = data[data.label!=label].drop('label',axis=1).values
    feature_names = data.drop('label',axis=1).columns.values
    return X,Y,feature_names

def load_pnoc001_radiomic(file_path, label='test',thresh = 90,times_uni = [0,60,120]):

    # load data
    data = pd.read_excel(file_path)
    df_res_sub = pd.read_csv('/Users/sbrueningk/Desktop/MMD/data/PNOC/fitres.csv', index_col=0)

    # Get response label
    df_res_sub['responder' + str(thresh)] = (df_res_sub['tc'] >= thresh).astype(int)

    # Get pats - prevent double instances
    pats_resp, ids_keep, resp = getUniquePats(df_res_sub, thresh)
    df_res_sub = df_res_sub.loc[ids_keep]
    df_res_sub['pnoc_case'] = pats_resp
    df_res_sub.set_index('pnoc_case', inplace=True)

    # get radiomic features
    df_radfeat = getRadiomicFeatures(data, pats_resp, times_uni=times_uni)
    df_res_sub = df_res_sub.loc[df_radfeat.index]

    # Prediction
    labels       = df_res_sub['responder' + str(thresh)]
    pats_nonresp = list(labels[labels==0].index)
    pats_resp    = list(labels[labels==1].index)


    data['label'] = labels.label
    data_subset = data[~data.label.isna()]
    if label == 'H0':
        XY = data_subset.drop('label',axis=1).values
        np.random.shuffle(XY)
        X = XY[:int(np.floor(XY.shape[0]/2))]
        Y = XY[int(np.ceil(XY.shape[0]/2)):]
    else:
        X = data_subset[data_subset.label==0].drop('label',axis=1).values
        Y = data_subset[data_subset.label==1].drop('label',axis=1).values
        X = df_radfeat.loc[pats_nonresp]
        Y = df_radfeat.loc[pats_resp]


    feature_names = X.columns.values


    return X, Y, feature_names

def load_pnoc001PI3K_radiomic(file_path, label='test',thresh = 90,times_uni = [0,60,120]):

    # load data
    data = pd.read_excel(file_path)
    df_res_sub = pd.read_csv('/Users/sbrueningk/Desktop/MMD/data/PNOC/fitres.csv', index_col=0)

    # Get response label
    df_res_sub['responder' + str(thresh)] = (df_res_sub['tc'] >= thresh).astype(int)

    # Get pats - prevent double instances
    pats_resp, ids_keep, resp = getUniquePats(df_res_sub, thresh)
    df_res_sub = df_res_sub.loc[ids_keep]
    df_res_sub['pnoc_case'] = pats_resp
    df_res_sub.set_index('pnoc_case', inplace=True)

    # get radiomic features
    df_radfeat = getRadiomicFeatures(data, pats_resp, times_uni=times_uni)
    df_res_sub = df_res_sub.loc[df_radfeat.index]

    # Prediction
    labels = df_res_sub['responder' + str(thresh)]
    pats_nonresp = list(labels[labels==0].index)
    pats_resp = list(labels[labels==1].index)
    X = df_radfeat.loc[pats_nonresp]
    Y = df_radfeat.loc[pats_resp]
    feature_names = X.columns.values


    return X, Y, feature_names


##### Optimization ######
#### SCIPY #####
def find_weights_sp(X,Y,func_z,max_iter=400,lam=1e-1,ftol=2.220446049250313e-09,kernel=None,verbose=True,**kwargs):
    def loss_fn(w,X,Y,func_z,lam,kernel):
        w = torch.from_numpy(w).requires_grad_()
        if kernel is None:
            loss = -func_z(w*X,w*Y) + lam*torch.abs(w).sum()
        else:
            loss = -func_z(w*X,w*Y,kernel) + lam*torch.abs(w).sum()
        loss.backward()
        return loss.detach().cpu().numpy(), w.grad.detach().cpu().numpy()

    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)

    # nx, d = X.shape
    # w0 = (1./d)*np.ones(d)
    w0 = np.ones(X.shape[1])

    # normalise lambda by number of features.
    # This way the chosen regularisation is independent of the number of features.
    lam_norm = lam/X.shape[1]

    loss_fn_partial = functools.partial(loss_fn,X=X,Y=Y,func_z=func_z,lam=lam_norm,kernel=kernel)

    res = minimize(loss_fn_partial, w0, method='L-BFGS-B', jac=True, options={'maxiter':max_iter,'iprint': 10 if verbose else -1,'ftol':ftol})

    # optimization info
    info = {'weights': np.abs(res['x']),'final_loss': res['fun'], 'success':res['success']}
    return info

#### Torch ####
def find_weights_torch(X,Y,func_z,max_iter=400,lam=1e-1,step_size=1e-2,batch_proportion=1.0,ftol=1e-3,kernel=None,verbose=True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    w = torch.ones(X.shape[1],requires_grad=True).to(device)
    X = torch.from_numpy(X).to(device)
    Y = torch.from_numpy(Y).to(device)

    # normalise lambda by number of features.
    # This way the chosen regularisation is independent of the number of features.
    lam_norm = lam/X.shape[0]

    if kernel is None:
        loss = lambda X,Y,w,lam: -func_z(w*X,w*Y) + lam*torch.abs(w).sum()
    else:
        loss = lambda X,Y,w,lam,kernel: -func_z(w*X,w*Y,kernel) + lam*torch.abs(w).sum()

    # //////// run gradient descent //////////////

    optimizer = torch.optim.SGD([w],lr=step_size,momentum=0.9,nesterov=True)
    # optimizer = torch.optim.Adam([w],lr=step_size)
    # optimizer = torch.optim.RMSprop([w],lr=step_size)
    # optimizer = torch.optim.LBFGS([w],lr=step_size)
    for t in range(max_iter):
        ind = np.random.choice(nx, min(int(batch_proportion*nx), nx), replace=False)
        if kernel is None:
            l = loss(X[ind, :],Y[ind, :],w,lam_norm)
        else:
            l = loss(X[ind, :],Y[ind, :],w,lam_norm,kernel)

        def closure(ind=ind):
            optimizer.zero_grad()
            # stochastic gradient descent
            if kernel is None:
                l = loss(X[ind, :],Y[ind, :],w,lam_norm)
            else:
                l = loss(X[ind, :],Y[ind, :],w,lam_norm,kernel)
            l.backward()
            return l
        optimizer.step(closure)
        
        # Project
        # w.data = w.data*(d/torch.sum(w.data))
        # w.data = torch.clamp(w.data,min=0.0) #torch.max(torch.stack([torch.zeros_like(w.data), w.data],axis=0),axis=0).values
        w.data = torch.abs(w.data)

        # record objective values
        if kernel is None:
            Z,l_new = func_z(w*X[ind, :], w*Y[ind, :]).detach().cpu().numpy(),loss(X[ind, :],Y[ind, :],w,lam_norm).detach().cpu().numpy()
        else:
            Z,l_new = func_z(w*X[ind, :],w*Y[ind, :],kernel).detach().cpu().numpy(),loss(X[ind, :],Y[ind, :],w,lam_norm,kernel).detach().cpu().numpy()
        # weights[t] = w.detach().cpu().numpy()

        # check the change of the objective values
        rel_change = abs((l_new-l.detach().cpu().numpy())/l.detach().cpu().numpy())
        if rel_change <= ftol:
            final_t = t
            break
        else:
            final_t = t

        # Print update
        if verbose and t % 10 == 0:
            print(f'Step {t+1} of {max_iter}. Loss: {l_new}. MMD: {Z}. relative change: {rel_change}')

    # optimization info
    info = {'weights': w.detach().cpu().numpy(), 'final_loss': l_new, 'final_MMD': Z, 'success':True}
    return info
