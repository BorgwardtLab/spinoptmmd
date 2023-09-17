import pandas as pd
import shap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import  roc_auc_score, accuracy_score, \
balanced_accuracy_score, f1_score, recall_score, precision_score
import os
from pathlib import Path
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, confusion_matrix, auc,\
    precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from collections import Counter
import warnings
from lightgbm import LGBMClassifier
import argparse
import statsmodels.api as sm
warnings.filterwarnings("ignore")
import pickle
from tqdm import tqdm


###########################################################################################################
def createSplit_noaddStratify(Y, n_splits = 5):


    # Check this - your variables need to be str here
    concat_labels = Y.copy().astype(int).astype(str)

    # Perform multiple splits - train, val, test
    df = pd.Series(Y.index)
    skf = StratifiedKFold(n_splits=n_splits)
    split = 0
    partition = dict()
    for train_index, test_index in skf.split(df, concat_labels):

        # First get validation and training together, then split again
        id_rest = df.loc[df.index.intersection(train_index)].values
        y_rest = concat_labels.iloc[train_index]
        id_trn, id_val, y_trn, y_val = train_test_split(id_rest, y_rest, test_size=0.25, random_state=1,
                                                        stratify=y_rest)

        # Test set
        id_tst = df.loc[df.index.intersection(test_index)].values

        # Save as dictionary
        partition['train'+str(split)] = list(id_trn)
        partition['validation'+str(split)] = list(id_val)
        partition['test'+str(split)] = id_tst
        split = split + 1

    try:
        Y = Y.to_frame()
    except:
        print('already DF')
    labels = dict(zip(Y.index.values, Y.astype(int)[Y.columns[0]].values))

    return partition, labels
def getData(partition, X, cv):
    X_train = X.loc[partition['train' + str(cv)], :]
    X_val = X.loc[partition['validation' + str(cv)], :]
    X_test = X.loc[partition['test' + str(cv)], :]

    return X_train, X_test,X_val
def getLabels(partition, labels, cv):
    # Generate data
    y_trn = [labels[ID] for i, ID in enumerate(partition['train' + str(cv)])]
    y_val = [labels[ID] for i, ID in enumerate(partition['validation' + str(cv)])]
    y_tst = [labels[ID] for i, ID in enumerate(partition['test' + str(cv)])]
    return y_trn, y_tst, y_val
def perf_90recall(models, X_tests, y_tests):
    if len(np.unique(y_tests)) == 2:

        pred_test = []
        y_test_merged = y_tests  # list(chain(*y_tests))

        pred_test = models.predict_proba(X_tests)[:, 1]

        precisions, recalls, thresholds = precision_recall_curve(y_test_merged, pred_test)
        threshold = thresholds[np.abs(recalls - 0.9).argmin()]

        pred_test = (models.predict_proba(X_tests)[:, 1] >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test_merged, pred_test).ravel()
        performance = {}
        performance['precision'] = tp / (tp + fp)
        performance['specificity'] = tn / (tn + fp)
        performance['sensitivity'] = tp / (tp + fn)
        performance['f1'] = f1_score(y_test_merged, pred_test)
        performance['npv'] = tn / (tn + fn)
        performance['accuracy'] = (tp + tn) / (tp + fp + tn + fn)
        performance['recall'] = tp / (tp + fn)
        # performance = pd.DataFrame(performance, index=[name])

        y_score = models.predict_proba(X_tests)[:, 1]
        aps = average_precision_score(y_test_merged, y_score)
        fp_rates, tp_rates, _ = roc_curve(y_test_merged, y_score)
        roc_auc = auc(fp_rates, tp_rates)

        # if not os.path.exists('{}/perf_90recall.csv'.format(output_path)):
        # performance.to_csv('{}/perf_90recall.csv'.format(output_path))
    # else:
    # performance.to_csv('{}/perf_90recall.csv'.format(output_path), header=False, mode='a')

    else:
        print('multi class!')
        # multiclass:
        # Macro averaged precision: calculate precision for all classes individually and then average them
        # Micro averaged precision: calculate class wise true positive and false positive and then use that to calculate overall precision

        tn = np.nan
        fp = np.nan
        fn = np.nan
        tp = np.nan
        y_pred = models.predict_proba(X_tests)
        y_pred_class = models.predict(X_tests)

        precision = precision_score(y_tests, y_pred_class, average='weighted')
        recall = recall_score(y_tests, y_pred_class, average='weighted')
        tp = accuracy_score(y_tests, y_pred_class)
        acc = balanced_accuracy_score(y_tests, y_pred_class)
        F1 = f1_score(y_tests, y_pred_class, average='weighted')
        roc_auc = roc_auc_score(y_tests, y_pred, average='weighted', multi_class='ovo')
        aps = np.nan  # average_precision_score(y, y_pred, average = 'weighted')

        performance = {}
        performance['precision'] = precision
        performance['f1'] = F1
        performance['accuracy'] = acc
        performance['recall'] = recall

    return tn, fp, fn, tp, performance['accuracy'], performance['precision'], \
           performance['recall'], roc_auc, aps, performance['f1']
def getPredictionSingleFold_nooutput(cv, partition, labels, clf_choice, clinical):

    # Get the labels
    y_train, y_test, y_val = getLabels(partition, labels, cv)
    y_trainVal = y_train+ y_val

    # Get data
    X_train, X_test, X_val = getData(partition, clinical, cv)
    X_trainVal = pd.concat([X_train,X_val])


    # Create a list where train data indices are -1 and validation data indices are 0
    split_index = [-1 if x in X_test.index else labels[x] for x in X_trainVal.index]
    ps = PredefinedSplit(test_fold=split_index)

    if clf_choice == 'MLP':
        clf = MLPClassifier(max_iter=1000)

        param_grid = {'solver': ['adam'],
                      'alpha': [0, 1e-04, 1e-02],
                      'activation': ['relu', 'logistic'],
                      'hidden_layer_sizes': [5, 10, 50],
                      'learning_rate_init': [0.001, 0.0001],
                      'early_stopping': [True],
                      'validation_fraction': [0.25]}
        clf = GridSearchCV(clf, param_grid, cv=3, verbose=1, scoring='roc_auc',refit=True)

    if clf_choice == 'LR':

        # Initialise logistic regression model, optimise hyperparameters by gridsearch
        clf = LogisticRegression(max_iter=10000, class_weight='balanced', random_state=1)
        param_grid = {'penalty': ["l1", 'l2', 'elasticnet'],
                      'C': np.logspace(-7, 4, 12),
                      'solver': ['liblinear']}
        # clf = GridSearchCV(clf, param_grid, cv=ps, verbose=1, scoring='roc_auc',refit=True)
        clf = GridSearchCV(clf, param_grid, cv=3, verbose=1, scoring='roc_auc', refit=True)

    if clf_choice == 'RF':
        # Initialise random forest, optimise hyperparameters by random grid
        clf = RandomForestClassifier(class_weight='balanced', random_state=1)

        # Number of trees in random forest
        n_estimators = [50, 200, 1000]

        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']

        # Maximum number of levels in tree
        max_depth = [3,5,7]
        max_depth.append(None)

        # Minimum number of samples required to split a node
        min_samples_split = [3,5,10]

        # Minimum number of samples required at each leaf node
        min_samples_leaf = [3, 6, 9]

        # Method of selecting samples for training each tree
        bootstrap = [True]

        # Create the random grid
        param_grid = {'n_estimators': n_estimators,
                      'max_features': max_features,
                      'max_depth': max_depth,
                      'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf,
                      'bootstrap': bootstrap,
                      'class_weight': ['balanced']}
        # clf = RandomizedSearchCV(clf, param_grid, cv=5, n_iter=50, verbose=1, scoring='roc_auc')
        clf = GridSearchCV(clf, param_grid, cv=3, verbose=1, scoring='roc_auc',refit=True)

    if clf_choice == 'lightGBM':

        # Initialise model, optimise hyperparameters by random sampling
        clf = LGBMClassifier(class_weight='balanced', random_state=1)

        # Maximum tree leaves for base learners
        num_leaves = [15, 31]

        # Boosting learning rate
        learning_rate = [0.1]

        # Number of boosted trees to fit
        n_estimators = [200]

        # L1 regularisation term on weights
        reg_alpha = [0, 1e-02]

        # L2 regularisation term on weights
        reg_lambda = [0, 1e-02]

        # Create the random grid
        param_grid = {'num_leaves': num_leaves,
                      'learning_rate': learning_rate,
                      'n_estimators': n_estimators,
                      'reg_alpha': reg_alpha,
                      'reg_lambda': reg_lambda}

        # Could replace for random search here, too
        clf = GridSearchCV(clf, param_grid, cv=3, verbose=1, scoring='roc_auc')


    # Fit
    clf.fit(X_trainVal, y_trainVal)

    # Save best estimator
    best_est = clf.best_estimator_
    best_est.fit(X_trainVal, y_trainVal)

    return best_est, X_train, X_val,X_test
def logistic_regression(X, X_additional, Y, n_proteins):
    pvals = []
    for i in tqdm(range(n_proteins)):  # a protein each time
        X_i = X[:, i]
        X_tot = np.c_[X_additional, X_i]
        model = sm.Logit(Y, X_tot).fit(disp=0, method='bfgs')
        pvals.append(model.pvalues[-1])
    return pvals

###########################################################################################################

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--i_cv', type=int, required=True)
parser.add_argument('--i_clf', type=int, required=True)
parser.add_argument('--i_data', type=int, required=True)
args           = parser.parse_args()

i_clf      = args.i_clf # Choose classifier
i_cv       = args.i_cv # CV fold
i_data      = args.i_data # dataset index

###########################################################################################################

path = '/MMD/' # path to folders
n_splits = [5, 5, 5, 5, 3, 3, 3, 3]
n_split = n_splits[i_data]

clfs       = ['lightGBM','LR']
clf_choice = clfs[i_clf] #Choice of classifier.

inputs    = ['ADNI','DRIAMS_Escherichia','DRIAMS_Klebsiella','DRIAMS_Staphylococcus',
             'lgg_rad_ATRX','lgg_rad_TP53','lgg_expr_ATRX','lgg_expr_TP53',]
input     = inputs[i_data ]



data_folders = ['ADNI/3DradiomicFeatures_HC_lr.csv',
                'DRIAMS/driams_Escherichia coli_Ceftriaxone_features.csv',
                'DRIAMS/driams_Klebsiella pneumoniae_Ceftriaxone_features.csv',
                'DRIAMS/driams_Staphylococcus aureus_Oxacillin_features.csv',
                'LGG/radiomic_feat.csv',
                'LGG/radiomic_feat.csv',
                'LGG/expr.csv',
                'LGG/expr.csv'
                ]
label_folders = ['ADNI/labels.csv',
                 'DRIAMS/driams_Escherichia coli_Ceftriaxone_labels.csv',
                'DRIAMS/driams_Klebsiella pneumoniae_Ceftriaxone_labels.csv',
                'DRIAMS/driams_Staphylococcus aureus_Oxacillin_labels.csv',
                'LGG/labels_ATRX.csv',
                'LGG/labels_TP53.csv',
                 'LGG/labels_ATRX.csv',
                 'LGG/labels_TP53.csv'
                 ]
feature_folders_lin = ['output/mean_contrast_names_adni_rad_test_lin_lam_0.1.csv',
                    'output/mean_contrast_names_driams_Escherichia coli_Ceftriaxone_test_lin_lam_0.1.csv',
                   'output/mean_contrast_names_driams_Klebsiella pneumoniae_Ceftriaxone_test_lin_lam_0.1.csv',
                   'output/mean_contrast_names_driams_Staphylococcus aureus_Oxacillin_test_lin_lam_0.1.csv',
                   'output/mean_contrast_names_lgg_rad_ATRX_lin_lam_0.1.csv',
                   'output/mean_contrast_names_lgg_rad_TP53_lin_lam_0.1.csv',
                    'output/mean_contrast_names_lgg_expr_ATRX_lin_lam_0.1.csv',
                'output/mean_contrast_names_lgg_expr_TP53_lin_lam_0.1.csv'
                   ]
feature_folders_quad = ['output/mean_contrast_names_adni_rad_test_quad_lam_0.1.csv',
                    'output/mean_contrast_names_driams_Escherichia coli_Ceftriaxone_test_quad_lam_0.1.csv',
                   'output/mean_contrast_names_driams_Klebsiella pneumoniae_Ceftriaxone_test_quad_lam_0.1.csv',
                   'output/mean_contrast_names_driams_Staphylococcus aureus_Oxacillin_test_quad_lam_0.1.csv',
                   'output/mean_contrast_names_lgg_rad_ATRX_quad_lam_0.1.csv',
                   'output/mean_contrast_names_lgg_rad_TP53_quad_lam_0.1.csv',
                        'output/mean_contrast_names_lgg_expr_ATRX_quad_lam_0.1.csv',
                        'output/mean_contrast_names_lgg_expr_TP53_quad_lam_0.1.csv'
                   ]




# Output prep
output_folder = 'results_'+input
output        = os.path.join(path,output_folder)
Path(output).mkdir(parents=True, exist_ok=True)
output_model  = os.path.join(path,output_folder,'models')
Path(output_model).mkdir(parents=True, exist_ok=True)
output_importance  = os.path.join(path,output_folder,'importance')
Path(output_importance).mkdir(parents=True, exist_ok=True)

# Splits
split_folder = os.path.join(path,'partitions')
Path(split_folder).mkdir(parents=True, exist_ok=True)

print('Working on '+input +'_'+clf_choice+ '_cv'+ str(i_cv))



###########################################################################################################
# Run prediction with feature selection

# Get label data
label = pd.read_csv(os.path.join(path, label_folders[i_data]), index_col=0)
data  = pd.read_csv(os.path.join(path, data_folders[i_data]), index_col=0)

# Check for labels if multiple images per patient
data = data.dropna(axis = 1)
data[data == '#DIV/0!'] = np.nan
data = data.dropna(axis=0).astype(float)
pats_keep = list(set(label.index)&set(data.index))
data = data.loc[pats_keep]
label = label.loc[pats_keep]
feature_names = np.array(list(data.columns))

# Load results
try:
    top_feature_MMD_lin  = feature_names[np.array([f for f in pd.read_csv(os.path.join(path, feature_folders_lin[i_data]), index_col=0, header = None).index])>0]
except:
    mmd_feat = pd.read_csv(os.path.join(path, feature_folders_lin[i_data]), header=None)
    inds = np.array([f[0] for f in mmd_feat.values]) > 0
    top_feature_MMD_lin = feature_names[inds]

#     print('No linear MMD features!')
#     top_feature_MMD_lin = []
try:
    top_feature_MMD_quad = feature_names[np.array([f for f in pd.read_csv(os.path.join(path, feature_folders_quad[i_data]), index_col=0, header = None).index])>0]
except:
    mmd_feat = pd.read_csv(os.path.join(path, feature_folders_quad[i_data]), header=None)
    inds = np.array([f[0] for f in mmd_feat.values]) > 0
    top_feature_MMD_quad = feature_names[inds]
    # print('No quadratic MMD features!')
    # top_feature_MMD_quad = []
n_feat_MMD_lin = len(top_feature_MMD_lin)
if n_feat_MMD_lin ==0:
    print('No significant Features found with MMD linear!')

n_feat_MMD_quad = len(top_feature_MMD_quad)
if n_feat_MMD_quad ==0:
    print('No significant Features found with MMD linear!')



# Get splits - create or load:
try:
    labels = np.load(os.path.join(path, split_folder, input + '_labels.npy'), allow_pickle='TRUE').item()
    partition = np.load(os.path.join(path, split_folder, input + '_partition.npy'), allow_pickle='TRUE').item()
except:
    print('creating new splits!')
    partition, labels = createSplit_noaddStratify(label,n_splits = n_split)
    np.save(os.path.join(path, split_folder, input + '_labels.npy'), labels)
    np.save(os.path.join(path, split_folder, input + '_partition.npy'), partition)
data = data.loc[labels.keys()]


# Prepare outputs
runs = ['all_features', 'UnivasFS_lin', 'SHAP_lin','MMD_lin','UnivasFS_quad', 'SHAP_quad','MMD_quad']
df_rank        = pd.DataFrame(index = data.columns, columns=runs)
df_importance  = pd.DataFrame(index = data.columns, columns=runs)
df_performance = pd.DataFrame(index = runs,columns = ['Prevalence','ROCAUC','AUPRC','relAPS','TN','FP', 'FN', 'TP',
                                                      'Accuracy', 'Precision', 'Recall'])

# Run prediction as nested CV
for cv in [i_cv]:

    print('Fold ' + str(cv))
    name_save = input +'_'+clf_choice+ '_cv'+ str(cv)
    print('Working on '+ name_save)


    # Scale protein data
    sc = StandardScaler()
    npx_reform_train = pd.DataFrame(sc.fit_transform(
        data.loc[partition['train' + str(cv)], :].values.astype(float)),
        index=data.loc[partition['train' + str(cv)], :].index,
        columns=data.columns)

    npx_reform_val = pd.DataFrame(sc.transform(
        data.loc[partition['validation' + str(cv)], :].values),
        index=data.loc[partition['validation' + str(cv)], :].index,
        columns=data.columns)

    npx_reform_test = pd.DataFrame(sc.transform(
        data.loc[partition['test' + str(cv)], :].values),
        index=data.loc[partition['test' + str(cv)], :].index,
        columns=data.columns)

    data_norm = npx_reform_train.append(npx_reform_val).append(npx_reform_test)


    #################################################################################################################
    # Predict with all features
    print('Prediction with all features')
    best_est_cv_all, X_train_all, X_val_all,X_test_all = \
        getPredictionSingleFold_nooutput(cv, partition, labels,  clf_choice, data_norm)
    pickle.dump(best_est_cv_all, open(os.path.join(output_model, name_save+'_model_allFeatures.save'), 'wb'))



    # Evaluate performance (all features)
    tn, fp, fn, tp, accuracy, precision, recall, \
    roc_auc_top, aps_top, f1 = perf_90recall(best_est_cv_all, X_test_all, [labels[p] for p in X_test_all.index])
    prev = np.mean([labels[p] for p in X_test_all.index])
    df_performance.loc['all_features',:] = [prev,roc_auc_top,aps_top,aps_top/prev,tn, fp, fn, tp, accuracy, precision, recall]
    #################################################################################################################


    # SHAP VALUES
    print('Running SHAP')
    if clf_choice == 'LR':
        explainer = shap.Explainer(best_est_cv_all, X_train_all, feature_names=X_train_all.columns)
    elif clf_choice == 'MLP' or clf_choice == 'lightGBM':
        explainer = shap.Explainer(best_est_cv_all.predict, X_train_all, feature_names=X_train_all.columns)
    elif clf_choice == 'RF':
        explainer = shap.TreeExplainer(best_est_cv_all, X_train_all, feature_names=X_train_all.columns,check_additivity=False)
    if clf_choice == 'RF':
        shap_values = explainer(X_train_all, check_additivity=False)
    else:
        num_features = X_train_all.shape[1]
        shap_values = explainer(X_train_all, max_evals=2 * num_features + 2)

    # SHAP plot
    feature_names = shap_values.feature_names
    try:
        shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
    except:
        shap_df = pd.DataFrame(shap_values.values[:, :, 0], columns=feature_names)
    vals = np.abs(shap_df.values).mean(0)
    shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                   columns=['col_name', 'feature_importance_vals'])
    shap_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    shap_importance.set_index('col_name', inplace=True)
    df_importance.loc[:,'SHAP'] = shap_importance.loc[df_importance.index,'feature_importance_vals']



    if n_feat_MMD_lin >0:
        # Feature selection - based on same number of feature as MMD linear
        top_features_SHAP_lin = shap_importance.index[:n_feat_MMD_lin]
        data_in_SHAP_lin = data_norm.loc[:, top_features_SHAP_lin]

        # Predict again with top features from univariate FS
        best_est_cv_SHAP_lin, X_train_SHAP_lin, X_val_SHAP_lin,X_test_SHAP_lin = \
            getPredictionSingleFold_nooutput(cv, partition, labels,  clf_choice, data_in_SHAP_lin)
        pickle.dump(best_est_cv_SHAP_lin, open(os.path.join(output_model, name_save + '_model_SHAP_lin.save'), 'wb'))


        # Evaluate performance (top features)
        tn, fp, fn, tp, accuracy, precision, recall, \
        roc_auc_top, aps_top, f1 = perf_90recall(best_est_cv_SHAP_lin, X_test_SHAP_lin, [labels[p] for p in X_test_SHAP_lin.index])
        prev = np.mean([labels[p] for p in X_test_SHAP_lin.index])
        df_performance.loc['SHAP_lin',:] = [prev,roc_auc_top,aps_top,aps_top/prev,tn, fp, fn, tp, accuracy, precision, recall]

    if n_feat_MMD_quad > 0:
        # Feature selection - based on same number of feature as MMD linear
        top_features_SHAP_quad = shap_importance.index[:n_feat_MMD_quad]
        data_in_SHAP_quad = data_norm.loc[:, top_features_SHAP_quad]

        # Predict again with top features from univariate FS
        best_est_cv_SHAP_quad, X_train_SHAP_quad, X_val_SHAP_quad, X_test_SHAP_quad = \
            getPredictionSingleFold_nooutput(cv, partition, labels, clf_choice, data_in_SHAP_quad)
        pickle.dump(best_est_cv_SHAP_quad,
                    open(os.path.join(output_model, name_save + '_model_SHAP_quad.save'), 'wb'))

        # Evaluate performance (top features)
        tn, fp, fn, tp, accuracy, precision, recall, \
        roc_auc_top, aps_top, f1 = perf_90recall(best_est_cv_SHAP_quad, X_test_SHAP_quad,
                                                 [labels[p] for p in X_test_SHAP_quad.index])
        prev = np.mean([labels[p] for p in X_test_SHAP_quad.index])
        df_performance.loc['SHAP_quad', :] = [prev, roc_auc_top, aps_top, aps_top / prev, tn, fp, fn, tp, accuracy,
                                             precision, recall]


    #################################################################################################################
    # Get feature importance ranking by univariate association
    print('Running univariate association')
    n, n_feat = npx_reform_train.shape
    X_additional = np.c_[np.ones(n)]  # , COVs_sc.values]
    X = npx_reform_train.values.astype(np.float64)
    y_train = [labels[p] for p in partition['train'+str(cv)]]
    pvals = logistic_regression(X, X_additional, y_train, n_feat)
    pvals = pd.Series(pvals, index=data_norm.columns).sort_values()
    df_importance.loc[:, 'UnivasFS'] = pvals[df_importance.index]


    if n_feat_MMD_lin > 0:
        # Feature selection
        top_features_UVFS_lin = pvals.index[:n_feat_MMD_lin]
        data_in_UVFS_lin  = data_norm.loc[:,top_features_UVFS_lin]


        # Predict again with top features from univariate FS
        best_est_cv_UVFS_lin, X_train_UVFS_lin, X_val_UVFS_lin,X_test_UVFS_lin = \
            getPredictionSingleFold_nooutput(cv, partition, labels,  clf_choice, data_in_UVFS_lin)
        pickle.dump(best_est_cv_UVFS_lin, open(os.path.join(output_model, name_save + '_model_UVFS_lin.save'), 'wb'))


        # Evaluate performance (top features)
        tn, fp, fn, tp, accuracy, precision, recall, \
        roc_auc_top, aps_top, f1 = perf_90recall(best_est_cv_UVFS_lin, X_test_UVFS_lin, [labels[p] for p in X_test_UVFS_lin.index])
        prev = np.mean([labels[p] for p in X_test_UVFS_lin.index])
        df_performance.loc['UnivasFS_lin',:] = [prev,roc_auc_top,aps_top,aps_top/prev,tn, fp, fn, tp, accuracy, precision, recall]

    if n_feat_MMD_quad > 0:
        # Feature selection
        top_features_UVFS_quad = pvals.index[:n_feat_MMD_quad]
        data_in_UVFS_quad = data_norm.loc[:, top_features_UVFS_quad]

        # Predict again with top features from univariate FS
        best_est_cv_UVFS_quad, X_train_UVFS_quad, X_val_UVFS_quad, X_test_UVFS_quad = \
            getPredictionSingleFold_nooutput(cv, partition, labels, clf_choice, data_in_UVFS_quad)
        pickle.dump(best_est_cv_UVFS_quad, open(os.path.join(output_model, name_save + '_model_UVFS_quad.save'), 'wb'))

        # Evaluate performance (top features)
        tn, fp, fn, tp, accuracy, precision, recall, \
        roc_auc_top, aps_top, f1 = perf_90recall(best_est_cv_UVFS_quad, X_test_UVFS_quad,
                                                 [labels[p] for p in X_test_UVFS_quad.index])
        prev = np.mean([labels[p] for p in X_test_UVFS_quad.index])
        df_performance.loc['UnivasFS_quad', :] = [prev, roc_auc_top, aps_top, aps_top / prev, tn, fp, fn, tp, accuracy,
                                             precision, recall]
    #################################################################################################################

    # MMD linear
    print('Running MMD linear')
    if len(top_feature_MMD_lin) > 0:

        # Feature selection
        data_in_MMD_lin = data_norm.loc[:, top_feature_MMD_lin]
        df_importance.loc[:, 'MMD_lin'] = [f for f in pd.read_csv(os.path.join(path, feature_folders_lin[i_data]), index_col=0, header = None).index]

        # Predict again with top features from univariate FS
        best_est_cv_MMD_lin, X_train_MMD_lin, X_val_MMD_lin, X_test_MMD_lin = \
            getPredictionSingleFold_nooutput(cv, partition, labels, clf_choice, data_in_MMD_lin)
        pickle.dump(best_est_cv_MMD_lin, open(os.path.join(output_model, name_save + '_model_MMD_lin.save'), 'wb'))

        # Evaluate performance (top features)
        tn, fp, fn, tp, accuracy, precision, recall, \
        roc_auc_top, aps_top, f1 = perf_90recall(best_est_cv_MMD_lin, X_test_MMD_lin, [labels[p] for p in X_test_MMD_lin.index])
        prev = np.mean([labels[p] for p in X_test_MMD_lin.index])
        df_performance.loc['MMD_lin', :] = [prev, roc_auc_top, aps_top, aps_top / prev,tn, fp, fn, tp, accuracy, precision, recall]

    #################################################################################################################
    # MMD quadratic
    print('Running MMD quadratic')
    if len( top_feature_MMD_quad)>0:

        # Feature selection
        data_in_MMD_quad = data_norm.loc[:, top_feature_MMD_quad]
        df_importance.loc[:, 'MMD_quad'] = [f for f in pd.read_csv(os.path.join(path, feature_folders_quad[i_data]), index_col=0, header = None).index]


        # Predict again with top features from univariate FS
        best_est_cv_MMD_quad, X_train_MMD_quad, X_val_MMD_quad, X_test_MMD_quad = \
            getPredictionSingleFold_nooutput(cv, partition, labels, clf_choice, data_in_MMD_quad)
        pickle.dump(best_est_cv_MMD_quad, open(os.path.join(output_model, name_save + '_model_MMD_quad.save'), 'wb'))

        # Evaluate performance (top features)
        tn, fp, fn, tp, accuracy, precision, recall, \
        roc_auc_top, aps_top, f1 = perf_90recall(best_est_cv_MMD_quad, X_test_MMD_quad, [labels[p] for p in X_test_MMD_quad.index])
        prev = np.mean([labels[p] for p in X_test_MMD_quad.index])
        df_performance.loc['MMD_quad', :] = [prev, roc_auc_top, aps_top, aps_top / prev,tn, fp, fn, tp, accuracy, precision, recall]

    #################################################################################################################


# Save results to df
df_performance.to_csv(os.path.join(output,name_save +'_performance.csv'))
df_importance.to_csv(os.path.join(output_importance,name_save +'_importance.csv'))











