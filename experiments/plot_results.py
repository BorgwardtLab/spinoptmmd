import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob

plt.rc('font', size=16)

def files_to_df(files):
    dfs = []
    for f in sorted(files):
        df = pd.read_csv(f)
        if 'lin_' in f and not 'baseline' in f:
            df.insert(0,'experiment','Lin-opt')
            # df.loc[:,'experiment'] = 'Lin-opt'
        if 'lin_' in f and 'baseline' in f:
            df.insert(0,'experiment','Lin')
            # df.loc[:,'experiment'] = 'Linear'
        if 'quad_' in f and not 'baseline' in f:
            df.insert(0,'experiment','Quad-opt')
            # df.loc[:,'experiment'] = 'Quad-opt'
        if 'quad_' in f and 'baseline' in f:
            df.insert(0,'experiment','Quad')
            # df.loc[:,'experiment'] = 'Quad'
        dfs.append(df)
    return pd.concat(dfs,axis=0,ignore_index=True)
            

data_root = 'output'

experiment = ['same_gaussian','mean_shift','variance_shift','blobs']
varying = ['d','lambda','sample_size']
modality = ['pvalue','h0_rejected','type_I_error','type_II_error','runtime']
modality_dict = {'pvalue':'P-value','h0_rejected':'Test power','type_I_error':'Type I error','type_II_error':'Type II error','runtime':'Runtime [s]'}
varying_dict = {'d':'Dimension','lambda':'lambda','sample_size':'Dataset size'}

for e in experiment:
    for v in varying:
        if e=='blobs' and v=='d':
            continue
        files = glob(data_root+'/'+f"{e}_*_{v}*.csv")
        df = files_to_df(files)
        df.loc[:,'lambda'] = df.loc[:,'lambda']/df.loc[:,'d']
        for m in modality:
            if m in df.columns.values:
                plt.figure()
                if m in ['pvalue','runtime']:
                    if m != 'runtime':
                        plt.errorbar(df[df.experiment=='Quad-opt'].loc[:,v].values,df[df.experiment=='Quad-opt'].loc[:,m].values,yerr=df[df.experiment=='Quad-opt'].loc[:,m+'_std'].values,label='Quad-opt',marker='o')
                        plt.errorbar(df[df.experiment=='Lin-opt'].loc[:,v].values,df[df.experiment=='Lin-opt'].loc[:,m].values,yerr=df[df.experiment=='Lin-opt'].loc[:,m+'_std'].values,label='Lin-opt',marker='v')
                    if v != 'lambda':
                        plt.errorbar(df[df.experiment=='Quad'].loc[:,v].values,df[df.experiment=='Quad'].loc[:,m].values,yerr=df[df.experiment=='Quad'].loc[:,m+'_std'].values,label='Quad',ls='dashdot',marker='o')
                        plt.errorbar(df[df.experiment=='Lin'].loc[:,v].values,df[df.experiment=='Lin'].loc[:,m].values,yerr=df[df.experiment=='Lin'].loc[:,m+'_std'].values,label='Lin',ls='dashdot',marker='v')
                else:
                    plt.plot(df[df.experiment=='Quad-opt'].loc[:,v].values,df[df.experiment=='Quad-opt'].loc[:,m].values,label='Quad-opt',marker='o')
                    plt.plot(df[df.experiment=='Lin-opt'].loc[:,v].values,df[df.experiment=='Lin-opt'].loc[:,m].values,label='Lin-opt',marker='v')
                    if v != 'lambda' and m not in ['type_I_error','type_II_error']:
                        plt.plot(df[df.experiment=='Quad'].loc[:,v].values,df[df.experiment=='Quad'].loc[:,m].values,label='Quad',ls='dashdot',marker='o')
                        plt.plot(df[df.experiment=='Lin'].loc[:,v].values,df[df.experiment=='Lin'].loc[:,m].values,label='Lin',ls='dashdot',marker='v')
                if v=='sample_size':
                    plt.legend()
                if v=='lambda':
                    plt.xscale('log')
                plt.xlabel(varying_dict[v])
                if e=='same_gaussian' and m=='h0_rejected':
                    plt.ylabel('Type I error')
                else:
                    plt.ylabel(modality_dict[m])
                plt.tight_layout()
                plt.savefig(f"output/{e}_vary_{v}_{m}.pdf")
                plt.close()