'''
This file contains the source code for the linear and quadratic MMD two sample tests.
They are refactored and extended versions of:  https://github.com/wittawatj/interpretable-test
We added our optimisation and feature selection algorithms and replaced numpy with torch operations.

In the future, this could be extended to run on GPUs.
'''
from abc import ABCMeta, abstractmethod
from six import with_metaclass

import numpy as np
import functools
from tqdm import tqdm
import torch

from scipy.optimize import minimize
from scipy.stats import norm, chi2, ncx2

import src.kernel
from src.utils import m_r, find_weights_sp, find_weights_torch, prune_features

class TwoSampleTest(with_metaclass(ABCMeta, object)):
    """Abstract class for two sample tests."""

    def __init__(self, alpha=0.01):
        """
        alpha: significance level of the test
        """
        self.alpha = alpha

    @abstractmethod
    def perform_test(self, tst_data):
        """perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        tst_data: an instance of TSTData
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_stat(self, tst_data):
        """Compute the test statistic"""
        raise NotImplementedError()

class LinearMMDTest(TwoSampleTest):
    """Two-sample test with linear MMD^2 statistic.
    """

    def __init__(self, kernel, alpha=0.01):
        """
        kernel: an instance of Kernel
        """
        self.kernel = kernel
        self.alpha = alpha

    def perform_test(self, tst_data, w=None):
        """perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        tst_data: an instance of TSTData
        """
        X, Y = tst_data.xy()
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        if w is None:
            w = 1
        else:
            w = torch.from_numpy(w)
        n = X.shape[0]
        stat, snd = LinearMMDTest.two_moments(w*X, w*Y, self.kernel)
        stat = stat.detach().cpu().item()
        snd = snd.detach().cpu().item()
        # var = snd - stat**2
        var = snd # Not sure why this was selected in the original code.
        pval = norm.sf(stat, loc=0, scale=(2.0*var/n)**0.5) # Is this 1/n correct?
        results = {'alpha': self.alpha, 'pvalue': pval, 'test_stat': stat,
                'h0_rejected': pval < self.alpha}
        return results
    
    @staticmethod
    def compute_stat(X, Y, kernel):
        return LinearMMDTest.two_moments(X, Y, self.kernel)[0]

    @staticmethod
    def two_moments(X, Y, kernel):
        """Compute linear mmd estimator and a linear estimate of
        the uncentred 2nd moment of h(z, z'). Total cost: O(n).

        return: (linear mmd, linear 2nd moment)
        """

        if X.shape[0] != Y.shape[0]:
            raise ValueError('Require sample size of X = size of Y')
        n = X.shape[0]
        if n%2 == 1:
            # make it even by removing the last row
            X = X[:-1]
            Y = Y[:-1]

        Xodd = X[::2, :]
        Xeven = X[1::2, :]
        assert Xodd.shape[0] == Xeven.shape[0]
        Yodd = Y[::2, :]
        Yeven = Y[1::2, :]
        assert Yodd.shape[0] == Yeven.shape[0]

        # linear mmd. O(n)
        xx = kernel.pair_eval_torch(Xodd, Xeven)
        yy = kernel.pair_eval_torch(Yodd, Yeven)
        xo_ye = kernel.pair_eval_torch(Xodd, Yeven)
        xe_yo = kernel.pair_eval_torch(Xeven, Yodd)
        h = xx + yy - xo_ye - xe_yo
        lin_mmd = torch.mean(h)
        """
        Compute a linear-time estimate of the 2nd moment of h = E_z,z' h(z, z')^2.
        Note that MMD = E_z,z' h(z, z').
        Require O(n). Same trick as used in linear MMD to get O(n).
        """
        lin_2nd = torch.mean(h**2)
        return lin_mmd, lin_2nd

    @staticmethod
    def median_distance(X,Y,p=2):
        # Cast into torch format
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        # Drop duplicated points to avoid distorting the median
        # Usually this is for all 0 points
        X = torch.unique(X,dim=0)
        Y = torch.unique(Y,dim=0)

        distances = torch.cdist(X,Y,p=p)
        # indices = torch.triu_indices(*distances.shape,offset=1)
        # median_distance = distances[indices].view(-1).quantile(0.5)
        median_distance = distances.view(-1).quantile(0.5)
        return 2*median_distance.item()**2

    @staticmethod
    def grid_search_kernel(X,Y, list_kernels, alpha):
        """
        Return from the list the best kernel that maximizes the test power.

        return: (best kernel index, list of test powers)
        """
        
        # Cast into torch format
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        n = X.shape[0]

        powers = torch.zeros(len(list_kernels))
        for ki, kernel in enumerate(list_kernels):
            lin_mmd, snd_moment = LinearMMDTest.two_moments(X, Y, kernel)
            var_lin_mmd = (snd_moment - lin_mmd**2)
            # test threshold from N(0, var)
            thresh = norm.isf(alpha, loc=0, scale=(2.0*var_lin_mmd/n)**0.5)
            power = norm.sf(thresh, loc=lin_mmd, scale=(2.0*var_lin_mmd/n)**0.5)
            #power = lin_mmd/var_lin_mmd
            powers[ki] = power
        best_ind = torch.argmax(powers)
        return best_ind, powers.numpy()

    @staticmethod
    def test_power(X,Y,kernel,alpha=0.01,lam_var=1e-8):
        n = X.shape[0]
        lin_mmd, lin_2nd = LinearMMDTest.two_moments(X, Y, kernel)
        lin_2nd += lam_var # for numerical stability
        var = lin_2nd - lin_mmd**2
        c_alpha = norm.isf(alpha,loc=0,scale=(2.0*lin_2nd/n).detach().cpu()**0.5)
        return (lin_mmd-c_alpha)/var**0.5

    @staticmethod
    def optimize_w(tst_data, alpha, 
                   kernel=None,
                   optimise_gwidth = False,
                   max_iter=400,
                   lam=1e-1,
                   step_size=1e-2,
                   batch_proportion=1.0,
                   ftol=2.220446049250313e-09,
                   verbose=True,
                   lam_var=1e-8, **kwargs):

        if type(kernel).__name__=='KGauss' and optimise_gwidth:
            list_kernels = [kernel_torch.KGauss(sigma2=s) for s in kernel.sigma2*2**np.linspace(-4,4,30)]
            kernel = QuadMMDTest.grid_search_kernel(test_data.X, tst_data.Y, list_kernels, alpha=alpha)

        func_z = functools.partial(LinearMMDTest.test_power,alpha=alpha,lam_var=lam_var)

        # info = optimization info
        info = find_weights_sp(tst_data.X,tst_data.Y, func_z,
                max_iter=max_iter, lam=lam, step_size=step_size,
                batch_proportion=batch_proportion,
                ftol=ftol, kernel=kernel,
                verbose=verbose)
        return info

    @staticmethod
    def significance_thresholds_perm(tst_data,
                                    compute_thresholds = False,
                                    alpha_feat = 0.01,
                                    n_permutations=400,
                                    alpha=0.01,
                                    kernel=None,
                                    power_stat=False,
                                    max_iter=400,
                                    lam=1e-1,
                                    step_size=1e-2,
                                    batch_proportion=1.0,
                                    ftol=2.220446049250313e-09,
                                    lam_var=1e-8,
                                    **kwargs):
        from multiprocessing import cpu_count
        from joblib import Parallel, delayed

        def trial(data_inp, nx, func_z,
                kernel=kernel,
                max_iter=max_iter, 
                lam=lam, 
                step_size=step_size,
                batch_proportion=batch_proportion,
                ftol=ftol,
                verbose=False):
            data = data_inp.copy()
            np.random.shuffle(data)
            info = find_weights_sp(data[:nx],data[nx:], func_z,
                               kernel=kernel,
                               max_iter=max_iter, 
                               lam=lam, 
                               step_size=step_size,
                               batch_proportion=batch_proportion,
                               ftol=ftol,
                               verbose=False)
            return info['weights']
            

        data = tst_data.stack_xy()
        nx,d = tst_data.X.shape
        func_z = functools.partial(LinearMMDTest.test_power,alpha=alpha,lam_var=lam_var)

        weights_arr = Parallel(n_jobs=cpu_count(),verbose=5)(delayed(trial)(data,nx,func_z) for i in range(n_permutations))
        weights_arr = np.array(weights_arr)

        if compute_thresholds:
            return np.percentile(weights_arr, 100*(1-alpha_feat),axis=0)
        else:
            return weights_arr
        
    @staticmethod
    def p_values_uni(tst_data, w,
                    alpha=0.01, 
                    kernel=None, 
                    repeats=1000, 
                    lam_var=1e-8,
                    **kwargs):
        X,Y = tst_data.xy()
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        w = torch.from_numpy(w)

        mmd2,mmd2_2nd = LinearMMDTest.two_moments(w*X, w*Y, kernel)
        mmd2_var = mmd2_2nd - mmd2**2

        p_values = np.zeros(len(w))
        for i in tqdm(range(len(w)),leave=False,desc='Univariate p-value of features'):
            w_use = w.clone()
            w_use[i] = 0.0 # drop a feature by setting its weight to 0
            mmd2_test,_ = LinearMMDTest.two_moments(w_use*X, w_use*Y, kernel)
            p_values[i] = norm.cdf(mmd2_test.item(), loc=mmd2, scale=np.sqrt(mmd2_var))
        return p_values
    
    @staticmethod
    def select_features(tst_data,w,alpha,
                        kernel=None,
                        pruning=False,
                        feature_percentage=0.1,
                        **kwargs):
        X,Y = tst_data.xy()
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        w = torch.from_numpy(w)
        w_use = w.clone()
        if pruning:
            effective_features = prune_features(w, feature_percentage)
        else:
            effective_features = range(len(w))

        n = X.shape[0]
        mmd2,mmd2_2nd = LinearMMDTest.two_moments(w*X, w*Y, kernel)
        c_alpha = norm.isf(alpha,loc=0,scale=(2.0*mmd2_2nd/n).detach().cpu()**0.5)

        mmd2_drops = torch.inf*torch.ones(len(w))
        for i in tqdm(effective_features,leave=False,desc='Finding lowest MMD'):
            w_tmp = w.clone()
            w_tmp[i] = 0.0 # drop a feature by setting its weight to 0
            mmd2_test,_ = LinearMMDTest.two_moments(w_tmp*X, w_tmp*Y, kernel)
            mmd2_drops[i] = mmd2_test-mmd2

        chosen_features = []
        run_flag = (mmd2 > c_alpha)
        while run_flag:
            min_idx = np.argmin(mmd2_drops)
            w_use[min_idx] = 0.0 # drop a feature by setting its weight to 0
            mmd2_drops[min_idx] = torch.inf # set infinite so this feature will never flag up again.

            mmd2_test,_ = LinearMMDTest.two_moments(w_use*X, w_use*Y, kernel)
            run_flag = (mmd2_test > c_alpha) and torch.any(mmd2_drops!=torch.inf).item()

            # Append feature index and p-value
            chosen_features.append(min_idx)
        return np.array(chosen_features)

class QuadMMDTest(TwoSampleTest):
    """
    Quadratic MMD test where the null distribution is computed by permutation.
    - Use a single U-statistic i.e., remove diagonal from the Kxy matrix.
    - The code is based on a Matlab code of Arthur Gretton from the paper
    A TEST OF RELATIVE SIMILARITY FOR MODEL SELECTION IN GENERATIVE MODELS
    ICLR 2016
    """

    def __init__(self, kernel, n_permute=400, alpha=0.01, use_1sample_U=False):
        """
        kernel: an instance of Kernel
        n_permute: number of times to do permutation
        """
        self.kernel = kernel
        self.n_permute = n_permute
        self.alpha = alpha
        self.use_1sample_U = use_1sample_U

    def perform_test(self, tst_data, w=None):
        """perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        tst_data: an instance of TSTData
        """
        d = tst_data.dim()
        X, Y = tst_data.xy()
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        if w is None:
            w = 1
        else:
            w = torch.from_numpy(w)


        alpha = self.alpha
        k = self.kernel
        repeats = self.n_permute

        mmd2_stat = self.compute_stat(w*X,w*Y, k, use_1sample_U=self.use_1sample_U)

        list_mmd2 = QuadMMDTest.permutation_list_mmd2(w*X, w*Y, k, repeats)
        # approximate p-value with the permutations
        pvalue = np.mean((list_mmd2 > mmd2_stat).numpy())
        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': mmd2_stat.item(),
                'h0_rejected': pvalue < alpha, 'c_alpha': np.percentile(list_mmd2.numpy(),100*(1-alpha))}
        return results

    @staticmethod
    def compute_stat(X, Y, k, use_1sample_U=True, power_stat=False, power_stat_full=False, func_var=None, lam_var=1e-8, alpha=0.01):
        """Compute the test statistic: empirical quadratic MMD^2"""
        nx = X.shape[0]
        ny = Y.shape[0]

        if nx != ny:
            raise ValueError('nx must be the same as ny')

        if not power_stat:
            mmd2 = QuadMMDTest.h1_mean_var(X, Y, k, is_var_computed=False,
                    use_1sample_U=use_1sample_U)
            return mmd2
        elif power_stat and not power_stat_full:
            mmd2, var = QuadMMDTest.h1_mean_var(X, Y, k, is_var_computed=True,
                    use_1sample_U=use_1sample_U,func_var=func_var,lam_var=lam_var)
            return mmd2/torch.sqrt(var)
        else:
            mmd2, var = QuadMMDTest.h1_mean_var(X, Y, k, is_var_computed=True,
                    use_1sample_U=use_1sample_U,func_var=func_var,lam_var=lam_var)
            perm_list = QuadMMDTest.permutation_list_mmd2(X, Y, k)
            c_alpha = np.percentile(perm_list,100*alpha)/nx
            return (mmd2-c_alpha)/torch.sqrt(var)

    @staticmethod
    def permutation_list_mmd2(X, Y, k, n_permute=400, seed=8273):
        """
        Repeatedly mix, permute X,Y and compute MMD^2. This is intended to be
        used to approximate the null distritubion.
        """
        XY = torch.cat((X, Y),dim=0)
        Kxyxy = k.eval_torch(XY, XY)

        rand_state = np.random.get_state()
        np.random.seed(seed)

        nxy = XY.shape[0]
        nx = X.shape[0]
        ny = Y.shape[0]

        list_mmd2 = torch.zeros(n_permute)

        for r in range(n_permute):
            #print r
            ind = np.random.choice(nxy, nxy, replace=False)
            # divide into new X, Y
            indx = ind[:nx]
            #print(indx)
            indy = ind[nx:]
            Kx = Kxyxy[np.ix_(indx, indx)]
            #print(Kx)
            Ky = Kxyxy[np.ix_(indy, indy)]
            Kxy = Kxyxy[np.ix_(indx, indy)]

            mmd2r = QuadMMDTest.h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed=False, use_1sample_U=True)
            list_mmd2[r] = mmd2r

        np.random.set_state(rand_state)
        return list_mmd2

    @staticmethod
    def h1_mean_var(X, Y, k, is_var_computed, use_1sample_U=True, func_var=None, lam_var=1e-8):
        """
        X: nxd numpy array
        Y: nxd numpy array
        k: a Kernel object
        is_var_computed: if True, compute the variance. If False, return None.
        use_1sample_U: if True, use one-sample U statistic for the cross term
          i.e., k(X, Y).

        Code based on Arthur Gretton's Matlab implementation for
        Bounliphone et. al., 2016.

        return (MMD^2, var[MMD^2]) under H1
        """

        nx = X.shape[0]
        ny = Y.shape[0]

        Kx = k.eval_torch(X, X)
        Ky = k.eval_torch(Y, Y)
        Kxy = k.eval_torch(X, Y)

        return QuadMMDTest.h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U, func_var=func_var, lam_var=lam_var)

    @staticmethod
    def h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U=True, func_var=None, lam_var=1e-8):
        """
        Same as h1_mean_var() but takes in Gram matrices directly.
        """

        nx = Kx.shape[0]
        ny = Ky.shape[0]
        xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))),(nx*(nx-1)))
        yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))),(ny*(ny-1)))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy) - torch.sum(torch.diag(Kxy))),(nx*(ny-1)))
        else:
            xy = torch.div(torch.sum(Kxy),(nx*ny))
        mmd2 = xx - 2*xy + yy

        if not is_var_computed:
            return mmd2
        else:
            return mmd2, func_var(Kx,Ky,Kxy,lam_var=lam_var)
        
    @staticmethod
    def compute_var_old(Kx, Ky, Kxy, lam_var=1e-8):
        # compute the variance; code is a torch version of Jitkrittum's
        Kxd = Kx - torch.diag(torch.diag(Kx))
        Kyd = Ky - torch.diag(torch.diag(Ky))
        m = Kx.shape[0]
        n = Ky.shape[0]
        v = torch.zeros(11)

        Kxd_sum = torch.sum(Kxd)
        Kyd_sum = torch.sum(Kyd)
        Kxy_sum = torch.sum(Kxy)
        Kxy2_sum = torch.sum(Kxy**2)
        Kxd0_red = torch.sum(Kxd, 1)
        Kyd0_red = torch.sum(Kyd, 1)
        Kxy1 = torch.sum(Kxy, 1)
        Kyx1 = torch.sum(Kxy, 0)

        v[0] = 1.0/m/(m-1)/(m-2)*( torch.matmul(Kxd0_red, Kxd0_red ) - torch.sum(Kxd**2) )
        v[1] = -( 1.0/m/(m-1) * Kxd_sum )**2
        v[2] = -2.0/m/(m-1)/n * torch.matmul(Kxd0_red, Kxy1)
        v[3] = 2.0/(m**2)/(m-1)/n * Kxd_sum*Kxy_sum
        v[4] = 1.0/n/(n-1)/(n-2)*( torch.matmul(Kyd0_red, Kyd0_red) - torch.sum(Kyd**2 ) )
        v[5] = -( 1.0/n/(n-1) * Kyd_sum )**2
        v[6] = -2.0/n/(n-1)/m * torch.matmul(Kyd0_red, Kyx1)
        v[7] = 2.0/(n**2)/(n-1)/m * Kyd_sum*Kxy_sum
        v[8] = 1.0/n/(n-1)/m * ( torch.matmul(Kxy1, Kxy1) - Kxy2_sum )
        v[9] = -2.0*( 1.0/n/m*Kxy_sum )**2
        v[10] = 1.0/m/(m-1)/n * ( torch.matmul(Kyx1, Kyx1) - Kxy2_sum )


        #%additional low order correction made to some terms compared with ICLR submission
        #%these corrections are of the same order as the 2nd order term and will
        #%be unimportant far from the null.

        #   %Eq. 13 p. 11 ICLR 2016. This uses ONLY first order term
        #   varEst = 4*(m-2)/m/(m-1) *  varEst  ;
        varEst1st = 4.0*(m-2)/m/(m-1) * torch.sum(v)

        Kxyd = Kxy - torch.diag(torch.diag(Kxy))
        #   %Eq. 13 p. 11 ICLR 2016: correction by adding 2nd order term
        #   varEst2nd = 2/m/(m-1) * 1/n/(n-1) * sum(sum( (Kxd + Kyd - Kxyd - Kxyd').^2 ));
        varEst2nd = 2.0/m/(m-1) * 1/n/(n-1) * torch.sum( (Kxd + Kyd - Kxyd - Kxyd.T)**2)

        #   varEst = varEst + varEst2nd;
        varEst = varEst1st + varEst2nd

        #   %use only 2nd order term if variance estimate negative
        if varEst<0:
            varEst = varEst2nd

        return varEst + lam_var

    @staticmethod
    def compute_var_biased(Kx, Ky, Kxy, lam_var=1e-8):
        '''
        From: https://arxiv.org/pdf/2002.09116.pdf
        Biased but simple approach. Might have its merits. See footnote 2 on page 3.
        '''
        assert Kxy.shape[0]==Kxy.shape[1]
        n = Kx.shape[0]
        H_red = (Kx+Ky-2*Kxy).sum(axis=1)
        v2 = (4/n**3)*(H_red**2).sum()-(4/n**4)*(H_red.sum())**2
        return v2 + lam_var

    @staticmethod
    def compute_var_unbiased(Kx, Ky, Kxy, lam_var=1e-8):
        '''
        From: https://arxiv.org/pdf/1611.04488.pdf, equation (5)
        Uniased but complicated approach.
        '''
        assert Kxy.shape[0]==Kxy.shape[1]
        m = Kx.shape[0]

        Kxt = Kx-torch.diag(torch.diag(Kx))
        Kyt = Ky-torch.diag(torch.diag(Ky))
        Kxd = Kxt.sum(dim=1,keepdim=True)
        Kyd = Kyt.sum(dim=1,keepdim=True)

        v2 = torch.zeros(8)
        v2[0] = (4/m_r(m,4))*((Kxd**2).sum()+(Kyd**2).sum())
        v2[1] = (4*(m**2-m-1))/(m**3*(m-1)**2)*((Kxy.sum(dim=1)**2).sum()+(Kxy.permute(1,0).sum(dim=1)**2).sum())
        v2[2] = -8/(m**2*(m**2-3*m+2))*(Kxd.permute(1,0).matmul(Kxy.sum(axis=1))+Kyd.permute(1,0).matmul(Kxy.permute(1,0).sum(axis=1)))
        v2[3] = 8/(m**2*m_r(m,3))*(Kxd.sum()+Kyd.sum())*Kxy.sum()
        v2[4] = -(2*(2*m-3))/(m_r(m,2)*m_r(m,4))*(Kxd.sum()**2+Kyd.sum()**2)
        v2[5] = -(4*(2*m-3))/(m**3*(m-1)**3)*Kxy.sum()**2
        v2[6] = - 2/(m*(m**3-6*m**2+11*m-6))*((Kxt**2).sum()+(Kyt**2).sum())
        v2[7] = (4*(m-2))/(m**2*(m-1)**3)*(Kxy**2).sum()
        return v2.sum()+lam_var

    @staticmethod
    def median_distance(X,Y,p=2):
        # Cast into torch format
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        # Drop duplicated points to avoid distorting the median
        # Usually this is for all 0 points
        X = torch.unique(X,dim=0)
        Y = torch.unique(Y,dim=0)

        distances = torch.cdist(X,Y,p=p)
        # indices = torch.triu_indices(*distances.shape,offset=1)
        # median_distance = distances[indices].view(-1).quantile(0.5)
        median_distance = distances.view(-1).quantile(0.5)
        return 2*median_distance.item()**2

    @staticmethod
    def grid_search_kernel(X,Y, list_kernels, alpha, func_var=None):
        """
        Return from the list the best kernel that maximizes the test power criterion.

        In principle, the test threshold depends on the null distribution, which
        changes with kernel. Thus, we need to recompute the threshold for each kernel
        (require permutations), which is expensive. However, asymptotically
        the threshold goes to 0. So, for each kernel, the criterion needed is
        the ratio mean/variance of the MMD^2. (Source: Arthur Gretton)
        This is an approximate to avoid doing permutations for each kernel
        candidate.
        """
        # Cast into torch format
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        n = X.shape[0]
 
        powers = torch.zeros(len(list_kernels))
        for ki, k in enumerate(list_kernels):
            power = QuadMMDTest.compute_stat(X, Y, k, power_stat=True, func_var=func_var, lam_var=0.0)
            powers[ki] = power
        best_ind = torch.argmax(powers)
        return best_ind, powers.numpy()

    @staticmethod
    def optimize_w(tst_data, alpha=0.01,
                             kernel=None,
                             optimise_gwidth = False,
                             power_stat=False,
                             power_stat_full=False,
                             func_var=None,
                             max_iter=400,
                             lam=1e-1,
                             step_size=1e-2,
                             batch_proportion=1.0,
                             ftol=2.220446049250313e-09,
                             lam_var=1e-8,
                             verbose=True):
        
        if type(kernel).__name__=='KGauss' and optimise_gwidth:
            list_kernels = [kernel_torch.KGauss(sigma2=s) for s in kernel.sigma2*2**np.linspace(-4,4,30)]
            kernel = QuadMMDTest.grid_search_kernel(test_data.X, tst_data.Y, list_kernels, alpha=alpha)

        if func_var is None:
            func_var = QuadMMDTest.compute_var_old
        func_z = functools.partial(QuadMMDTest.compute_stat,power_stat=power_stat,power_stat_full=power_stat_full,func_var=func_var,lam_var=lam_var,alpha=alpha)

        # info = optimization info
        info = find_weights_sp(tst_data.X,tst_data.Y, func_z,
                               kernel=kernel,
                               max_iter=max_iter, 
                               lam=lam, 
                               step_size=step_size,
                               batch_proportion=batch_proportion,
                               ftol=ftol,
                               verbose=verbose)
        return info
    
    @staticmethod
    def significance_thresholds_perm(tst_data,
                                    compute_thresholds = False,
                                    alpha_feat = 0.01,
                                    n_permutations=400,
                                    alpha=0.01,
                                    kernel=None,
                                    power_stat=False,
                                    power_stat_full=False,
                                    func_var=None,
                                    max_iter=400,
                                    lam=1e-1,
                                    step_size=1e-2,
                                    batch_proportion=1.0,
                                    ftol=2.220446049250313e-09,
                                    lam_var=1e-8,
                                    **kwargs):
        from multiprocessing import cpu_count
        from joblib import Parallel, delayed

        def trial(data_inp, nx, func_z,
                kernel=kernel,
                max_iter=max_iter, 
                lam=lam, 
                step_size=step_size,
                batch_proportion=batch_proportion,
                ftol=ftol,
                verbose=False):
            data = data_inp.copy()
            np.random.shuffle(data)
            info = find_weights_sp(data[:nx],data[nx:], func_z,
                               kernel=kernel,
                               max_iter=max_iter, 
                               lam=lam, 
                               step_size=step_size,
                               batch_proportion=batch_proportion,
                               ftol=ftol,
                               verbose=False)
            return info['weights']
            

        data = tst_data.stack_xy()
        nx,d = tst_data.X.shape

        if func_var is None:
            func_var = QuadMMDTest.compute_var_old
        func_z = functools.partial(QuadMMDTest.compute_stat,power_stat=power_stat,power_stat_full=power_stat_full,func_var=func_var,lam_var=lam_var,alpha=alpha)

        weights_arr = Parallel(n_jobs=cpu_count(),verbose=5)(delayed(trial)(data,nx,func_z) for i in range(n_permutations))
        weights_arr = np.array(weights_arr)

        if compute_thresholds is None:
            return np.percentile(weights_arr, 100*(1-alpha_feat),axis=0)
        else:
            return weights_arr
        
    @staticmethod
    def p_values_uni(tst_data, w,
                    kernel=None, 
                    use_1sample_U=True, 
                    func_var=None, 
                    lam_var=1e-8,
                    **kwargs):
        X,Y = tst_data.xy()
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        w = torch.from_numpy(w)

        if func_var is None:
            func_var = QuadMMDTest.compute_var_old

        mmd2,mmd2_var = QuadMMDTest.h1_mean_var(w*X, w*Y, kernel, is_var_computed=True,
                    use_1sample_U=use_1sample_U,func_var=func_var,lam_var=lam_var)

        p_values = np.zeros(len(w))
        for i in tqdm(range(len(w)),leave=False,desc='Univariate p-value of features'):
            w_use = w.clone()
            w_use[i] = 0.0 # drop a feature by setting its weight to 0
            mmd2_test = QuadMMDTest.compute_stat(w_use*X, w_use*Y, kernel, use_1sample_U=use_1sample_U)
            p_values[i] = norm.cdf(mmd2_test.item(), loc=mmd2, scale=np.sqrt(mmd2_var))
        return p_values

    @staticmethod
    def select_features(tst_data,w,alpha,
                        kernel=None, 
                        use_1sample_U=True,
                        pruning=False,
                        feature_percentage=0.1, 
                        **kwargs):
        X,Y = tst_data.xy()
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        w = torch.from_numpy(w)
        w_use = w.clone()
        if pruning:
            effective_features = prune_features(w, feature_percentage)
        else:
            effective_features = range(len(w))

        mmd2 = QuadMMDTest.compute_stat(w*X, w*Y, kernel, use_1sample_U=use_1sample_U)
        c_alpha = np.percentile(QuadMMDTest.permutation_list_mmd2(w*X, w*Y, kernel).numpy(),100*(1-alpha))
        mmd2_drops = torch.inf*torch.ones(len(w))
        for i in tqdm(effective_features,leave=False,desc='Finding lowest MMD'):
            w_tmp = w.clone()
            w_tmp[i] = 0.0 # drop a feature by setting its weight to 0
            mmd2_test = QuadMMDTest.compute_stat(w_tmp*X, w_tmp*Y, kernel, use_1sample_U=use_1sample_U)
            mmd2_drops[i] = mmd2_test-mmd2

        chosen_features = []
        run_flag = (mmd2 > c_alpha)
        while run_flag:
            min_idx = np.argmin(mmd2_drops)
            w_use[min_idx] = 0.0 # drop a feature by setting its weight to 0
            mmd2_drops[min_idx] = torch.inf # set infinite so this feature will never flag up again.

            mmd2_test = QuadMMDTest.compute_stat(w_use*X, w_use*Y, kernel, use_1sample_U=use_1sample_U)
            run_flag = (mmd2_test > c_alpha) and torch.any(mmd2_drops!=torch.inf).item()

            # Append feature index and p-value
            chosen_features.append(min_idx)
        return np.array(chosen_features)