'''
This module contains the kernel classes.
It is based on the kernel module of https://github.com/wittawatj/interpretable-test
We rewrote the kernel methods in the torch package and removed non-differentiable kernels
'''

from six import with_metaclass
from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.signal as sig
import scipy.spatial as spa
import torch

class Kernel(with_metaclass(ABCMeta, object)):
    """Abstract class for kernels"""

    @abstractmethod
    def eval(self, X1, X2):
        """Evalute the kernel on data X1 and X2 """
        pass

    @abstractmethod
    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ..."""
        pass

    @abstractmethod
    def eval_torch(self, X1, X2):
        """Evalute the kernel on data X1 and X2 """
        pass

    @abstractmethod
    def pair_eval_torch(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ..."""
        pass

class KHoPoly(Kernel):
    """Homogeneous polynomial kernel of the form
    (x.dot(y))**d
    """
    def __init__(self, degree):
        assert degree > 0
        self.degree = degree

    def eval(self, X1, X2):
        return np.dot(X1, X2.T)**self.degree

    def pair_eval(self, X, Y):
        return np.sum(X*Y, 1)**self.degree

    def eval_torch(self, X1, X2):
        return torch.matmul(X1, X2.permute(1,0))**self.degree

    def pair_eval_torch(self, X, Y):
        return torch.sum(X*Y, 1)**self.degree

    def __str__(self):
        return 'KHoPoly(d=%d)'%self.degree

class KLinear(Kernel):
    def eval(self, X1, X2):
        return np.dot(X1, X2.T)

    def pair_eval(self, X, Y):
        return np.sum(X*Y, 1)

    def eval_torch(self, X1, X2):
        return torch.matmul(X1, X2.permute(1,0))

    def pair_eval_torch(self, X, Y):
        return torch.sum(X*Y, dim=1)

    def __str__(self):
        return "KLinear()"

class KGauss(Kernel):

    def __init__(self, sigma2):
        assert sigma2 > 0, 'sigma2 must be > 0'
        self.sigma2 = sigma2

    def eval(self, X1, X2):
        """
        Evaluate the Gaussian kernel on the two 2d numpy arrays.

        Parameters
        ----------
        X1 : n1 x d numpy array
        X2 : n2 x d numpy array

        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        (n1, d1) = X1.shape
        (n2, d2) = X2.shape
        assert d1==d2, 'Dimensions of the two inputs must be the same'
        D2 = spa.distance.cdist(X1,X2)**2
        K = np.exp(-D2/self.sigma2)
        return K

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...

        Parameters
        ----------
        X, Y : n x d numpy array

        Return
        -------
        a numpy array with length n
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert n1==n2, 'Two inputs must have the same number of instances'
        assert d1==d2, 'Two inputs must have the same dimension'
        D2 = np.sum( (X-Y)**2, 1)
        Kvec = np.exp(-D2/self.sigma2)
        return Kvec

    def eval_torch(self, X1, X2):
        """
        Evaluate the Gaussian kernel on the two 2d numpy arrays.

        Parameters
        ----------
        X1 : n1 x d numpy array
        X2 : n2 x d numpy array

        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        (n1, d1) = X1.shape
        (n2, d2) = X2.shape
        assert d1==d2, 'Dimensions of the two inputs must be the same'
        D2 = torch.cdist(X1,X2)**2
        K = torch.exp(-D2/self.sigma2)
        return K

    def pair_eval_torch(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...

        Parameters
        ----------
        X, Y : n x d numpy array

        Return
        -------
        a numpy array with length n
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert n1==n2, 'Two inputs must have the same number of instances'
        assert d1==d2, 'Two inputs must have the same dimension'
        D2 = torch.sum( (X-Y)**2, dim=1)
        Kvec = torch.exp(-D2/self.sigma2)
        return Kvec

    def __str__(self):
        return "KGauss(w2=%.3f)"%self.sigma2