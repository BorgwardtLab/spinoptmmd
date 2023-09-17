'''
This file contains the source code for the dataset classese.
They are refactored versions of:  https://github.com/wittawatj/interpretable-test
'''

import numpy as np
from src.utils import tr_te_indices,subsample_ind

class TSTData(object):
    """Class representing data for two-sample test"""

    """
    properties:
    X, Y: numpy array
    """

    def __init__(self, X, Y, label=None):
        """
        :param X: n x d numpy array for dataset X
        :param Y: n x d numpy array for dataset Y
        """
        self.X = X
        self.Y = Y
        # short description to be used as a plot label
        self.label = label

        nx, dx = X.shape
        ny, dy = Y.shape

        #if nx != ny:
        #    raise ValueError('Data sizes must be the same.')
        if dx != dy:
            raise ValueError('Dimension sizes of the two datasets must be the same.')

    def __str__(self):
        mean_x = np.mean(self.X, 0)
        std_x = np.std(self.X, 0)
        mean_y = np.mean(self.Y, 0)
        std_y = np.std(self.Y, 0)
        prec = 4
        desc = ''
        desc += 'E[x] = %s \n'%(np.array_str(mean_x, precision=prec ) )
        desc += 'E[y] = %s \n'%(np.array_str(mean_y, precision=prec ) )
        desc += 'Std[x] = %s \n' %(np.array_str(std_x, precision=prec))
        desc += 'Std[y] = %s \n' %(np.array_str(std_y, precision=prec))
        return desc

    def dimension(self):
        """Return the dimension of the data."""
        dx = self.X.shape[1]
        return dx

    def dim(self):
        """Same as dimension()"""
        return self.dimension()

    def stack_xy(self):
        """Stack the two datasets together"""
        return np.vstack((self.X, self.Y))

    def xy(self):
        """Return (X, Y) as a tuple"""
        return (self.X, self.Y)

    def mean_std(self):
        """Compute the average standard deviation """

        #Gaussian width = mean of stds of all dimensions
        X, Y = self.xy()
        stdx = np.mean(np.std(X, 0))
        stdy = np.mean(np.std(Y, 0))
        mstd = old_div((stdx + stdy),2.0)
        return mstd
        #xy = self.stack_xy()
        #return np.mean(np.std(xy, 0)**2.0)**0.5

    def split_tr_te(self, tr_proportion=0.5, seed=820):
        """Split the dataset into training and test sets. Assume n is the same
        for both X, Y.

        Return (TSTData for tr, TSTData for te)"""
        X = self.X
        Y = self.Y
        nx, dx = X.shape
        ny, dy = Y.shape
        if nx != ny:
            raise ValueError('Require nx = ny')
        Itr, Ite = tr_te_indices(nx, tr_proportion, seed)
        label = '' if self.label is None else self.label
        tr_data = TSTData(X[Itr, :], Y[Itr, :], 'tr_' + label)
        te_data = TSTData(X[Ite, :], Y[Ite, :], 'te_' + label)
        return (tr_data, te_data)

    def subsample(self, n, seed=87):
        """Subsample without replacement. Return a new TSTData """
        if n > self.X.shape[0] or n > self.Y.shape[0]:
            raise ValueError('n should not be larger than sizes of X, Y.')
        ind_x = subsample_ind( self.X.shape[0], n, seed )
        ind_y = subsample_ind( self.Y.shape[0], n, seed )
        return TSTData(self.X[ind_x, :], self.Y[ind_y, :], self.label)

    ### end TSTData class

class SSBlobs(object):
    # Adapted from Jitkrittum et al.
    """Mixture of 2d Gaussians arranged in a 2d grid. This dataset is used
    in Chwialkovski et al., 2015 as well as Gretton et al., 2012.
    Part of the code taken from Dino Sejdinovic and Kacper Chwialkovski's code."""

    def __init__(self, blob_distance=5, num_blobs=4, stretch=2, angle=(np.pi/4.0)):
        self.blob_distance = blob_distance
        self.num_blobs = num_blobs
        self.stretch = stretch
        self.angle = angle

    def dim(self):
        return 2

    def sample(self, n, seed):
        rstate = np.random.get_state()
        np.random.seed(seed)

        x = SSBlobs.gen_blobs(stretch=1, angle=0, blob_distance=self.blob_distance,
                num_blobs=self.num_blobs, num_samples=n)

        y = SSBlobs.gen_blobs(stretch=self.stretch, angle=self.angle,
                blob_distance=self.blob_distance, num_blobs=self.num_blobs,
                num_samples=n)

        np.random.set_state(rstate)
        return TSTData(x, y, label='blobs_s%d'%seed)

    @staticmethod
    def gen_blobs(stretch, angle, blob_distance, num_blobs, num_samples):
        """Generate 2d blobs dataset """

        # rotation matrix
        r = np.array( [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]] )
        eigenvalues = np.diag(np.array([np.sqrt(stretch), 1]))
        mod_matix = np.dot(r, eigenvalues)
        mean = (float(blob_distance * (num_blobs-1))/ 2)
        mu = np.random.randint(0, num_blobs,(num_samples, 2))*blob_distance - mean
        return np.random.randn(num_samples,2).dot(mod_matix) + mu

