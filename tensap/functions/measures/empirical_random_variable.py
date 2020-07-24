'''
Module empirical_random_variable.

Copyright (c) 2020, Anthony Nouy, Erwan Grelier
This file is part of tensap (tensor approximation package).

tensap is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

tensap is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with tensap.  If not, see <https://www.gnu.org/licenses/>.

'''

import scipy
import numpy as np
import tensap


class EmpiricalRandomVariable(tensap.RandomVariable):
    '''
    Class EmpiricalRandomVariable. A random variable fitted using gaussian
    kernel smoothing, this class gives best results in the case of normal
    distributions.

    Attributes
    ----------
    sample : numpy.array
        The sample used to generate the random variable.
    bandwidth : float
        The computed bandwidth for the kernel density estimator.

    '''

    def __init__(self, sample):
        '''
        Constructor for the class EmpiricalRandomVariable.

        A sample must be provided, which is used to fit a probability density
        function using Scott's rule.

        Parameters
        ----------
        sample : numpy.ndarray or list
            The sample used used to fit the probability density function using
            Scott's rule.

        Returns
        -------
        None.

        '''
        tensap.RandomVariable.__init__(self)

        self.sample = sample
        #  Scott's rule [Scott, D.W. (1992) Multivariate Density Estimation.
        # Theory, Practice and Visualization. New York: Wiley.]
        self.bandwidth = 3.5 * np.std(sample) * np.size(sample) ** (-1/3)

    def shift(self, b, s):
        '''
        Shift the random variable using the provided bias and scaling factor.

        Parameters
        ----------
        bias : float
            The bias.
        scaling : float
            The scaling factor.

        Returns
        -------
        tensap.EmpiricalRandomVariable
            The shifted random variable.

        '''
        return EmpiricalRandomVariable(s*self.sample+b)

    def transfer(self, Y, x):
        assert isinstance(Y, tensap.RandomVariable), \
            'The first argument must be a RandomVariable.'

        # If Y is the standard random variable associated to X
        if isinstance(Y, (EmpiricalRandomVariable,
                      tensap.EmpiricalRandomVariable)) and \
            np.linalg.norm(Y.sample - (self.sample - np.mean(self.sample)) /
                           np.std(self.sample)) / np.linalg.norm(Y.sample) < \
                np.finfo(float).eps:
            y = (x - np.mean(self.sample)) / np.std(self.sample)
        else:
            y = Y.icdf(self.cdf(x))
        return y

    def get_standard_random_variable(self):
        '''
        Return the standard empirical random variable with zero mean and
        unit standard deviation.

        Returns
        -------
        tensap.EmpiricalRandomVariable
            The standard empirical random variable.

        '''
        x = np.reshape(self.sample, [-1, 1])
        x = (x - np.tile(np.mean(x, 0), (x.shape[0], 1))) / \
            np.tile(np.std(x, 0), (x.shape[0], 1))
        return EmpiricalRandomVariable(np.ravel(x))

    def support(self):
        return np.array([-np.inf, np.inf])

    def orthonormal_polynomials(self, *args):
        p = tensap.EmpiricalPolynomials(self, *args)
        m = np.mean(self.sample)
        s = np.std(self.sample)
        if np.abs(m) > np.finfo(float).eps or \
                np.abs(s-1) > np.finfo(float).eps:
            p = tensap.ShiftedOrthonormalPolynomials(p, m, s)
        return p

    def get_parameters(self):
        return self.sample, self.bandwidth

    def cdf(self, x):
        kde = scipy.stats.gaussian_kde(self.sample, self.bandwidth)
        return np.vectorize(lambda z: kde.integrate_box_1d(-np.inf, z))(x)

    def icdf(self, x, *args, **kwargs):
        '''
        Compute the inverse cumulative distribution function (icdf) of the
        RandomVariable at points x.

        Parameters
        ----------
        x : float or list or numpy.ndarray
            The points at which the icdf is to be evaluated.
        *args, **kwargs : tuples
            Additional parameters for scipy.optimize's function brentq.

        Returns
        -------
        numpy.ndarray
            The evaluations of the icdf at points x.

        '''
        # TODO Optimize the method
        sup = self.truncated_support()
        return np.vectorize(lambda p: scipy.optimize.brentq(
            lambda y: self.cdf(y)-p, sup[0], sup[1], *args, **kwargs))(x)

    def pdf(self, x):
        xi, h = self.get_parameters()
        n = xi.size

        X = np.tile(np.reshape(x, [1, -1]), (n, 1))
        XI = np.tile(np.reshape(xi, [-1, 1]), (1, np.size(x)))
        return np.ravel(1/(n*h*np.sqrt(2*np.pi)) *
                        np.sum(np.exp(-0.5*(X-XI)**2/h**2), 0))

    def random_variable_statistics(self):
        return np.mean(self.sample), np.var(self.sample)

    def random(self, n=1):
        return self.icdf(np.random.rand(int(n)))
