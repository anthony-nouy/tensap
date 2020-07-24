'''
Module normal_random_variable.

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

from numpy import inf, array
from scipy.stats import norm
from numpy.random import randn
import tensap


class NormalRandomVariable(tensap.RandomVariable):
    '''
    Class NormalRandomVariable.

    Attributes
    ----------
    moments : numpy.array
        The moments of the normal random variable (if computed).
    mu : float, optional
        The mean of the normal random variable. The default is 0.
    sigma : float, optional
        The standard deviation of the normal random variable. The default is 1.

    '''

    def __init__(self, mean=0, standard_deviation=1):
        '''
        Normal random variable with provided mean and standard deviation.

        Parameters
        ----------
        mean : float, optional
            The mean of the normal random variable. The default is 0.
        standard_deviation : float, optional
            The standard deviation of the normal random variable. The default
            is 1.

        Returns
        -------
        None.

        '''
        tensap.RandomVariable.__init__(self)

        self.mu = mean
        self.sigma = standard_deviation

    def __repr__(self):
        return ('<{}: mu = {}, sigma = {}>').format(
            self.__class__.__name__,
            self.mu,
            self.sigma,
            t='\t', n='\n')

    def cdf(self, x):
        '''
        Evaluate the cumulative distribution function (cdf) of the normal
        random variable at points x.

        Parameters
        ----------
        x : list or numpy.ndarray
            The points at which the cdf is to be evaluated.

        Returns
        -------
        numpy.ndarray
            The evaluations of the cdf.

        '''
        return norm.cdf(x, self.mu, self.sigma)

    def icdf(self, x):
        '''
        Evaluate the inverse cumulative distribution function (icdf) of the
        normal random variable at points x.

        Parameters
        ----------
        x : list or numpy.ndarray
            The points at which the icdf is to be evaluated.

        Returns
        -------
        numpy.ndarray
            The evaluations of the icdf.

        '''
        return norm.ppf(x, self.mu, self.sigma)

    def pdf(self, x):
        '''
        Evaluate the probability density function (pdf) of the normal
        random variable at points x.

        Parameters
        ----------
        x : list or numpy.ndarray
            The points at which the pdf is to be evaluated.

        Returns
        -------
        numpy.ndarray
            The evaluations of the pdf.

        '''
        return norm.pdf(x, self.mu, self.sigma)

    def shift(self, bias, scaling):
        '''
        Shift the normal random variable using the provided bias and scaling
        factor.

        The mean mu of the random variable becomes mu + bias, and its standard
        deviation sigma becomes sigma * scaling.

        Parameters
        ----------
        bias : float
            The bias.
        scaling : float
            The scaling factor.

        Returns
        -------
        RV : tensap.NormalRandomVariable
            The shifted normal random variable.

        '''
        shifted_rv = tensap.NormalRandomVariable(self.mu, self.sigma)
        shifted_rv.mu += bias
        shifted_rv.sigma *= scaling
        return shifted_rv

    @staticmethod
    def get_standard_random_variable():
        '''
        Return the standard normal random variable with mean 0 and standard
        deviation 1.

        Returns
        -------
        tensap.NormalRandomVariable
            The standard normal random variable.

        '''
        return tensap.NormalRandomVariable()

    @staticmethod
    def support():
        '''
        Return the support of the normal random variable.

        Returns
        -------
        numpy.ndarray
            Support of the normal random variable.

        '''
        return array([-inf, inf])

    def orthonormal_polynomials(self, *max_degree):
        '''
        Return the max_degree-1 first orthonormal polynomials associated with
        the NormalRandomVariable.

        Parameters
        ----------
        max_degree : int, optional
            The maximum degree of the returned polynomials. The default is
            None, choosing the default maximum degree associated with the
            constructor of the polynomials.

        Returns
        -------
        poly : tensap.OrthonormalPolynomials
            The generated orthonormal polynomials.

        '''
        poly = tensap.HermitePolynomials(*max_degree)
        if self != NormalRandomVariable(0, 1):
            # print('ShiftedOrthonormalPolynomials are created.')
            poly = tensap.ShiftedOrthonormalPolynomials(poly, self.mu,
                                                        self.sigma)
        return poly

    def get_parameters(self):
        '''
        Return the parameters of the normal random variable.

        Returns
        -------
        float
            The mean of the random variable.
        float
            The standard deviation of the random variable.

        '''
        return self.mu, self.sigma

    def random_variable_statistics(self):
        return self.mu, self.sigma**2

    def random(self, n):
        return randn(int(n)) * self.sigma + self.mu
