'''
Module uniform_random_variable.

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

from numpy import array
from scipy.stats import uniform
from numpy.random import rand
import tensap


class UniformRandomVariable(tensap.RandomVariable):
    '''
    Class UniformRandomVariable.

    Attributes
    ----------
    moments : numpy.array
        The moments of the uniform random variable (if computed).
    inf : float, optional
        The lower bound of the support of the random variable. The default
        is -1.
    sup : float, optional
        The upper bound of the support of the random variable. The default
        is 1.

    '''

    def __init__(self, inf=-1, sup=1):
        '''
        Uniform random variable on [inf, sup].

        Parameters
        ----------
        inf : float, optional
            The lower bound of the support of the random variable. The default
            is -1.
        sup : float, optional
            The upper bound of the support of the random variable. The default
            is 1.

        Returns
        -------
        None.

        '''
        tensap.RandomVariable.__init__(self)

        self.inf = inf
        self.sup = sup

    def __repr__(self):
        return ('<{} on [{}, {}]>').format(self.__class__.__name__,
                                           self.inf,
                                           self.sup)

    def cdf(self, x):
        '''
        Evaluate the cumulative distribution function (cdf) of the uniform
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
        return uniform.cdf(x, self.inf, self.sup - self.inf)

    def icdf(self, x):
        '''
        Evaluate the inverse cumulative distribution function (icdf) of the
        uniform random variable at points x.

        Parameters
        ----------
        x : list or numpy.ndarray
            The points at which the icdf is to be evaluated.

        Returns
        -------
        numpy.ndarray
            The evaluations of the icdf.

        '''
        return uniform.ppf(x, self.inf, self.sup - self.inf)

    def pdf(self, x):
        '''
        Evaluate the probability density function (pdf) of the uniform
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
        return uniform.pdf(x, self.inf, self.sup - self.inf)

    def shift(self, bias, scaling):
        '''
        Shift the uniform random variable using the provided bias and scaling
        factor.

        The lower bound inf becomes scaling*inf + bias, and the upper bound
        sup becomes scaling*sup + bias.

        Parameters
        ----------
        bias : float
            The bias.
        scaling : float
            The scaling factor.

        Returns
        -------
        shifted_rv : tensap.UniformRandomVariable
            The shifted uniform random variable.

        '''
        shifted_rv = tensap.UniformRandomVariable(self.inf, self.sup)
        shifted_rv.inf = scaling * shifted_rv.inf + bias
        shifted_rv.sup = scaling * shifted_rv.sup + bias
        return shifted_rv

    @staticmethod
    def get_standard_random_variable():
        '''
        Return the standard uniform random variable on [-1, 1].

        Returns
        -------
        tensap.UniformRandomVariable
            The standard uniform random variable.

        '''
        return tensap.UniformRandomVariable()

    def support(self):
        '''
        Return the support of the uniform random variable.

        Returns
        -------
        numpy.ndarray
            Support of the uniform random variable.

        '''
        return array([self.inf, self.sup])

    def orthonormal_polynomials(self, *max_degree):
        '''
        Return the max_degree-1 first orthonormal polynomials associated with
        the UniformRandomVariable.

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
        poly = tensap.LegendrePolynomials(*max_degree)
        if self != UniformRandomVariable(-1, 1):
            # print('ShiftedOrthonormalPolynomials are created.')
            poly = tensap.ShiftedOrthonormalPolynomials(poly,
                                                        (self.inf+self.sup)/2,
                                                        (self.sup-self.inf)/2)
        return poly

    def get_parameters(self):
        '''
        Return the parameters of the uniform random variable.

        Returns
        -------
        float
            The lower bound of the support of the random variable.
        float
            The upper bound of the support of the random variable.

        '''
        return self.inf, self.sup

    def random_variable_statistics(self):
        return (self.inf + self.sup)/2, (self.sup - self.inf)**2/12

    def random(self, n=1):
        return rand(int(n)) * (self.sup - self.inf) + self.inf
