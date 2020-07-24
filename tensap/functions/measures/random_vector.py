'''
Module random_vector.

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

from copy import deepcopy
import numpy as np
import tensap


class RandomVector(tensap.ProbabilityMeasure):
    '''
    Class RandomVector.

    Attributes
    ----------
    random_variables : numpy.ndarray
        The RandomVariable objects constituting the random vector.
    copula : tensap.Copula
        The copula of the random vector.

    '''

    def __init__(self, random_variables, order=None,
                 copula=tensap.IndependentCopula()):
        if isinstance(random_variables, tensap.RandomVector):
            random_variables = random_variables.random_variables

        if order is not None and isinstance(random_variables,
                                            tensap.RandomVariable):
            random_variables = [random_variables] * order

        self.random_variables = np.atleast_1d(deepcopy(random_variables))
        self.copula = copula

    def __repr__(self):
        return ('<{}:{n}' +
                '{t}random_variables = {},{n}' +
                '{t}copula = {}>').format(self.__class__.__name__,
                                          self.random_variables,
                                          self.copula.__class__.__name__,
                                          t='\t', n='\n')

    def marginal(self, ind):
        if not isinstance(self.copula, tensap.IndependentCopula):
            raise NotImplementedError(
                'Not implemented for non IndenpendentCopula.')
        return RandomVector(self.random_variables[ind])

    def ndim(self):
        return np.sum([x.ndim() for x in self.random_variables])

    @property
    def size(self):
        '''
        Return the number of random variables constituting the RandomVector.

        Returns
        -------
        int
            The number of random variables constituting the RandomVector.

        '''
        return len(self.random_variables)

    def __eq__(self, rv_2):
        if not (isinstance(self, tensap.RandomVector) and
                isinstance(rv_2, tensap.RandomVector)):
            is_equal = False
        elif len(self.random_variables) != len(rv_2.random_variables):
            is_equal = False
        else:
            is_equal = True
            for ind in zip(self.random_variables, rv_2.random_variables):
                is_equal = is_equal and (ind[0] == ind[1])
        return is_equal

    def __neq__(self, rv_2):
        return not (self == rv_2)

    def support(self):
        assert isinstance(self.copula, tensap.IndependentCopula), \
            'Not implemented for non IndenpendentCopula.'
        return [x.support() for x in self.random_variables]

    def truncated_support(self):
        '''
        Return the truncated support of the random vector.

        Raises
        ------
        NotImplementedError
            If the copula is not an IndependentCopula.

        Returns
        -------
        sup : numpy.ndarray
            The truncated support of the random vector.

        '''
        assert isinstance(self.copula, tensap.IndependentCopula), \
            'Not implemented for non IndependentCopula.'
        return [x.truncated_support() for x in self.random_variables]

    def get_standard_random_vector(self):
        '''
        Return the standard RandomVector associated with self.

        Returns
        -------
        tensap.RandomVector
            The standard RandomVector.

        '''
        return RandomVector([x.get_standard_random_variable() for x in
                             self.random_variables])

    def iso_probabilistic_grid(self, n):
        '''
        Generate a grid of (n[0]-1) x ... x (n[d-1]-1) points
        (x_{i_1}^1, ..., x_{i_d}^d) such that the N = (n[0] x ... x(n[d-1]
        sets [x_{i_1-1}^1, x_{i_1}^1] x ... x [x_{i_d-1}^1, x_{i_d}^1] have
        the same probability p = 1/N.

        Parameters
        ----------
        n : int
            The number of points of the grid plus one in each or all the
            dimensions.

        Returns
        -------
        tensap.FullTensorGrid
            The iso-probabilistic grid.

        '''
        assert isinstance(self.copula, tensap.IndependentCopula), \
            'The method only works for independent copulas.'

        d = self.size
        if np.size(n) == 1 and n < 1:
            n = int(np.ceil(n**(-1/d)))

        if np.size(n) != d:
            n = np.full(d, n)

        b = [x.iso_probabilistic_grid(y) for x, y in zip(self.random_variables,
                                                         n)]
        return tensap.FullTensorGrid(b)

    def lhs_random(self, n):
        '''
        Latin Hypercube Sampling of the RandomVector of n points.

        Requires the package pyDOE.

        Parameters
        ----------
        n : int
            Number of points.

        Returns
        -------
        list
            List containing the coordinates of the Latin Hypercube Sampling in
            each dimension.

        '''
        from pyDOE import lhs
        A = lhs(self.size, samples=n)
        A = RandomVector(tensap.UniformRandomVariable(0, 1),
                         self.size).transfer(self, A)
        return [A[:, k] for k in range(self.size)]

    def pdf(self, x):
        '''
        Compute the probability density function (pdf) of each RandomVariable
        in self at points x, x must have self.ndim() columns.

        Parameters
        ----------
        x : list or numpy.ndarray
            The points at which the pdf is to be evaluated.

        Returns
        -------
        numpy.ndarray
            The evaluations of the pdf at points x.

        '''
        if isinstance(x, list):
            x = np.hstack([np.reshape(y, [-1, 1]) for y in x])

        x = np.atleast_2d(x)

        p = np.zeros(x.shape)
        for i in range(self.size):
            p[:, i] = np.ravel(self.random_variables[i].pdf(x[:, i]))
        p = np.prod(p, axis=1)

        if not isinstance(self.copula, tensap.IndependentCopula):
            u = np.zeros(x.shape)
            for i in range(self.size):
                u[:, i] = np.ravel(self.random_variables[i].cdf(x[:, i]))
            p = p * self.copula.pdf(u)

        return p

    def cdf(self, x):
        '''
        Compute the cumulative distribution function (cdf) at points x, x must
        have self.ndim() columns.

        Parameters
        ----------
        x : list or numpy.ndarray
            The points at which the cdf is to be evaluated.

        Returns
        -------
        numpy.ndarray
            The evaluations of the cdf at points x.

        '''
        if isinstance(x, list):
            x = np.hstack([np.reshape(y, [-1, 1]) for y in x])

        x = np.atleast_2d(x)

        u = np.zeros(x.shape)
        for i in range(self.size):
            u[:, i] = np.ravel(self.random_variables[i].cdf(x[:, i]))
        return self.copula.cdf(u)

    def orthonormal_polynomials(self, max_degree=None):
        '''
        Return the max_degree-1 first orthonormal polynomials associated with
        the RandomVector.

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
        if max_degree is None:
            return [x.orthonormal_polynomials() for x in self.random_variables]

        if np.isscalar(max_degree):
            max_degree = np.full(self.size, max_degree)

        assert len(max_degree) == self.size, 'Wrong size for max_degree.'

        return [x.orthonormal_polynomials(max_degree[ind]) for
                ind, x in enumerate(self.random_variables)]

    def mean(self):
        '''
        Return the mean of the random variable.

        Returns
        -------
        float
            The mean of the random variable.

        '''
        return np.atleast_1d([x.mean() for x in self.random_variables])

    def std(self):
        '''
        Return the standard deviation of the random variable.

        Returns
        -------
        float
            The standard deviation of the random variable.

        '''
        return np.atleast_1d([x.std() for x in self.random_variables])

    def variance(self):
        '''
        Return the variance of the random variable.

        Returns
        -------
        float
            The variance of the random variable.

        '''
        return np.atleast_1d([x.variance() for x in self.random_variables])

    def transfer(self, Y, x):
        '''
        Transfer from the tensap.RandomVector X to the tensap.RandomVector Y,
        at points x.

        Parameters
        ----------
        Y : tensap.RandomVector
            The target RandomVector of the transfer.
        x : list or numpy.ndarray
            The input points.

        Returns
        -------
        y : numpy.ndarray
            The transfered points.

        '''
        if isinstance(x, list):
            x = np.hstack([np.reshape(y, [-1, 1]) for y in x])

        x = np.atleast_2d(x)

        if isinstance(Y, tensap.RandomVariable):
            Y = RandomVector(Y, self.size)

        assert self.size == Y.size, \
            'The two RandomVector must have the same dimension.'

        y = np.zeros((x.shape[0], Y.size))
        for i in range(self.size):
            y[:, i] = np.ravel(self.random_variables[i].transfer(
                Y.random_variables[i], x[:, i]))
        return y

    def transpose(self, perm):
        '''
        Transpose (permute) the components of a random vector.

        Parameters
        ----------
        perm : list or numpy.array
            The permutation indices.

        Returns
        -------
        tensap.RandomVector
            The random vector with transposed (permuted) components.

        '''
        return RandomVector(self.random_variables[perm])

    def random(self, n=1):
        '''
        Generate n random numbers according to the distribution of the
        RandomVector.

        Parameters
        ----------
        n : int
            The number of random numbers generated.

        Returns
        -------
        numpy.ndarray
            The generated numbers.

        '''
        assert isinstance(self.copula, tensap.IndependentCopula), \
            'Not implemented for non IndenpendentCopula.'
        return np.transpose(np.array([x.random(n)
                                      for x in self.random_variables]))
