'''
Module functional_basis.

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

from abc import abstractmethod
from scipy.sparse import diags
import numpy as np
import tensap


class FunctionalBasis:
    '''
    Class FunctionalBasis.

    Attributes
    ----------
    measure : tensap.Measure
        The measure associated with the FunctionalBasis.

    is_orthonormal : bool
        Indicates if the basis is orthonormal with respect to the associated
        measure.

    '''

    def __init__(self):
        '''
        Constructor for the class FunctionalBasis.

        Returns
        -------
        None.

        '''
        self.measure = None
        self.is_orthonormal = False

    def __repr__(self):
        return ('<{}:{n}' +
                '{t}measure = {},{n}' +
                '{t}is_orthonormal = {}>').format(self.__class__.__name__,
                                                  self.measure,
                                                  self.is_orthonormal,
                                                  t='\t', n='\n')

    def random(self, n=1, measure=None):
        '''
        Evaluate the basis using n points drawn randomly according to measure
        if provided, or to self.measure otherwise.

        Parameters
        ----------
        n : int, optional
            The number of random evaluations. The default is 1.
        measure : tensap.ProbabilityMeasure, optional
            The probability measure used for the generation of the input
            points. The default is None, indicating to use self.measure.

        Returns
        -------
        basis_eval : numpy.ndarray
            Random evaluations of the basis functions.
        x : numpy.ndarray
            The input points.

        '''
        if measure is None:
            measure = self.measure
        if not isinstance(measure, tensap.ProbabilityMeasure):
            raise ValueError('Must provide a ProbabilityMeasure.')

        x = measure.random(n)
        basis_eval = np.reshape(self.eval(x), (n, self.cardinal()))
        return basis_eval, x

    @staticmethod
    def storage():
        '''
        Return the storage requirement of the FunctionalBasis.

        Returns
        -------
        int
            The storage requirement of the FunctionalBasis.

        '''
        return 0

    def adaptation_path(self):
        '''
        Return the adaptation path of the functional basis.

        Returns
        -------
        numpy.ndarray
            Boolean array, where n is the dimension of the functional basis,
            and m is the number of elements in the adaptation path,
            column P[:,i] corresponds to a sparsity pattern.

        '''
        return np.triu(np.full([self.cardinal()]*2, True))

    def interpolate(self, y, x=None):
        '''
        Provide an interpolation on a functional basis of a function (or values
        of the function) y associated with a set of n interpolation points x.

        Parameters
        ----------
        y : function or list or numpy.ndarray
            The function to interpolate, or values of it.
        x : list or numpy.ndarray, optional
            The interpolation points. The default is None, indicating to
            deduce them from the basis.

        Returns
        -------
        f : tensap.FunctionalBasisArray
            The computed interpolation.

        '''
        if x is None:
            x = self.interpolation_points()
        try:
            y = y(x)
        except Exception:
            pass

        if np.ndim(y) == 1:
            y = np.reshape(y, [-1, 1])

        hx = self.eval(x)
        data = np.linalg.solve(hx, y)
        f = tensap.FunctionalBasisArray(data, self, np.shape(y)[1])
        f.measure = self.measure
        return f

    @staticmethod
    def mean():
        '''
        Return the mean of the basis functions.

        Returns
        -------
        numpy.ndarray
            The mean of the basis functions.

        '''
        raise NotImplementedError('No generic implementation of the method.')

    def expectation(self):
        '''
        Return the expectation of the basis functions. Equivalent to
        self.mean().

        Returns
        -------
        numpy.ndarray
            The expectation of the basis functions.

        '''
        return self.mean()

    @staticmethod
    def conditional_expectation():
        '''
        Compute the conditional expectation of self with respect to
        the random variables dims (a subset of range(d)). The expectation
        with respect to other variables (in the complementary set of
        dims) is taken with respect to the probability measure given by
        tensap.RandomVector XdimsC if provided, or with respect to the
        probability measure associated with the corresponding bases of
        the function.

        Parameters
        ----------
        dims : numpy.ndarray
            The dimensions in which the expectation is computed.
        XdimsC: tensap.RandomVector, optional
            The random vector used for the computation of the conditional
            expectation. The default is None, indicating to use the
        probability measure associated with the basis.

        Returns
        -------
        f : tensap.FunctionalBasisArray
            The conditional expectation of the function.

        '''
        raise NotImplementedError('No generic implementation of the method.')

    @staticmethod
    def kron():
        '''
        Compute the Kronecker product of two bases. For two functional bases
        f_i, i = 1, ..., n and g_j, j = 1, ..., m, return a functional basis
        h_k, k = 1, ..., nm.

        Returns
        -------
        tensap.FunctionalBasis
            The obtained basis.

        '''
        NotImplementedError('Method not implemented.')

    def projection(self, fun, G):
        '''
        Compute the projection of the function fun onto the functional basis
        using the integration rule G.

        Parameters
        ----------
        fun : function or tensap.Function
            The function to project.
        G : tensap.IntegrationRule
            The integration rule used for the projection.

        Returns
        -------
        tensap.FunctionalBasisArray
            The projection of the function fun onto the functional basis using
            the integration rule G.

        '''

        A = self.eval(G.points)
        W = diags(G.weights)

        y = fun(G.points)
        if self.is_orthonormal:
            u = np.matmul(np.transpose(A), W.dot(y))
        else:
            u = np.linalg.solve(np.matmul(np.transpose(A), W.dot(A)),
                                np.matmul(np.transpose(A), W.dot(y)))
        if u.ndim == 1:
            u = np.reshape(u, [-1, 1])
        return tensap.FunctionalBasisArray(u, self, u.shape[1])

    def interpolation_points(self, *args):
        '''
        Return the interpolation points for the basis.

        See also FunctionalBasis.magic_points.

        Parameters
        ----------
        *args : tuple
            The inputs arguments of the method FunctionalBasis.magic_points.

        Returns
        -------
        numpy.ndarray
            The interpolation points for the basis.

        '''
        return self.magic_points(*args)[0]

    def magic_points(self, x=None, J=None):
        '''
        Provide the magic points associated with a functional basis f selected
        in a given set of points x.

        The method uses magicIndices(F,numel(f)) on the matrix F of evaluations
        of f at points x.

        Parameters
        ----------
        x : list or numpy.ndarray, optional
            The points used to construct the matrix F. The default is None,
            indicating to choose x automatically based on self.measure.
        J : numpy.ndarray, optional
            The default is None. If not none, selected the magic indices with
            tensap.magic_indices(F[:, J], self.cardinal(), 'left')[0]

        Returns
        -------
        points : numpy.ndarray
            The magic points.
        ind : numpy.ndarray
            The locations of the magic points in x.
        output : dict
            A dictionnary of outputs of the method.

        '''
        if x is None:
            if isinstance(self.measure, (tensap.DiscreteMeasure,
                                         tensap.DiscreteRandomVariable)):
                x = self.measure.values
            else:
                x = self.measure.random(self.cardinal()*100)

        if np.ndim(x) == 1:
            x = np.expand_dims(x, 1)

        assert x.shape[0] >= self.cardinal(), \
            ('The number of points must be higher than the number of basis ' +
             'functions.')

        F = self.eval(x)
        if J is not None:
            ind = tensap.magic_indices(F[:, J], self.cardinal(), 'left')[0]
        else:
            ind = tensap.magic_indices(F)[0]
        points = x[ind, :]
        if np.ndim(points) == 1:
            points = np.expand_dims(points, 1)

        # Estimation of the Lebesgue constant
        h_x = np.transpose(self.eval(points))
        A = np.transpose(np.linalg.solve(h_x, np.transpose(F)))

        output = {'lebesgue_constant': np.max(np.sum(np.abs(A), 1))}

        return points, ind, output

    def domain(self):
        '''
        Return the domain of the set of basis functions, which is the support
        of the associated measure.

        Returns
        -------
        numpy.ndarray
            The domain of the set of basis functions

        '''
        return self.measure.support()

    def christoffel(self, x):
        # TODO christoffel
        raise NotImplementedError('Method not implemented.')

    def orthonormalize(self):
        '''
        Orthonormalize the basis.

        Returns
        -------
        out : tensap.SubFunctionalBasis
            The orthonormalized basis.

        '''
        G = self.gram_matrix()
        out = self
        if np.linalg.norm(G - np.eye(G.shape[0]), 2):
            A = np.linalg.inv(np.linalg.cholesky(G))
            out = tensap.SubFunctionalBasis(self, np.transpose(A))
        out.is_orthonormal = True
        return out

    def optimal_sampling_measure(self):
        # TODO optimal_sampling_measure
        raise NotImplementedError('Method not implemented.')

    def plot(self, indices=None, n=10000, *args):
        '''
        Plot the functions of the basis.

        Parameters
        ----------
        indices : list or numpy.ndarray, optional
            Indices of the functions to be plotted. The default is None,
            indicating all the functions.
        n : int, optional
            The number of points used for the plot. The default is 10000.
        *args : tuple
            Additional parameters used by matplotlib.pyplot's function plot.

        Returns
        -------
        None.

        '''
        assert self.ndim() == 1, 'Method not implemented.'

        import matplotlib.pyplot as plt

        sup = self.measure.truncated_support()
        if np.size(n) == 1:
            x = np.linspace(sup[0], sup[1], n)
        else:
            x = np.ravel(n)

        if indices is None:
            hx = self.eval(x)
        else:
            hx = self.eval(x, indices)

        plt.plot(x, hx, *args)
        plt.show()

    @abstractmethod
    def cardinal(self):
        '''
        Return the number of basis functions.

        Returns
        -------
        int
            The number of basis functions.

        '''

    @abstractmethod
    def ndim(self):
        '''
        Return the dimension n for f defined in R^n.

        Returns
        -------
        int
            The dimension n for f defined in R^n.

        '''

    @abstractmethod
    def eval(self, x):
        '''
        Return the evaluation of the basis functions at the points x.

        Parameters
        ----------
        x : list or numpy.ndarray
            The points at which the basis functions are to be evaluted.

        Returns
        -------
        numpy.ndarray
            The evaluations of the basis functions at the points x.

        '''
