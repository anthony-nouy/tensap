'''
Module functional_basis_array.

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


class FunctionalBasisArray(tensap.Function):
    '''
    Class FunctionalBasisArray.

    Attributes
    ----------
    data : numpy.ndarray, optional
            The coefficents of the function on the basis. The default is None.
    basis : tensap.FunctionalBasis, optional
        The basis. The default is None.
    shape : list or numpy.ndarray, optional
        Array such that the function is with values in
        R^(shape[0] x shape[1] x ...). The default is 1.

    '''

    def __init__(self, data=None, basis=None, shape=None):
        '''
        Constructor for the class FunctionalBasisArray

        Parameters
        ----------
        data : numpy.ndarray, optional
            The coefficents of the function on the basis. The default is None.
        basis : tensap.FunctionalBasis, optional
            The basis. The default is None.
        shape : list or numpy.ndarray, optional
            Array such that the function is with values in
            R^(shape[0] x shape[1] x ...). The default is 1.

        Returns
        -------
        None.

        '''
        tensap.Function.__init__(self)

        if shape is None and (data is not None or basis is not None):
            shape = 1

        self.data = np.ravel(data)
        self.basis = deepcopy(basis)
        self.shape = np.atleast_1d(shape)
        self.output_shape = self.shape
        self.data = np.reshape(self.data,
                               np.concatenate(([basis.cardinal()],
                                               self.shape)))

    def __add__(self, g):
        return FunctionalBasisArray(self.data + g.data, self.basis, self.shape)

    def __neg__(self):
        return FunctionalBasisArray(-self.data, self.basis, self.shape)

    def __sub__(self, g):
        return FunctionalBasisArray(self.data - g.data, self.basis, self.shape)

    def __mul__(self, g):
        if isinstance(g, FunctionalBasisArray):
            g = g.data
        return FunctionalBasisArray(self.data * g, self.basis, self.shape)

    def matmul(self, v):
        '''
        Compute the matrix multiplication of self.data with v.

        Parameters
        ----------
        v : numpy.ndarray
            The array used in the matrix multiplication.

        Returns
        -------
        tensap.FunctionalBasisArray
            The result of the matrix multiplication.

        '''
        data = np.matmul(self.data, v)
        return FunctionalBasisArray(data, self.basis, data.shape[1:])

    def matdiv(self, v):
        '''
        Compute the matrix multiplication of self.data with the inverse of v.

        Parameters
        ----------
        v : numpy.ndarray
            The array used in the matrix multiplication.

        Returns
        -------
        tensap.FunctionalBasisArray
            The result of the matrix multiplication.

        '''
        data = np.transpose(np.linalg.solve(np.transpose(v),
                                            np.transpose(self.data)))
        return FunctionalBasisArray(data, self.basis, self.shape)

    def dot(self, g, dim=None):
        '''
        Compute the dot product between the arrays self.data and g.data
        treated as collections of vectors. The function calculates
        the dot product of corresponding vectors along the first
        array dimension whose size does not equal 1.

        Parameters
        ----------
        g : tensap.FunctionalBasisArray
            The second object of the dot product.
        dim : int or list or numpy.ndarray, optional
            The dimension along which the dot product is computed. The default
            is None, indicating all the dimensions.

        Returns
        -------
        float or numpy.ndarray
            The result of the dot product.

        '''
        return np.sum(self.data * g.data, dim, keepdims=True)

    def norm(self, p='fro'):
        '''
        Compute the p-norm of the array self.data.

        See also numpy.linalg.norm.

        Parameters
        ----------
        p : int or numpy.inf or -numpy.inf or string, optional
            The order of the norm. The default is 'fro'.

        Returns
        -------
        float
            The norm of self.data.

        '''
        return np.linalg.norm(np.reshape(self.data,
                                         [self.basis.cardinal(), -1],
                                         order='F'), p)

    @staticmethod
    def is_random():
        '''
        Determine if the object is random.

        Returns
        -------
        bool
            Boolean equal to True if the object is random.

        '''
        return True

    def mean(self, measure=None):
        '''
        Compute the expectation of the function, according to the measure
        associated with the tensap.ProbabilityMeasure measure if provided, or
        to the standard tensap.ProbabilityMeasure associated with each
        polynomial if not.

        Parameters
        ----------
        measure : tensap.ProbabilityMeasure, optional
            The probability measure used for the computation of the
            expectation. The default is None, indicating to use the standard
            tensap.ProbabilityMeasure associated with each polynomial.

        Returns
        -------
        numpy.ndarray
            The expectation of the function.

        '''
        M = self.basis.mean(measure)
        M = np.tile(np.ravel(M), np.concatenate((self.shape, [1])))
        M = np.transpose(M, np.concatenate(([np.ndim(M)-1],
                                            range(np.ndim(M)-1))))
        return np.sum(self.data*M, 0, keepdims=True)

    def expectation(self, measure=None):
        '''
        Compute the expectation of the function, according to the measure
        associated with the tensap.ProbabilityMeasure measure if provided, or
        to the standard tensap.ProbabilityMeasure associated with each
        polynomial if not.

        Parameters
        ----------
        measure : tensap.ProbabilityMeasure, optional
            The probability measure used for the computation of the
            expectation. The default is None, indicating to use the standard
            tensap.ProbabilityMeasure associated with each polynomial.

        Returns
        -------
        numpy.ndarray
            The expectation of the function.

        '''
        return self.mean(measure)

    def variance(self, measure=None):
        '''
        Compute the variance of the function, according to the measure
        associated with the tensap.ProbabilityMeasure measure if provided, or
        to the standard tensap.ProbabilityMeasure associated with each
        polynomial if not.

        Parameters
        ----------
        measure : tensap.ProbabilityMeasure, optional
            The probability measure used for the computation of the variance.
            The default is None, indicating to use the standard
            tensap.ProbabilityMeasure associated with each polynomial.

        Returns
        -------
        numpy.ndarray
            The variance of the function.

        '''
        m = self.expectation(measure)
        return self.dot_product_expectation(self, None, measure) - m**2

    def std(self, *args):
        '''
        Compute the standard deviation of the function, according to the
        measure associated with the tensap.ProbabilityMeasure measure if
        provided, or to the standard tensap.ProbabilityMeasure associated with
        each polynomial if not.

        Parameters
        ----------
        measure : tensap.ProbabilityMeasure, optional
            The probability measure used for the computation of the standard
            deviation. The default is None, indicating to use the standard
            tensap.ProbabilityMeasure associated with each polynomial.

        Returns
        -------
        numpy.ndarray
            The standard deviation of the function.

        '''
        return np.sqrt(self.variance(*args))

    def dot_product_expectation(self, g, dims=None, measure=None):
        '''
        Compute the expectation of self(X)g(X), where X is the probability
        measure associated with the underlying basis, or measure if provided.

        For vector-valued functions of X, dims specifies the dimensions of
        self and g corresponding to the RandomVector measure.

        Parameters
        ----------
        g : tensap.FunctionalBasisArray
            The second function of the product.
        dims : list or numpy.ndarray, optional
            The dimensions of self and g corresponding to the RandomVector
            measure. The default is None.
        measure : tensap.ProbabilityMeasure, optional
            The probability measure used for the computation of the
            expectation. The default is None, indicating to use the probability
            measure associated with the underlying basis.

        Raises
        ------
        NotImplementedError
            If the method is not implemented.

        Returns
        -------
        float or numpy.ndarray
            The result of the dot product.

        '''
        if dims is None:
            dims = range(self.basis.cardinal())
        if not (self.basis == g.basis) or not self.basis.is_orthonormal:
            raise NotImplementedError('Method not implemented.')

        return self.dot(g, 0)

    def norm_expectation(self, measure=None):
        '''
        Compute the L^2 norm of self(measure). If measure is not provided, use
        the probability measure associated with the underlying basis of self.

        Parameters
        ----------
        measure : tap.ProbabilityMeasure, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        float
            The L2 norm of the function.

        '''
        return np.sqrt(self.dot_product_expectation(self, None, measure))

    def conditional_expectation(self, dims, *args):
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
        *args : tuple
            Additional parameters. See also the method conditional_expectation
            of the underlying basis.

        Returns
        -------
        f : tensap.FunctionalBasisArray
            The conditional expectation of the function.

        '''
        h = self.basis.conditional_expectation(dims, *args)
        f = deepcopy(self)
        f.data = np.matmul(h.data, np.reshape(self.data,
                                              [self.basis.cardinal(),
                                               np.prod(self.shape)],
                                              order='F'))
        f.data = np.reshape(f.data, (f.data.shape[0], f.shape), order='F')
        f.basis = h.basis
        return f

    def variance_conditional_expectation(self, alpha):
        '''
        Compute the variance of the conditional expectation of the function in
        dimensions in alpha.

        Parameters
        ----------
        alpha : numpy.ndarray
            The dimensions in which the variance of the conditional expectation
            of the function if computed.

        Returns
        -------
        numpy.ndarray
            The variance of the conditional expectation of the function.

        '''
        alpha = np.atleast_2d(alpha)
        m = self.expectation()
        v = np.zeros((np.shape(alpha)[0], np.prod(self.shape)))
        for i in range(np.shape(alpha)[0]):
            u = alpha[i, :]
            if np.all([isinstance(x, bool) for x in u]):
                u = np.nonzero(u)[0]

            if np.size(u) == 0:
                v[i, :] = 0
            else:
                mi = self.conditional_expectation(u)
                vi = mi.dot_product_expectation(mi) - m**2
                v[i, :] = np.ravel(vi)

        return np.reshape(v, np.concatenate(([np.shape(alpha)[0]],
                                             self.shape)), order='F')

    def eval(self, x, *args):
        if np.ndim(x) == 1:
            x = np.expand_dims(x, 1)
        H = self.basis.eval(x, *args)
        return self.eval_with_bases_evals(H, *args)

    def eval_with_bases_evals(self, H):
        '''
        Compute the evaluations of the function using the evaluations of the
        basis H.

        Parameters
        ----------
        H : numpy.ndarray
            The evaluations of the basis.

        Returns
        -------
        numpy.ndarray
            The evaluations of the function.

        '''
        y = np.matmul(H, np.reshape(self.data, [self.basis.cardinal(),
                                                np.prod(self.shape)],
                                    order='F'))
        return np.reshape(y, np.concatenate(([H.shape[0]], self.shape)),
                          order='F')

    def eval_derivative(self, n, x):
        '''
        Compute the n-derivative of the function at points x in R^d, with n a
        multi-index of size d.

        Parameters
        ----------
        n : int or list or numpy.ndarray
            The derivation order in all the dimensions, or the derivation
            orders for each dimension.
        x : numpy.ndarray
            The points used for the evaluation of the derivative.

        Raises
        ------
        NotImplementedError
            If the method is not implemented for the basis.

        Returns
        -------
        numpy.ndarray
            The evaluation of the n-derivative of the function at the points x.

        '''
        try:
            H = self.basis.eval_derivative(n, x)
            return self.eval_with_bases_evals(H)
        except Exception:
            raise NotImplementedError('Method not implemented for the basis.')

    def derivative(self, n):
        '''
        Compute the n-derivative of the function.

        Parameters
        ----------
        n : int
            The derivation order.

        Returns
        -------
        df : tensap.FunctionalBasisArray()
            The n-derivative of the function.

        '''
        df = deepcopy(self)
        df.basis = self.basis.derivative(n)
        return df

    def random(self, n=1, measure=None):
        '''
        Compute evaluations of the function at an array of points of size n,
        drawn randomly according to the tensap.ProbabilityMeasure measure if
        provided, or to the standard tensap.ProbabilityMeasure associated with
        each polynomial if not.

        Parameters
        ----------
        n : int, optional
            The number of random evaluations. The default is 1.
        measure : tensap.ProbabilityMeasure, optional
            The probability measure used to draw the points of evaluation. The
            default is None, indicating to use the standard
            tensap.ProbabilityMeasure associated with each polynomial.

        Returns
        -------
        numpy.ndarray
            The random evaluations of the function.
        numpy.ndarray
            The points used for the evaluations of the function.

        '''
        fx, x = self.basis.random(n, measure)
        y = np.matmul(fx, np.reshape(self.data, [self.basis.cardinal(),
                                                 np.prod(self.shape)],
                                     order='F'))
        return np.reshape(y, np.concatenate(([fx.shape[0]], self.shape)),
                          order='F'), x

    def get_random_vector(self):
        '''
        Return the random vector associated with the basis functions of the
        object.

        Returns
        -------
        tensap.RandomVector
            The random vector associated with the basis functions of the
            object.

        '''
        return self.basis.get_random_vector()

    def storage(self):
        '''
        The storage complexity of the object.

        Returns
        -------
        int
            The storage complexity of the object.

        '''
        return np.size(self.data)

    def sparse_storage(self):
        '''
        The storage complexity of the object, taking into account the sparsity.

        Returns
        -------
        int
            The storage complexity of the object, taking into account the
            sparsity.

        '''
        return np.count_nonzero(self.data)

    def projection(self, basis, indices=None):
        '''
        Projection of the object on a functional basis using multi-indices
        indices if provided, or the multi-indices associated with
        the functional basis if not.

        Parameters
        ----------
        basis : tensap.FunctionalBasis (tensap.FullTensorProductFunctionalBasis
                or tensap.SparseTensorProductFunctionalBasis)
            The basis used for the projection.
        indices : tensap.MultiIndices, optional
            The multi-indices used for the projection. The default is None,
            indicating to use basis.indices.

        Returns
        -------
        FunctionalBasisArray
            The obtained projection.

        '''
        if indices is None:
            if isinstance(basis, tensap.SparseTensorProductFunctionalBasis):
                indices = basis.indices
            else:
                raise ValueError('Must specify a MultiIndices.')

        if self.basis.ndim() == basis.ndim() and \
                self.basis.cardinal() <= basis.cardinal():
            d = np.zeros((basis.cardinal(), np.prod(self.shape)))
            _, ia, ib = self.basis.indices.intersect_indices(indices)
            d[ib, :] = self.data[ia, :]
            if np.ndim(self.data) != np.ndim(d):
                d = np.reshape(d, [basis.cardinal(), self.shape], order='F')

            if isinstance(basis, tensap.FullTensorProductFunctionalBasis):
                H = tensap.FullTensorProductIntegrationRule(basis.bases)
            elif isinstance(basis, tensap.SparseTensorProductFunctionalBasis):
                H = tensap.SparseTensorProductFunctionalBasis(basis.bases,
                                                              indices)
            g = FunctionalBasisArray(d, H, self.shape)
        else:
            raise NotImplementedError('Method not implemented.')
        return g

    def sub_functional_basis(self):
        '''
        Converts the FunctionalBasisArray into a tensap.SubFunctionalbasis.

        Returns
        -------
        tensap.SubFunctionalbasis
            The FunctionalBasisArray as a tensap.SubFunctionalbasis.

        '''
        return tensap.SubFunctionalBasis(self.basis, self.data)

    def get_coefficients(self):
        '''
        Return the coefficients of the object.

        Returns
        -------
        numpy.ndarray
            The coefficients of the object.

        '''
        return np.reshape(self.data, np.concatenate(([self.basis.cardinal()],
                                                     self.shape)),
                          order='F')
