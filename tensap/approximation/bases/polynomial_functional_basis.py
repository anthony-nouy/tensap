'''
Module polynomial_functional_basis.

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

import numpy as np
import tensap


class PolynomialFunctionalBasis(tensap.FunctionalBasis):
    '''
    Class PolynomialFunctionalBasis.

    Attributes
    ----------
    basis : tensap.UnivariatePolynomials
        The polynomials associated with the basis.

    indices : list or numpy.ndarray
        The indices of the selected polynomials.

    measure : tensap.Measure
        The measure associated with basis.

    is_orthonormal : bool
        Boolean equal to true, indicating that the basis is orthonormal with
        respect to measure.

    '''

    def __init__(self, basis, indices):
        '''
        Constructor for the class PolynomialFunctionalBasis.

        Parameters
        ----------
        basis : tensap.UnivariatePolynomials
            The polynomials associated with the basis.
        indices : list or numpy.ndarray
            The indices of the selected polynomials.

        Returns
        -------
        None.

        '''
        tensap.FunctionalBasis.__init__(self)
        self.basis = basis
        self.indices = np.array(indices)
        self.measure = basis.measure

        if isinstance(basis, (tensap.OrthonormalPolynomials,
                              tensap.ShiftedOrthonormalPolynomials)):
            self.is_orthonormal = True

    def __repr__(self):
        return ('<{}:{n}' +
                '{t}basis = {},{n}' +
                '{t}indices = {},{n}' +
                '{t}measure = {},{n}' +
                '{t}is_orthonormal = {}>').format(self.__class__.__name__,
                                                  self.basis,
                                                  self.indices,
                                                  self.measure,
                                                  self.is_orthonormal,
                                                  t='\t', n='\n')

    def __eq__(self, basis_2):
        if not isinstance(basis_2, PolynomialFunctionalBasis):
            out = False
        else:
            out = self.basis == basis_2.basis and \
                np.array_equal(self.indices, basis_2.indices)
        return out

    def cardinal(self):
        return self.indices.size

    def eval(self, x):
        '''
        Evaluate the polynomials of self.basis of degrees in self.indices at
        points x.

        Parameters
        ----------
        x : list or numpy.ndarray
            The points at which the basis functions are to be evaluated.

        Returns
        -------
        numpy.ndarray
            The evaluations of the polynomials of self.basis of degrees in
            self.indices at points x.

        '''
        return self.basis.polyval(self.indices, x)

    def kron(self, q):
        m = np.max(self.indices) + np.max(q.indices)
        b = PolynomialFunctionalBasis(self.basis, range(m))
        b.measure = self.measure

        def fun(x):
            return np.reshape(np.tile(np.expand_dims(self.eval(x), 2),
                                      (1, 1, q.cardinal())) *
                              np.tile(np.transpose(
                                  np.expand_dims(q.eval(x), 2), [0, 2, 1]),
                                  (1, self.cardinal(), 1)),
                              (np.shape(x)[0], self.cardinal()*q.cardinal()),
                              order='F')
        measure = b.measure
        int_rule = measure.gauss_integration_rule(2*m)
        return tensap.SubFunctionalBasis(b,
                                         self.projection(fun, int_rule).data)

    def derivative(self, k, measure=None):
        '''
        Compute the k-th order derivative of the functions of the basis
        projected on itself.

        Parameters
        ----------
        k : int
            The order of the derivative.
        measure : tensap.Measure, optional
            The measure used fot the projection. The default is None,
            indicating to use self.measure if it is a tensap.RandomVariable.

        Raises
        ------
        ValueError
            If no Measure is provided or can be extracted from self.

        Returns
        -------
        tensap.SubFunctionalBasis
            The k-th order derivative of the functions of the basis projected
            on itself.

        '''
        if measure is None:
            assert self.measure is not None, 'Must specify a Measure.'
            measure = self.measure

        nb_pts = int(np.ceil(np.max(self.indices) - (k-1)/2))
        int_rule = measure.gauss_integration_rule(nb_pts)
        def fun(x): return self.basis.dn_polyval(k, self.indices, x)
        return self.projection(fun, int_rule).sub_functional_basis()

    def gradient(self, measure=None):
        '''
        Compute the first order derivative of the functions of the basis
        projected on itself.

        Parameters
        ----------
        measure : tensap.Measure, optional
            The measure used fot the projection. The default is None,
            indicating to use self.measure if it is a tensap.RandomVariable.

        Returns
        -------
        tensap.SubFunctionalBasis
            The first order derivative of the functions of the basis projected
             on itself.

        '''
        return self.derivative(1, measure)

    def hessian(self, measure):
        '''
        Compute the second order derivative of the functions of the basis
        projected on itself.

        Parameters
        ----------
        measure : tensap.Measure, optional
            The measure used fot the projection. The default is None,
            indicating to use self.measure if it is a tensap.RandomVariable.

        Returns
        -------
        tensap.SubFunctionalBasis
            The second order derivative of the functions of the basis projected
             on itself.

        '''
        return self.derivative(2, measure)

    def eval_derivative(self, k, x):
        '''
        Evaluate the k-th order derivative of the functions of the basis at the
        points x.

        Parameters
        ----------
        k : int
            The order of the derivative.
        x : list or numpy.ndarray
            The points at which the k-th derivative of the basis functions are
            to be evaluated.

        Returns
        -------
        numpy.ndarray
            The evaluations of the k-th derivative of the polynomials of
            self.basis of degrees in self.indices at points x.

        '''
        return self.basis.dn_polyval(k, self.indices, x)

    def gram_matrix(self, measure=None):
        '''
        Compute the Gram matrix of the basis. The Gram matrix is the matrix of
        the dot products between each possible couple of basis functions. The
        dot product in the dimension i is computed according to measure if
        provided, or according to self.measure otherwise.

        Parameters
        ----------
        measure : tensap.Measure, optional
            The measure according to which the dot product is computed. The
            default is None, indicating to use self.measure.

        Raises
        ------
        NotImplementedError
            If the attribute basis of self does not have a method moment.

        Returns
        -------
        numpy.ndarray
            The Gram matrix of the basis.

        '''
        if measure is None:
            assert self.measure is not None, 'Must specify a Measure.'
            measure = self.measure
        try:
            ind = np.reshape(self.indices, [-1, 1])
            rep = np.hstack((
                np.reshape(np.tile(ind, [len(ind), 1]), [-1, 1]),
                np.reshape(np.tile(ind, [1, len(ind)]), [-1, 1])))
            return np.reshape(self.basis.moment(rep, measure), [len(ind)]*2)
        except AttributeError:
            raise NotImplementedError('Method not implemented.')

    def one(self):
        '''
        Return the coefficients associated with the basis so that it returns
        one.

        Returns
        -------
        list
            The list of coefficients so that the basis returns one.

        '''
        c, ind = self.basis.one()
        ok = np.isin(ind, self.indices)
        if not np.all(ok):
            raise ValueError('The constant 1 cannot be represented.')
        out = np.zeros(self.indices.size)
        out[np.isin(self.indices, ind)] = c
        return out

    def mean(self, measure=None):
        '''
        Return the expectation of the basis functions, accorging to measure if
        provided, and to self.measure ortherwise.

        Parameters
        ----------
        measure : None or tensap.RandomVariable, optional
            The measure according to which the mean is computed. The default is
            None, indicating to use self.measure.

        Returns
        -------
        numpy.ndarray
            The mean of the basis functions.

        '''
        return self.basis.mean(self.indices, measure)

    def random_variable(self):
        '''
        Return the tensap.RandomVariable associated with self if it exists.

        Returns
        -------
        out : tensap.RandomVariable or []
            The tensap.RandomVariable associated with self if it exists, and []
            otherwise.

        '''
        if hasattr(self.basis, 'random_variable'):
            out = self.basis.random_variable
        else:
            print('Empty random variable.')
            out = []
        return out

    def optimal_sampling_measure(self):
        # TODO optimal_sampling_measure
        raise NotImplementedError('Method not implemented.')

    def christoffel(self):
        # TODO christoffel
        raise NotImplementedError('Method not implemented.')

    def domain(self):
        return self.basis.domain()

    def ndim(self):
        return self.basis.ndim()
