'''
Module polynomials.

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

from abc import ABC, abstractmethod
import numpy as np
import tensap


class UnivariatePolynomials(ABC):
    '''
    Class UnivariatePolynomials.

    '''

    def polyval(self, ind, x):
        '''
        Evaluate the polynomials of degrees contained in ind at the points x.

        Parameters
        ----------
        ind : list or numpy.ndarray
            The degrees of the polynomials to be evaluated.
        x : list or numpy.ndarray
            The points at which the polynomials are to be evaluated.

        Returns
        -------
        numpy.ndarray
            The evaluation of the selected polynomials at the points x.

        '''
        x = np.ravel(x)
        coef = np.fliplr(self.poly_coeff(np.arange(np.max(ind)+1)))

        out = np.zeros((x.size, coef.shape[0]))
        for deg in range(coef.shape[0]):
            out[:, deg] = np.polyval(coef[deg, :], x)
        return out[:, ind]

    def d_poly_coef(self, ind):
        '''
        Return the coefficients of the first order derivative of the
        polynomials of degrees contained in ind.

        Parameters
        ----------
        ind : list or numpy.ndarray
            The degrees of the polynomials to be evaluated.

        Returns
        -------
        numpy.ndarray
            The coefficients of the first order derivative of the polynomials
            of order contained in ind.

        '''
        return self.poly_coeff(ind)[:, 1:] * \
            np.tile(np.arange(1, np.max(ind)+1), (len(ind), 1))

    def dn_poly_coeff(self, n, ind):
        '''
        Return the coefficients of the n-th order derivative of the
        polynomials of degrees contained in ind.

        Parameters
        ----------
        n : int
            The degrees of derivation of the polynomials.
        ind : list or numpy.ndarray
            The orders of the polynomials to be evaluated.

        Returns
        -------
        numpy.ndarray
            The coefficients of the n-th order derivative of the polynomials
            of order contained in ind.

        '''
        coef = np.prod(np.tile(np.arange(np.max(ind)-n+1), (n, 1)) +
                       np.tile(np.expand_dims(np.arange(1, n+1), axis=1),
                               (1, np.max(ind)-n+1)), 0)
        coef = self.poly_coeff(ind)[:, n:] * np.tile(coef, (len(ind), 1))
        if coef.size == 0:
            coef = np.zeros(len(ind))
        return coef

    def d_polyval(self, ind, x):
        '''
        Evaluate the first order derivative of the polynomials of degrees
        contained in ind at the points x.

        Parameters
        ----------
        ind : list or numpy.ndarray
            The degrees of the polynomials to be evaluated.
        x : list or numpy.ndarray
            The points at which the derivatives of the polynomials are to be
            evaluated.

        Returns
        -------
        numpy.ndarray
            The evaluation of the first order derivatives of the selected
            polynomials at the points x.

        '''
        return self.dn_polyval(1, ind, x)

    def dn_polyval(self, n, ind, x):
        '''
        Evaluate the n-th order derivative of the polynomials of degrees
        contained in ind at the points x.

        Parameters
        ----------
        ind : list or numpy.ndarray
            The degrees of the polynomials to be evaluated.
        x : list or numpy.ndarray
            The points at which the derivatives of the polynomials are to be
            evaluated.

        Returns
        -------
        numpy.ndarray
            The evaluation of the n-th order derivatives of the selected
            polynomials at the points x.

        '''
        x = np.ravel(x)
        coef = np.fliplr(self.dn_poly_coeff(n, np.arange(np.max(ind)+1)))

        out = np.zeros((x.size, coef.shape[0]))
        for deg in range(coef.shape[0]):
            out[:, deg] = np.polyval(coef[deg, :], x)
        return out[:, ind]

    def mean(self, ind, measure=None):
        '''
        Return the mean of the polynomials of degrees contained in ind, with a
        Measure given by measure if provided, or to self.measure otherwise.

        Parameters
        ----------
        ind : list or numpy.ndarray
            The degrees of the polynomials for which the mean is to be
            computed.
        measure : tensap.Measure, optional
            The measure used for the computation of the mean. The default is
            None, indicating to use self.measure.

        Returns
        -------
        numpy.ndarray
            The mean of the selected polynomials.

        '''
        if measure is None:
            assert self.measure is not None, 'Must provide a Measure.'
            out = self.moment(np.reshape(ind, [-1, 1])) / self.measure.mass()
        else:
            out = self.moment(np.reshape(ind, [-1, 1]), measure) / \
                measure.mass()
        return out

    def moment(self, ind, measure=None):
        '''
        Return the moments of the family of polynomials p_i(X), i in ind,
        of a random variable X, using a gauss integration rule.

        Assuming ind is a numpy.ndarray:
            - if ind.ndim == 1, return the float
                m = E(p_ind[0](X)...p_ind[-1](X)),
            - else if ind.ndim == 2, return the vector
                m = (E(p_ind[j, 0](X)...p_ind[j, -1](X)) :
                    j = 1, ..., ind.shape[0]).

        Parameters
        ----------
        ind : list or numpy.ndarray
            The degrees of the polynomials for which the moments are to be
            computed.
        measure : tensap.Measure, optional
            The measure used for the computation of the moments. The default is
            None, indicating to use self.measure.

        Returns
        -------
        numpy.ndarray
            The moments of the family of polynomials.

        '''
        if measure is None:
            measure = self.measure

        ind = np.atleast_2d(ind)
        out = np.zeros(ind.shape[0])

        max_deg = np.max(np.sum(ind, 1))  # Maximum degree of the product
        nb_pts = int(np.ceil((max_deg+1)/2))  # Number of points

        G = measure.gauss_integration_rule(nb_pts)
        for i in range(ind.shape[0]):
            # Creation of the function to integrate
            def poly(x):
                out_loc = self.polyval(ind[i, 0], x)
                for j in np.arange(1, ind.shape[1]):
                    out_loc = out_loc * self.polyval(ind[i, j], x)
                return out_loc
            out[i] = G.integrate(poly)
        return out

    @staticmethod
    def ndim():
        '''
        Return the dimension of the output of the polynomials.

        Returns
        -------
        int
            The dimension of the output of the polynomials.

        '''
        return 1

    @abstractmethod
    def one(self):
        '''
        Coefficients and corresponding indices for the decomposition of the
        constant function 1.

        '''

    @abstractmethod
    def is_orthonormal(self):
        '''
        Check the orthonormality of the basis created by the functions of self.

        Returns
        -------
        bool
            Indicates if the polynomials are orthonormal.

        '''

    @abstractmethod
    def poly_coeff(self, ind):
        '''
        Compute the coefficients of the monomials used to create the
        polynomials of degree specified in ind.

        Parameters
        ----------
        ind : ind or numpy.ndarray
            The orders of the polynomials to be evaluated.

        '''


class CanonicalPolynomials(UnivariatePolynomials):
    '''
    Class CanonicalPolynomials.

    Attributes
    ----------
    measure : None or tensap.Measure
        The measure associated with the canonical polynomials.

    '''

    def __init__(self, measure=None):
        '''
        Constructor for the class CanonicalPolynomials.

        Parameters
        ----------
        measure : tensap.Measure, optional
            The measure associated with the canonical polynomials. The default
            is None.

        Raises
        ------
        ValueError
            If the provided measure is not a tensap.Measure object.

        Returns
        -------
        None.

        '''
        if measure is not None and not isinstance(measure, tensap.Measure):
            raise ValueError('Must provide a measure.')
        self.measure = measure

    def __eq__(self, poly_2):
        return isinstance(poly_2, CanonicalPolynomials)

    @staticmethod
    def is_orthonormal():
        return False

    def poly_coeff(self, ind):
        return np.eye(np.max(ind)+1)[ind, :]

    def d_polyval(self, ind, x):
        x = np.expand_dims(np.ravel(x), axis=1)
        ind = np.atleast_1d(ind)
        out = np.zeros((x.size, len(ind)))
        rep = ind != 0
        out[:, rep] = x**(ind[rep]-1) * np.tile(ind[rep], (x.size, 1))
        return out

    @staticmethod
    def domain():
        '''
        Return the domain of the canonical polynomials.

        Returns
        -------
        list
            The domain of the canonical polynomials.

        '''
        return [-np.inf, np.inf]

    @staticmethod
    def one():
        return 1, 0
