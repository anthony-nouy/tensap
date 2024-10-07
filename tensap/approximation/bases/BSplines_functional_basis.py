# Copyright (c) 2020, Anthony Nouy, Erwan Grelier
# This file is part of tensap (tensor approximation package).

# tensap is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# tensap is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with tensap.  If not, see <https://www.gnu.org/licenses/>.


"""
Module BSplines_functional_basis.

"""

import numpy as np
import tensap


class BSplinesFunctionalBasis(tensap.FunctionalBasis):
    """
    BSplinesFunctionalBasis class representing B-splines with given knots and degree.

    Attributes
    ----------
    knots : numpy.ndarray
        The array of knots defining the spline.

    degree : int
        The degree of the spline.
    """

    def __init__(self, knots=[], degree=[]):
        """
        Initializes the B-Splines with the given knots and degree.

        Parameters
        ----------
        knots : numpy.ndarray or list
            Array of knots defining the spline.
        degree : int
            The degree of the spline.
        """
        self.knots = np.ravel(knots)
        self.degree = degree
        self.is_orthonormal = False
        self.measure = tensap.LebesgueMeasure(self.knots[0], self.knots[-1])

    def cardinal(self):
        """
        Returns the number of B-splines defined by the basis.

        Returns
        -------
        int
            The number of B-splines.
        """
        return self.knots.size - 1 - self.degree

    def ndim(self):
        """
        Returns the number of dimensions of the basis (always 1 for univariate splines).

        Returns
        -------
        int
            Number of dimensions (1).
        """
        return 1

    def eval_sequence(self, x):
        """
        Evaluate the B-Splines at the points `x` from degree 0 to the B-Splines.

        Parameters
        ----------
        x : numpy.ndarray or list
            Array of points at which to evaluate the B-splines.

        Returns
        -------
        Bx : numpy.ndarray
            The array of B-spline basis functions for internal use.
        """
        x = np.ravel(x)
        t = self.knots
        m = t.size
        n = x.size

        Bx = np.zeros((n, m - 1, self.degree + 1))
        for i in range(m - 1):
            Bx[:, i, 0] = (x > t[i]) & (x <= t[i + 1])

        for j in np.arange(1, self.degree + 1):
            for i in range(m - 1 - j):
                Bx[:, i, j] = (x - t[i]) / (t[i + j] - t[i]) * Bx[:, i, j - 1] + \
                    (t[i + j + 1] - x) / (t[i + j + 1] - t[i + 1]) * Bx[:, i + 1, j - 1]

        return Bx

    def eval(self, x):
        """
        Evaluate the B-splines at the points `x`.

        Parameters
        ----------
        x : numpy.ndarray or list
            Array of points at which to evaluate the B-splines.

        Returns
        -------
        Bx : numpy.ndarray
            The array of B-spline basis functions for internal use.
        """

        t = self.knots

        Bx = self.eval_sequence(x)
        Bx = Bx[:, :t.size - 1 - self.degree, -1]  # Last layer
        return Bx

    def eval_derivative(self, n, x):
        """
        Evaluate the n-th derivative of the B-spline at the points `x`.

        Parameters
        ----------
        n : int
            The order of the derivative to evaluate.
        x : numpy.ndarray or list
            Array of points at which to evaluate the derivative.

        Returns
        -------
        dBx : numpy.ndarray
            Array containing the evaluated n-th derivatives of the B-splines.
        """
        t = self.knots
        x = np.ravel(x)

        dBx = self.eval_sequence(x)

        for k in np.arange(1, n + 1):
            dBx_old = dBx
            dBx = np.zeros((x.size, t.size - 1, self.degree + 1))

            for j in np.arange(1, self.degree + 1):
                for i in range(t.size - 1 - j):
                    dBx[:, i, j] = k / (t[i + j] - t[i]) * dBx_old[:, i, j - 1] - \
                        k / (t[i + j + 1] - t[i + 1]) * dBx_old[:, i + 1, j - 1] + \
                        (x - t[i]) / (t[i + j] - t[i]) * dBx[:, i, j - 1] + \
                        (t[i + j + 1] - x) / (t[i + j + 1] - t[i + 1]) * dBx[:, i + 1, j - 1]

        return dBx[:, :t.size - 1 - self.degree, -1]

    def gauss_integration_rule(self, n):
        """
        Returns a Gaussian integration rule with n-points per interval.

        Parameters
        ----------
        n : int
            The number of points for the Gaussian quadrature rule.

        Returns
        -------
        IntegrationRule
            A rule for integrating over the spline basis.
        """
        # Assuming a PiecewisePolynomialFunctionalBasis exists
        p = tensap.PiecewisePolynomialFunctionalBasis(self.knots, 0)
        return p.gauss_integration_rule(n)

    @staticmethod
    def cardinal_bspline(m):
        """
        Creates a cardinal B-Spline of degree `m`.

        Parameters
        ----------
        m : int
            Degree of the B-spline.

        Returns
        -------
        BSplinesFunctionalBasis
            The cardinal B-Spline basis of degree `m`.
        """
        return BSplinesFunctionalBasis(np.arange(m + 2), m)

    @staticmethod
    def with_extra_knots(t, m):
        """
        Returns a B-Spline basis with extra knots added on both sides of the interval.

        Parameters
        ----------
        t : numpy.ndarray or list
            Array of knots.
        m : int
            Degree of the B-spline.

        Returns
        -------
        BSplinesFunctionalBasis
            The B-Spline basis with extra knots.
        """
        t = np.ravel(t)
        tl = t[0] + np.arange(-m, 0) * (t[1] - t[0])
        tr = t[-1] + np.arange(1, m + 1) * (t[-1] - t[-2])
        t_full = np.concatenate((tl, t, tr))
        return tensap.BSplinesFunctionalBasis(t_full, m)
