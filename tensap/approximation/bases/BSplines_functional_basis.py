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
        tensap.FunctionalBasis.__init__(self)
        self.knots = np.ravel(knots)
        self.degree = degree
        self.measure = tensap.LebesgueMeasure(self.knots[0], self.knots[-1])
        self.is_orthonormal = False

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
                Bx[:, i, j] = (x - t[i]) / (t[i + j] - t[i]) * Bx[:, i, j - 1] + (
                    t[i + j + 1] - x
                ) / (t[i + j + 1] - t[i + 1]) * Bx[:, i + 1, j - 1]

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
        Bx = Bx[:, : t.size - 1 - self.degree, -1]  # Last layer
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
                    dBx[:, i, j] = (
                        k / (t[i + j] - t[i]) * dBx_old[:, i, j - 1]
                        - k / (t[i + j + 1] - t[i + 1]) * dBx_old[:, i + 1, j - 1]
                        + (x - t[i]) / (t[i + j] - t[i]) * dBx[:, i, j - 1]
                        + (t[i + j + 1] - x)
                        / (t[i + j + 1] - t[i + 1])
                        * dBx[:, i + 1, j - 1]
                    )

        dBx = dBx[:, : t.size - 1 - self.degree, -1]
        return dBx

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
        B = tensap.BSplinesFunctionalBasis(t_full, m)
        B.measure = tensap.LebesgueMeasure(t[0], t[-1])

        return B


class DilatedBSplines:
    def __init__(self, n, b=2):
        self.degree = n  # Degree
        self.base = b  # base of dilation

    def eval(self, i, x):
        """
        Evaluate tthe dilated B-Splines of indices i=[l,j]
        at points x, with l the level and j the local index

        Parameters
        ----------
        k : int
            The order of the derivative.
        i : list of numpry arrays
            i[0] is the level l (numpy array of shape (m, ))
            i[1] is the local index (numpy array of shape (m, ))
        x : numpy ndarray of shape (n,)
            The points.

        Returns
        -------
        Bx : numpy ndarray of shape (n,m)

        """
        x = np.ravel(x)
        m = self.degree
        b = self.base
        psi = BSplinesFunctionalBasis.cardinal_bspline(m)
        level = np.ravel(i[0])
        local_index = np.ravel(i[1])
        X = np.outer(x, b**level) - np.outer(np.ones(x.size), local_index)
        S = np.outer(np.ones(x.size), b ** (level / 2))

        Bx = psi.eval(X.ravel()).ravel() * S.ravel()
        Bx = np.reshape(Bx, (x.size, level.size))

        return Bx

    def eval_derivative(self, k, i, x):
        """
        Evaluate the k-th derivative of dilated B-Splines of indices i=[l,j]
        at points x, with l the level and j the local index

        Parameters
        ----------
        k : int
            The order of the derivative.
        i : list of numpry arrays
            i[0] is the level l (numpy array of shape (m, ))
            i[1] is the local index (numpy array of shape (m, ))
        x : numpy ndarray of shape (n,)
            The points.

        Returns
        -------
        dBx : numpy ndarray of shape (n,m)

        """
        x = np.ravel(x)
        m = self.degree
        b = self.base
        psi = BSplinesFunctionalBasis.cardinal_bspline(m)
        level = np.ravel(i[0])
        local_index = np.ravel(i[1])
        X = np.outer(x, b**level) - np.outer(np.ones(x.size), local_index)
        S = np.outer(np.ones(x.size), b ** (level * (k + 1 / 2)))

        dBx = psi.eval_derivative(k, X.ravel()).ravel() * S.ravel()
        dBx = np.reshape(dBx, (x.size, level.size))

        return dBx

    def indices_with_level_bounded_by(self, L):
        """
        Returns the indices of Dilated BSplines of level less or equal to L

        Parameters
        ----------
        L : int
            The level.

        Returns
        -------
        level : numpy array
            The level.
        local_index : numpy array
            The local index within the level.

        """

        level = np.zeros(0, dtype=int)
        local_index = np.zeros(0, dtype=int)
        for k in range(L + 1):
            lk, jk = self.indices_with_level(k)
            level = np.concatenate((level, lk))
            local_index = np.concatenate((local_index, jk))
        return level, local_index

    def indices_with_level(self, L):
        """
        Returns the indices of Dilated BSplines of level L

        Parameters
        ----------
        L : int
            The level.

        Returns
        -------
        level : numpy array
            The level.
        local_index : numpy array
            The local index within the level.
        """

        m = self.degree
        b = self.base
        local_index = np.arange(-m, b**L, dtype=int)
        level = np.full(local_index.size, L, dtype=int)
        return level, local_index


class DilatedBSplinesFunctionalBasis(tensap.FunctionalBasis):

    def __init__(self, B, Ind):
        """
        Functional basis of DilatedBSplines of degree n on
        (0,1), using b-adic dilations
        Ind = [L,J] is a list of numpy arrays of size (n,)

        Parameters
        ----------
        B : DilatedBSplines

        I : list of 2 numpy arrays
            I[0] and I[1] are of shape (n,)

        Returns
        -------
        DilatedBSplinesFunctionalBasis.

        """

        tensap.FunctionalBasis.__init__(self)
        if not isinstance(B, DilatedBSplines):
            raise ValueError("must provide a DilatedBSplines")

        if isinstance(Ind, tuple):
            Ind = list(Ind)
        elif not isinstance(Ind, list):
            raise ValueError("must provide a list or tuple")

        Ind[0] = np.ravel(Ind[0])
        Ind[1] = np.ravel(Ind[1])

        self.basis = B
        self.indices = Ind

    def eval(self, x):
        return self.basis.eval(self.indices, x)

    def cardinal(self):
        return self.indices[0].size

    def ndims(self):
        return 1

    def eval_derivative(self, k, x):
        """
        Evaluates the k-th derivative of the functional basis at points x

        Parameters
        ----------
        k : int
            The order of derivation.
        x : numpy array
            The points.

        Returns
        -------
        numpy array of shape (x.size ,self.cardinal())

        """
        return self.basis.eval_derivative(k, self.indices, x)

    @staticmethod
    def with_level_bounded_by(n, b, level):
        """
        Creates a DilatedBSplinesFunctionalBasis with degree n
        b-adic dilation, and functions with level less than level

        Parameters
        ----------
        n : int
            The degree of the splines.
        b : int
            The base of dilation.
        level : int
            maximum level.

        Returns
        -------
        DilatedBSplinesFunctionalBasis

        """

        B = DilatedBSplines(n, b)
        Ind = B.indices_with_level_bounded_by(level)
        return DilatedBSplinesFunctionalBasis(B, Ind)
