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
Module piecewise_polynomial_functional_basis.

"""

import numpy as np
import tensap

# from scipy.sparse import lil_matrix


class PiecewisePolynomialFunctionalBasis(tensap.FunctionalBasis):

    def __init__(self, points, p):
        """
        Constructor for the PiecewisePolynomialFunctionalBasis class, which
        defines a functional basis composed of piecewise polynomials of degree p
        on an interval [a,b]. The basis is orthonormal with respect to the
        uniform probability measure on (a,b).
        Attributes
        ----------
        points : list or numpy.ndarray
            the knots.

        p : integer or list or numpy.ndarray
            the polynomial degree, or list of polynomial degrees
            in the intervals

        measure : tensap.Measure
            The measure associated with basis (LebesgueMeasure).

        is_orthonormal : boolean
            Boolean equal to true, indicating that the basis is orthonormal with
            respect to the measure.

        """
        tensap.FunctionalBasis.__init__(self)
        self.points = np.ravel(points)  # Flatten points
        if np.isscalar(p):
            p = np.tile(p, self.points.size - 1)  # Replicate if p is scalar
        self.p = np.ravel(p)  # Ensure p is 1D
        self.is_orthonormal = True
        self.measure = tensap.LebesgueMeasure(self.points[0], self.points[-1])

    def eval(self, x, indices=None):
        """
        Evaluates the functional basis at the points x.
        """
        x = np.ravel(x)
        pos = self.element_number(x)
        p1 = self.points[:-1]
        p2 = self.points[1:]
        h = p2 - p1

        u = (x - p1[pos]) / h[pos]
        U = tensap.LebesgueMeasure(0, 1)
        pol = tensap.PolynomialFunctionalBasis(
            U.orthonormal_polynomials(), np.arange(np.max(self.p) + 1)
        )
        pu = pol.eval(u)
        hx = np.zeros((x.size, self.cardinal()))

        for i in range(self.points.size - 1):
            I_rows = np.where(pos == i)[0]
            if I_rows.size > 0:
                J = np.sum(self.p[:i] + 1) + np.arange(self.p[i] + 1)
                hx[np.ix_(I_rows, J)] = pu[I_rows, : self.p[i] + 1] / np.sqrt(h[i])

        if indices is not None:
            hx = hx[:, indices]

        return hx

    def eval_derivative(self, k, x, indices=None):
        """
        Evaluates the k-th derivative of the functional basis at the points x.
        """
        x = np.ravel(x)
        pos = self.element_number(x)
        p1 = self.points[:-1]
        p2 = self.points[1:]
        h = p2 - p1
        u = (x - p1[pos]) / h[pos]
        U = tensap.LebesgueMeasure(0, 1)
        pol = tensap.PolynomialFunctionalBasis(
            U.orthonormal_polynomials(), np.arange(np.max(self.p) + 1)
        )
        pu = pol.eval_derivative(k, u)
        hx = np.zeros((x.size, self.cardinal()))

        for i in range(self.points.size - 1):
            loc = np.where(pos == i)[0]
            if loc.size > 0:
                J = np.sum(self.p[:i] + 1) + np.arange(self.p[i] + 1)
                scale = h[i]
                hx[np.ix_(loc, J)] = (
                    pu[loc, : self.p[i] + 1] / np.sqrt(scale) * h[i] ** (-k)
                )

        if indices is not None:
            hx = hx[:, indices]

        return hx

    def cardinal(self):
        """
        Returns the cardinality of the basis.
        """
        return np.sum(self.p + 1)

    def ndim(self):
        """
        Returns the number of dimensions of the basis (here 1).

        """
        return 1

    def mean(self):
        """
        Computes the mean of the functional basis.
        """
        p1 = self.points[:-1]
        p2 = self.points[1:]
        h = p2 - p1
        m = np.zeros(self.cardinal())
        q = np.cumsum(np.concatenate((0, self.p[:-1] + 1)))
        m[q] = np.sqrt(h)
        return m

    def interpolation_points(self):
        """
        Returns interpolation points for the
        PiecewisePolynomialFunctionalBasis.
        """
        unique_p = np.unique(self.p)
        u = [0.5 * (1 + tensap.chebyshev_points(p + 1)) for p in unique_p]

        p1 = self.points[:-1]
        p2 = self.points[1:]
        h = p2 - p1
        x = np.zeros(self.cardinal())

        for i in range(self.points.size - 1):
            loc = np.where(self.p[i] == unique_p)[0][0]
            ui = np.ravel(u[loc])
            J = np.sum(self.p[:i] + 1) + np.arange(self.p[i] + 1)
            x[J] = p1[i] + ui * h[i]

        return x

    def magic_points(self):
        """
        Returns magic points for the PiecewisePolynomialFunctionalBasis.
        """
        unique_p = np.unique(self.p)
        u = []
        for i in range(unique_p.size):
            U = tensap.UniformRandomVariable(0, 1)
            pol = tensap.PolynomialFunctionalBasis(
                U.orthonormal_polynomials(), np.arange(unique_p[i] + 1)
            )
            u.append(pol.magic_points()[0])

        p1 = self.points[:-1]
        p2 = self.points[1:]
        h = p2 - p1
        x = np.zeros(self.cardinal())

        for i in range(self.points.size - 1):
            loc = np.where(self.p[i] == unique_p)[0][0]
            ui = np.ravel(u[loc])
            J = np.sum(self.p[:i] + 1) + np.arange(self.p[i] + 1)
            x[J] = p1[i] + ui * h[i]

        return x

    def element_number(self, x):
        """
        Determines the element number for each point in x.
        """
        x = np.ravel(x)
        n = self.points.size - 1
        pos = np.zeros(x.size, dtype=int)
        for i in range(n):
            mask = (x >= self.points[i]) & (x < self.points[i + 1])
            pos[mask] = i
        pos[x >= self.points[-1]] = n - 1
        pos[x < self.points[0]] = 0
        return pos

    @staticmethod
    def hp(a, b, h, p):
        """
        Creates a PiecewisePolynomialFunctionalBasis with a given mesh size.
        """
        n = int(np.ceil((b - a) / h))
        points = np.linspace(a, b, n + 1)
        return tensap.PiecewisePolynomialFunctionalBasis(points, p)

    @staticmethod
    def np(a, b, n, p):
        """
        Creates a PiecewisePolynomialFunctionalBasis with a given number of elements.
        """
        points = np.linspace(a, b, n + 1)
        return tensap.PiecewisePolynomialFunctionalBasis(points, p)

    @staticmethod
    def singularityhp_adapted(a, b, s, h):
        """
        Creates a PiecewisePolynomialFunctionalBasis adapted around singularities.
        """
        s = np.ravel(s)
        e = np.concatenate((np.concatenate(([a], s)), [b]))
        e = np.unique(e)

        x = []
        p = []

        for i in range(e.size - 1):
            ai = e[i]
            bi = e[i + 1]
            li = bi - ai
            ne = int(np.ceil(np.log2(1 / h)))
            if ai in s and bi in s:
                pi = np.concatenate([np.arange(ne), np.arange(ne - 1, -1, -1)])
                xi = np.concatenate(
                    [
                        [0],
                        2.0 ** (-np.arange(ne, 1, -1)),
                        [1 / 2],
                        1 - 2.0 ** (-np.arange(2, ne + 1, 1)),
                        [1],
                    ]
                )
            elif ai in s:
                pi = np.arange(0, ne)
                xi = np.concatenate([[0], 2.0 ** (-np.arange(ne, 0, -1)), [1]])
                pi = np.arange(xi.size - 1)
            elif bi in s:
                xi = np.concatenate([[0], 1 - 2.0 ** (-np.arange(1, ne + 1)), [1]])
                pi = np.arange(xi.size - 2, -1, -1)
            if i < e.size - 2:
                xi = xi[:-1]
            xi = ai + xi * li
            p.append(pi)
            x.append(xi)

        x = np.concatenate(x)
        p = np.concatenate(p)
        return tensap.PiecewisePolynomialFunctionalBasis(x, p)
