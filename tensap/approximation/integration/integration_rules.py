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
Module integration_rules.

"""

import numpy as np
import tensap


class IntegrationRule:
    """
    Class IntegrationRule.

    Attributes
    ----------
    points : numpy.ndarray
        The integration points.
    weights : numpy.ndarray
        The integration weights.

    """

    def __init__(self, points, weights):
        """
        Constructor for the class IntegrationRule.

        Parameters
        ----------
        points : list or numpy.ndarray
        The integration points.
        weights : list or numpy.ndarray
            The integration weights.

        Returns
        -------
        None.

        """
        self.points = np.atleast_1d(points)
        self.weights = np.atleast_1d(weights)

    def ndim(self):
        """
        Return the number of integration points.

        Returns
        -------
        int
            The number of integration points.

        """
        return self.points.shape[1]

    def integrate(self, fun):
        """
        Integrate the function fun using the integration rule.

        Parameters
        ----------
        fun : function
            The function to be integrated.

        Returns
        -------
        int
            The integrated function using the integration rule.

        """
        fun_eval = fun(self.points)
        return np.dot(np.ravel(fun_eval), self.weights)

    def tensorize(self, dim):
        """
        Create a tensap.FullTensorProductIntegrationRule in dimension dim
        using the object.

        Parameters
        ----------
        dim : int
            The dimension of the tensap.FullTensorProductIntegrationRule.

        Returns
        -------
        tensap.FullTensorProductIntegrationRule
            The integration rule in dimension dim.

        """
        points = tensap.FullTensorGrid(self.points, dim)
        weights = [self.weights] * dim
        return tensap.FullTensorProductIntegrationRule(points, weights)

    @staticmethod
    def gauss(measure, *args):
        """
        Call the method gauss_integration_rule of a measure.

        Parameters
        ----------
        measure : tensap.Measure
            The measure associated to which the integration
            rule is to be computed.
        *args : misc
            Additional arguments.

        Returns
        -------
        tensap.IntegrationRule
            The integration rule associated with the measure.

        """
        return measure.gauss_integration_rule(*args)

    def gauss_legendre_composite(knots, n):
        """
        Returns a piecewise Gauss-Legendre quadrature associated
        with given (sorted) points knots[0] ... knots[m]. It uses a n-points
        gauss integration rule per interval.

        Parameters
        ----------
        knots : numpy.array or list
            The list of points that define the intervals.
        n : int or list or numpy array
            The number of integration points per interval.
            If n is an integer, the same
            number of points is used per interval.

        Returns
        -------
        tensap.IntegrationRule
            The integration rule.

        """

        w = np.array([])
        x = np.array([])

        knots = np.sort(np.ravel(knots))

        if np.isscalar(n):
            n = np.tile(n, knots.size - 1)  # Replicate if p is scalar
        n = np.ravel(n)  # Ensure p is 1D

        for k in range(knots.size - 1):
            supp = knots[k: k + 2]
            g = tensap.LebesgueMeasure(supp[0], supp[1]).gauss_integration_rule(n[k])
            x = np.append(x, g.points)
            w = np.append(w, g.weights)

        return tensap.IntegrationRule(x, w)


class FullTensorProductIntegrationRule(IntegrationRule):
    """
    Class FullTensorProductIntegrationRule.

    """

    def __init__(self, points, weights):
        tensap.IntegrationRule.__init__(self, [], [])

        if isinstance(points, list):
            points = tensap.FullTensorGrid(points)
        elif not isinstance(points, tensap.FullTensorGrid):
            raise ValueError("The points must be a FullTensorGrid or a list.")

        assert (
            isinstance(weights, list) and len(weights) == points.ndim()
        ), "The weights must be a list of length the length of points."

        self.points = points
        self.weights = [np.ravel(x) for x in weights]

    def ndim(self):
        return self.points.ndim()

    def integrate(self, fun):
        Irule = self.integration_rule()
        return Irule.integrate(fun)

    def integration_rule(self):
        points = self.points.array()
        weights = self.weights_on_grid()
        return tensap.IntegrationRule(points, weights)

    def weights_on_grid(self):
        """
        Return the weights on the grid.

        Returns
        -------
        numpy.ndarray
            The weights on the grid.

        """
        weights = tensap.CanonicalTensor(
            [np.reshape(x, [-1, 1]) for x in self.weights], [1]
        )
        return np.ravel(weights.full().numpy(), "F")

    def gauss_legendre_composite(knots, n):
        """
        Returns a d-dimensional Gauss-Legendre quadrature associated
        with d-dimensional uniform grid. It uses tensorization of n-points
        gauss integration rule per hyperrectangle.
        knots is a list of length d: the resulting rule is the tensorization
        of 1-dimensional Piecewise Gauss Legendre integration rules
        associated with grids knots[0] ... knots[d-1]

        Parameters
        ----------
        knots : list or tuple of length d
            The tuple of grids.
        n : int
            The number of integration points per interval
            for 1-dimensional rules.

        Returns
        -------
        tensap.IntegrationRule
            The integration rule.

        """
        if isinstance(knots, list):
            knots = tuple(knots)
        if not isinstance(knots, tuple):
            raise ValueError("must provide a tuple of length d.")

        g = [tensap.IntegrationRule.gauss_legendre_composite(x, n) for x in knots]
        points = [x.points for x in g]
        weights = [x.weights for x in g]

        return tensap.FullTensorProductIntegrationRule(points, weights)
