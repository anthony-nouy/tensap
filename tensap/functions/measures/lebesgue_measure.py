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
Module lebesgue_variable.

"""

import numpy as np
from numpy import array
from numpy.random import rand
import tensap


class LebesgueMeasure(tensap.Measure):
    """
    Class LebesgueMeasure.

    Attributes
    ----------
    a : float, optional
        The lower bound of the support of the random variable. The default
        is 0.
    b : float, optional
        The upper bound of the support of the random variable. The default
        is 1.

    """

    def __init__(self, a=0, b=1):
        """
            Lebesgue Measure on [a,b].

            Parameters
            ----------
        a : float, optional
            The lower bound of the support of the random variable. The default
            is 0.
        b : float, optional
            The upper bound of the support of the random variable. The default
            is 1.

            Returns
            -------
            None.

        """
        tensap.Measure.__init__(self)

        self.a = a
        self.b = b

    def __repr__(self):
        return ("<{} on [{}, {}]>").format(self.__class__.__name__, self.a, self.b)

    def shift(self, m, s):
        """
        If self a Lebesgue measure on [a,b], returns the Lebesgue measure on interval [m+sa,m+sb]

        Parameters
        ----------
        m : float
        s : float

        Returns
        -------
        tensap.LebesgueMeasure

        """
        shifted = tensap.LebesgueMeasure(self.a, self.b)
        shifted.a = shifted.a * s + m
        shifted.b = shifted.b * s + m
        return shifted

    @staticmethod
    def ndim():
        """
        Return the dimension of the random variable, equal to 1.

        Returns
        -------
        int
            The dimension of the random variable.

        """
        return 1

    def support(self):
        """
        Return the support of the measure.

        Returns
        -------
        numpy.ndarray

        """
        return array([self.a, self.b])

    def truncated_support(self):
        """
        Return the support of the measure.

        Returns
        -------
        numpy.ndarray

        """
        if self.mass() == np.inf:
            raise ValueError(
                "truncated support not defined \
                             for Lebesgue measure on unbounded domain."
            )

        return self.support()

    def __eq__(self, L2):
        if not (isinstance(self, LebesgueMeasure) and isinstance(L2, LebesgueMeasure)):
            is_equal = False
        else:
            is_equal = True
            param_1 = self.get_parameters()
            param_2 = L2.get_parameters()
            for ind in zip(param_1, param_2):
                is_equal = is_equal and (ind[0] == ind[1])
        return is_equal

    def __neq__(self, L2):
        return not (self == L2)

    def mass(self):
        """
        Return the mass of the measure.

        Returns
        -------
        numpy.ndarray

        """

        return self.b - self.a

    def gauss_integration_rule(self, nb_pts):
        """
        Return the nb_pts-points gauss integration
        rule associated with the
        measure of self, using Golub-Welsch algorithm.

        Parameters
        ----------
        nb_pts : int
            The number of integration points.

        Returns
        -------
        tensap.IntegrationRule
            The integration rule associated with the measure of self.

        """
        RV = tensap.UniformRandomVariable(self.a, self.b)
        I0 = RV.gauss_integration_rule(nb_pts)
        I0.weights = I0.weights * self.mass()

        return I0

    def orthonormal_polynomials(self):
        """
        Return orthonormal polynomials associated with
        the LebesgueMeasure on [a,b].

        """
        poly = tensap.LegendrePolynomialsLebesgue()

        if self != LebesgueMeasure(-1, 1):
            poly = tensap.ShiftedOrthonormalPolynomials(
                poly, (self.a + self.b) / 2, (self.b - self.a) / 2
            )
        return poly

    def get_parameters(self):
        """
        Return the parameters of the Lebesgue measure on [a,b].

        Returns
        -------
        float
            The lower bound of the support of the measure.
        float
            The upper bound of the support of the measure

        """
        return self.a, self.b

    def random(self, n=1):
        return rand(int(n)) * (self.b - self.a) + self.a
