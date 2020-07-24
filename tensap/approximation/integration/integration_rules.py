'''
Module integration_rules.

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


class IntegrationRule:
    '''
    Class IntegrationRule.

    Attributes
    ----------
    points : numpy.ndarray
        The integration points.
    weights : numpy.ndarray
        The integration weights.

    '''

    def __init__(self, points, weights):
        '''
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

        '''
        self.points = np.atleast_1d(points)
        self.weights = np.atleast_1d(weights)

    def ndim(self):
        '''
        Return the number of integration points.

        Returns
        -------
        int
            The number of integration points.

        '''
        return self.points.shape[1]

    def integrate(self, fun):
        '''
        Integrate the function fun using the integration rule.

        Parameters
        ----------
        fun : function
            The function to be integrated.

        Returns
        -------
        int
            The integrated function using the integration rule.

        '''
        fun_eval = fun(self.points)
        return np.dot(np.ravel(fun_eval), self.weights)

    def tensorize(self, dim):
        '''
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

        '''
        points = tensap.FullTensorGrid(self.points, dim)
        weights = [self.weights]*dim
        return tensap.FullTensorProductIntegrationRule(points, weights)

    @staticmethod
    def gauss(random_variable, *args):
        '''
        Call the method gauss_integration_rule of tensap.RandomVariable or
        tensap.RandomVector.

        Parameters
        ----------
        random_variable : tensap.RandomVariable or tensap.RandomVector
            The random variable or vector associated to which the integration
            rule is to be computed.
        *args : misc
            Additional arguments.

        Returns
        -------
        tensap.IntegrationRule
            The integration rule associated with the random variable or vector.

        '''
        return random_variable.gauss_integration_rule(*args)


class FullTensorProductIntegrationRule(IntegrationRule):
    '''
    Class FullTensorProductIntegrationRule.

    '''

    def __init__(self, points, weights):
        tensap.IntegrationRule.__init__(self, points, weights)

        if isinstance(points, list):
            points = tensap.FullTensorGrid(points)
        elif not isinstance(points, tensap.FullTensorGrid):
            raise ValueError('The points must be a FullTensorGrid or a list.')

        assert isinstance(weights, list) and len(weights) == points.ndim(), \
            'The weights must be a list of length the length of points.'

        self.points = points
        self.weights = [np.ravel(x) for x in weights]

    def ndim(self):
        return self.points.ndim()

    def integrate(self, fun):
        points = self.points.array()
        weights = self.weights_on_grid()
        try:
            f_x = fun.eval(points)
        except Exception:
            try:
                f_x = fun(points)
            except Exception:
                raise ValueError('The function must be evaluable.')
        return np.sum(weights * f_x)

    def weights_on_grid(self):
        '''
        Return the weights on the grid.

        Returns
        -------
        numpy.ndarray
            The weights on the grid.

        '''
        weights = tensap.CanonicalTensor([np.reshape(x, [-1, 1]) for
                                          x in self.weights], [1])
        return np.ravel(weights.full().numpy())
