'''
Module discrete_measure.

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


class DiscreteMeasure(tensap.Measure):
    '''
    Class DiscreteMeasure: discrete measure in R^d
    sum_{i=1}^n w_i delta_{x_i}.

    Attributes
    ----------
    values : numpy.ndarray
        Array containing the set of values (x_1,...,x_N) taken by the random
        variable, with x_i in R^d, i = 1, ..., N.
    weights : numpy.ndarray, optional
        Array containing the weights P(X=x_1), ..., P(X=x_N). The default
        is None, indicating to take weights equal to 1.

    '''

    def __init__(self, values, weights=None):
        '''
        Constructor for the class DiscreteMeasure.

        Parameters
        ----------
        values : list or numpy.ndarray
        Array containing the set of values (x_1,...,x_N) taken by the random
        variable, with x_i in R^d, i = 1, ..., N.
    weights : list or numpy.ndarray, optional
        Array containing the weights P(X=x_1), ..., P(X=x_N). The default
        is None, indicating to take weights equal to 1.

        Raises
        ------
        ValueError
            If the arguments do not have the same size.

        Returns
        -------
        None.

        '''
        tensap.Measure.__init__(self)

        self.values = np.array(values)
        if values.ndim == 1:
            values = np.reshape(values, [-1, 1])
        N = values.shape[0]
        if weights is None:
            weights = np.ones(N)
        elif np.size(weights) != N:
            raise ValueError('The arguments must have the same size.')
        self.weights = np.ravel(weights)

    def ndim(self):
        return self.values.shape[1]

    def mass(self):
        return np.sum(self.weights)

    def __eq__(self, Y):
        return isinstance(Y, (DiscreteMeasure, tensap.DiscreteMeasure)) and \
            np.array_equal(self.values, Y.values) and \
            np.array_equal(self.weights, Y.weights)

    def support(self):
        return [np.min(self.values, 0), np.max(self.values, 0)]

    def plot(self, *args):
        '''
        Plot a graphical representation of the discrete measure.

        Parameters
        ----------
        *args : tuple
            Additional parameters for matplotlib.pyplot's function vlines.

        Returns
        -------
        None.

        '''
        import matplotlib.pyplot as plt

        x = np.ravel(self.values)
        y = self.weights
        delta = np.max(x) - np.min(x)
        ax = [np.min(x)-delta/10, np.max(x)+delta/10]
        plt.vlines(x, np.zeros(y.shape), y, *args)
        plt.xlim(*ax)
        plt.ylim(0, 1.1*np.max(y))

    def integration_rule(self):
        '''
        Return the integration rule associated with the measure

        Returns
        -------
        tensap.IntegrationRule
            The integration rule associated with the measure

        '''
        return tensap.IntegrationRule(self.values, self.weights)

    def random(self, n=1):
        '''
        Generate n random numbers according to the probability distribution
        obtained by rescaling the DiscreteMeasure.

        Parameters
        ----------
        n : int, optional
            The number of random numbers generated. The default is 1.

        Returns
        -------
        numpy.ndarray
            The n generated random numbers.

        '''
        Y = tensap.DiscreteRandomVariable(np.reshape(
            np.arange(self.weights.size), [-1, 1]),
            self.weights/np.sum(self.weights))
        ind = Y.icdf(np.random.rand(n)).astype(int)
        return self.values[ind, :]
