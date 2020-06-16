'''
Module discrete_random_variable.

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


class DiscreteRandomVariable(tensap.RandomVariable):
    '''
    Class DiscreteRandomVariable.

    Attributes
    ----------
    values : list or numpy.ndarray
        Array containing the set of values (x_1,...,x_N) taken by the random
        variable, with x_i in R^d, i = 1, ..., N.
    probabilities : list or numpy.ndarray, optional
        Array containing the probabilities P(X=x_1), ..., P(X=x_N). The default
        is None, indicating to create a uniform law: P(X=x_i)=1/N,
        i = 1, ..., N.

    '''

    def __init__(self, values, probabilities=None):
        '''
        Discrete random variable taking a finite set of values in R^d.

        Parameters
        ----------
        values : list or numpy.ndarray
            Array containing the set of values (x_1,...,x_N) taken by the
            random variable, with x_i in R^d, i = 1, ..., N.
        probabilities : list or numpy.ndarray, optional
            Array containing the probabilities P(X=x_1), ..., P(X=x_N). The
            default is None, indicating to create a uniform law: P(X=x_i)=1/N,
            i = 1, ..., N.

        Raises
        ------
        ValueError
            If the provided probabilities are not authorized.

        Returns
        -------
        None.

        '''
        tensap.RandomVariable.__init__(self)

        if np.ndim(values) == 1:
            values = np.reshape(values, (-1, 1))

        self.values = values
        N = values.shape[0]
        if probabilities is None:
            probabilities = np.full(N, 1/N)
        elif len(probabilities) != N:
            raise ValueError('The arguments must have the same size.')
        elif not np.all([x >= 0 for x in probabilities]):
            raise ValueError('All the probabilities should be >= 0.')

        self.probabilities = np.array(probabilities)
        if np.abs(np.sum(self.probabilities) - 1) > np.finfo(float).eps:
            self.probabilities = self.probabilities / \
                np.sum(self.probabilities)

    def get_standard_random_variable(self):
        '''
        Return the standard discrete random variable.

        Returns
        -------
        tensap.UniformRandomVariable
            The standard discrete random variable.

        '''
        values = np.array(self.values)
        values = (values - np.tile(self.mean(), (values.shape[0], 1))) / \
            np.tile(self.std(), (values.shape[0], 1))
        return DiscreteRandomVariable(values, self.probabilities)

    def support(self):
        return np.vstack((np.min(self.values, 0), np.max(self.values, 0)))

    def plot(self, quantity, *args):
        '''
        Plot the desired quantity, chosen between 'pdf', 'cdf' or 'icdf'.

        Parameters
        ----------
        quantity : str
            The desired quantity, chosen between 'pdf', 'cdf' or 'icdf'.
        *args : tuple
            Additional parameters for matplotlib.pyplot's function step (for
            the cdf), vlines (for the pdf) or plot (for the icdf).

        Raises
        ------
        ValueError
            If the provided argument quantity is wrong.

        Returns
        -------
        None.

        '''
        import matplotlib.pyplot as plt

        x = np.ravel(self.values)
        y = self.probabilities
        delta = np.max(x) - np.min(x)
        ax = [np.min(x)-delta/10, np.max(x)+delta/10]

        if quantity == 'cdf':
            x = np.concatenate(([ax[0]], x, [ax[1]]))
            y = np.concatenate(([0, 0], np.cumsum(y)))
            plt.step(x, y, *args)
            plt.xlim(*ax)
            plt.ylim(0, 1.1*np.max(y))
        elif quantity == 'pdf':
            plt.vlines(x, np.zeros(y.shape), y, *args)
            plt.xlim(*ax)
            plt.ylim(0, 1.1*np.max(y))
        elif quantity == 'icdf':
            u = np.linspace(0, 1, 500)
            y = self.icdf(u)
            plt.plot(u, y, *args)
            plt.xlim(0, 1)
            plt.ylim(*ax)
        else:
            raise ValueError('Wront argument value.')
        plt.show()

    def integration_rule(self):
        '''
        Return the integration rule object associated with the discrete random
        variable.

        Returns
        -------
        tensap.IntegrationRule
            The integration rule object associated with the discrete random
            variable.

        '''
        return tensap.IntegrationRule(self.values, self.probabilities)

    def get_parameters(self):
        return self.values, self.probabilities

    def cdf(self, x):
        '''
        Compute the cumulative density function of X at x.

        Parameters
        ----------
        x : float, list or numpy.ndarray
            The points at which the cumulative density function of X is to be
            evaluated.

        Returns
        -------
        numpy.ndarray
            The evaluations of the cumulative density function of X at x.

        '''
        x = np.array(x)
        shape = x.shape
        M = x.size

        x = np.tile(np.reshape(x, (1, M)), (self.values.size, 1))
        v = np.tile(np.reshape(self.values, (-1, 1)), (1, M))
        w = np.tile(np.reshape(self.probabilities, (-1, 1)), (1, M))
        w[v > x] = 0
        return np.reshape(np.sum(w, 0), shape)

    def icdf(self, p):
        '''
        Compute the inverse cumulative density function of X at p (quantile).

        Parameters
        ----------
        p : float, list or numpy.ndarray
            The points at which the inverse cumulative density function of X is
            to be evaluated.

        Returns
        -------
        numpy.ndarray
            The evaluations of the inverse cumulative density function of X at
            x.

        '''
        p = np.array(p)
        shape = p.shape
        v = np.vstack(([-np.inf], np.reshape(self.values, (-1, 1))))
        F = np.vstack(([0], np.reshape(self.probabilities, (-1, 1))))

        M = p.size
        p = np.tile(np.reshape(p, (1, M)), (F.size, 1))
        F = np.tile(np.reshape(np.cumsum(F), (-1, 1)), (1, M))
        return np.reshape(v[np.nonzero(np.cumsum(np.transpose(F >= p), 1) ==
                                       1)[1]], shape)

    def mean(self):
        return np.matmul(np.transpose(self.probabilities), self.values)

    def var(self):
        '''
        Return the variance of the random variable.

        Returns
        -------
        float
            The variance of the random variable.

        '''
        return np.matmul(np.transpose(self.probabilities), self.values**2) - \
            self.mean()**2

    def random_variable_statistics(self):
        '''
        Return the mean and the variance of the discrete random variable.

        Returns
        -------
        float
            The mean of the random variable.
        float
            The variance of the random variable.

        '''
        return self.mean(), self.var()

    def random(self, n=1):
        '''
        Generate n random numbers according to the distribution of the
        RandomVariable.

        Parameters
        ----------
        n : int
            The number of random numbers generated.

        Returns
        -------
        numpy.ndarray
            The generated numbers.

        '''
        Y = DiscreteRandomVariable(np.arange(len(self.probabilities)),
                                   self.probabilities)
        ind = Y.icdf(np.random.rand(int(n)))
        return np.squeeze(self.values[ind.astype(int), :], 1)
