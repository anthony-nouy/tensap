'''
Module random_variable.

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

from copy import deepcopy
import numpy as np
import tensap


class RandomVariable(tensap.ProbabilityMeasure):
    '''
    Class RandomVariable.

    Attributes
    ----------
    moments : numpy.array
        The moments of the normal random variable (if computed).

    '''

    def __init__(self):
        '''
        Constructor of the class RandomVariable.

        The moments attribute remains empty as long as the moments have not
        been computed using the method moment.

        Returns
        -------
        None.

        '''
        self.moments = np.array([])

    @staticmethod
    def ndim():
        '''
        Return the dimension of the random variable, equal to 1.

        Returns
        -------
        int
            The dimension of the random variable.

        '''
        return 1

    def __eq__(self, rv_2):
        if not (isinstance(self, tensap.RandomVariable) and
                isinstance(rv_2, tensap.RandomVariable)):
            is_equal = False
        elif not isinstance(self, type(rv_2)):
            is_equal = False
        else:
            is_equal = True
            param_1 = self.get_parameters()
            param_2 = rv_2.get_parameters()
            for ind in zip(param_1, param_2):
                is_equal = is_equal and (ind[0] == ind[1])
        return is_equal

    def __neq__(self, rv_2):
        return not (self == rv_2)

    def number_of_parameters(self):
        '''
        Compute the number of parameters that admits the random variable.

        Returns
        -------
        int
            The number of parameters that admits the random variable.

        '''
        return np.size(self.get_parameters())

    def pdf(self, x):
        '''
        Compute the probability density function (pdf) of the RandomVariable
        at points x.

        Parameters
        ----------
        x : float or list or numpy.ndarray
            The points at which the pdf is to be evaluated.

        Returns
        -------
        numpy.ndarray
            The evaluations of the pdf at points x.

        '''
        raise NotImplementedError('No generic implementation of the method.')

    def cdf(self, x):
        '''
        Compute the cumulative distribution function (cdf) of the
        RandomVariable at points x.

        Parameters
        ----------
        x : float or list or numpy.ndarray
            The points at which the cdf is to be evaluated.

        Returns
        -------
        numpy.ndarray
            The evaluations of the cdf at points x.

        '''
        raise NotImplementedError('No generic implementation of the method.')

    def icdf(self, x):
        '''
        Compute the inverse cumulative distribution function (icdf) of the
        RandomVariable at points x.

        Parameters
        ----------
        x : float or list or numpy.ndarray
            The points at which the icdf is to be evaluated.

        Returns
        -------
        numpy.ndarray
            The evaluations of the icdf at points x.

        '''
        raise NotImplementedError('No generic implementation of the method.')

    def iso_probabilistic_grid(self, n):
        '''
        Return a set of n+1 points (x_0, ..., x_{n}) such that the n sets
        (x0, x_1), [x_1, x_2) ... [x_{n-1}, x_{n})  have all the same
        probability p = 1/n (with x0 = self.min() and x_{n+1}=self.max()).

        Parameters
        ----------
        n : int
            The number of points of the grid plus one.

        Returns
        -------
        numpy.ndarray
            The iso-probabilistic grid.

        '''
        if n < 1:
            n = int(np.ceil(1/n))

        if n == 1:
            g = []
        elif n == 2:
            g = [self.icdf(0.5)]
        else:
            g = self.icdf(np.linspace(1/n, 1-1/n, n-1))
        return np.concatenate(([self.min()], g, [self.max()]))

    def discretize(self, n):
        '''
        Return a discrete random variable taking n possible values x1, ..., xn,
        these values being the quantiles of self of probability 1/(2n) + i/n,
        i=0n ..., n-1 and such that P(Xn >= xn) = 1/n.

        Parameters
        ----------
        n : int
            The number of possible values the discrete random variable can
            take.

        Returns
        -------
        tensap.DiscreteRandomVariable
            The obtained discrete random variable.

        '''
        u = np.linspace(1/(2*n), 1-1/(2*n), n)
        x = self.icdf(u)
        return tensap.DiscreteRandomVariable(x)

    def gauss_integration_rule(self, nb_pts):
        '''
        Return the nb_pts-points gauss integration rule associated with the
        measure of self, using Golub-Welsch algorithm.

        Parameters
        ----------
        nb_pts : int
            The number of integration points.

        Returns
        -------
        tensap.IntegrationRule
            The integration rule associated with the measure of self.

        '''
        poly = self.orthonormal_polynomials(nb_pts+1)
        if isinstance(poly, tensap.ShiftedOrthonormalPolynomials):
            shift = poly.shift
            scaling = poly.scaling
            poly = poly.polynomials
            flag = True
        else:
            flag = False

        coef = poly._recurrence_coefficients
        if coef.shape[1] < nb_pts:
            coef = poly.recurrence(poly.measure, nb_pts-1)
        else:
            coef = coef[:, :nb_pts]

        # Jacobi matrix
        if nb_pts == 1:
            jacobi_matrix = np.diag(coef[0, :])
        else:
            jacobi_matrix = np.diag(coef[0, :]) + \
                np.diag(np.sqrt(coef[1, 1:]), -1) + \
                np.diag(np.sqrt(coef[1, 1:]), 1)

        # Quadrature points are the eigenvalues of the Jacobi matrix, weights
        # are deduced from the eigenvectors
        eig_values, eig_vectors = np.linalg.eig(jacobi_matrix)
        points = np.sort(eig_values)
        ind = np.argsort(eig_values)
        eig_vectors = eig_vectors[:, ind]

        weights = eig_vectors[0, :]**2 / np.sqrt(np.sum(eig_vectors**2, 0))

        if flag:
            points = shift + scaling * points
        return tensap.IntegrationRule(points, weights)

    def lhs_random(self, n, p=1):
        '''
        Latin Hypercube Sampling of the random variable self of n points in
        dimension p.

        Requires the package pyDOE.

        Parameters
        ----------
        n : int
            Number of points.
        p : int, optional
            The dimension. The default is 1.

        Returns
        -------
        numpy.ndarray
            The coordinates of the Latin Hypercube Sampling in each dimension.

        '''
        from pyDOE import lhs
        A = lhs(p, samples=n)
        U = tensap.UniformRandomVariable(0, 1)
        A = [U.transfer(self, A[:, i]) for i in range(A.shape[1])]
        return np.transpose(np.array(A))

    def likelihood(self, x):
        '''
         Compute the log-likelihood of the random variable on sample x.

        Parameters
        ----------
        x : list or numpy.ndarray
            The sample used to compute the log-likelihood.

        Returns
        -------
        float
            The log-likelihood of the random variable on sample x.

        '''
        P = self.pdf(x)
        return np.sum(np.log(P + np.finfo(float).eps))

    def max(self):
        '''
        Compute the maximum value that can take the inverse cumulative
        distribution function of the random variable.

        Returns
        -------
        float
            The maximum value that can take the inverse cumulative distribution
            function of the random variable.

        '''
        return np.max(self.support())

    def min(self):
        '''
        Compute the minimum value that can take the inverse cumulative
        distribution function of the random variable.

        Returns
        -------
        float
            The minimum value that can take the inverse cumulative distribution
            function of the random variable.

        '''
        return np.min(self.support())

    def moment(self, ind, nargout=1):
        '''
        Compute the moments of self of orders contained in ind, defined as
        E(X^ind[î]).
        If a second output argument is asked, the computed moments are stored
        in the random variable X.

        Parameters
        ----------
        ind : list or numpy.ndarray
            The orders of the moments.
        nargout : int, optional
            Indicates the number of expected outputs. The default is 1,
            indicating to return only the moments.

        Returns
        -------
        numpy.ndarray
            The computed moments.
        tensap.RandomVariable
            The RandomVariable object with the computed moments stored in the
            attribute moments.

        '''
        ind = np.atleast_1d(ind)
        if np.size(self.moments)-1 >= np.max(ind):
            out = self.moments[ind]
        else:
            out = np.zeros(np.size(ind))
            nb_pts = int(np.ceil((np.max(ind)+1)/2))
            G = self.gauss_integration_rule(nb_pts)
            for nb, ind_loc in enumerate(ind):
                out[nb] = G.integrate(lambda x: x ** ind_loc)

        if nargout == 1:
            return out
        X = deepcopy(self)
        X.moments = out
        return out, X

    def mean(self):
        '''
        Return the mean of the random variable.

        Returns
        -------
        float
            The mean of the random variable.

        '''
        return self.random_variable_statistics()[0]

    def std(self):
        '''
        Return the standard deviation of the random variable.

        Returns
        -------
        float
            The standard deviation of the random variable.

        '''
        return np.sqrt(self.random_variable_statistics()[1])

    def variance(self):
        '''
        Return the variance of the random variable.

        Returns
        -------
        float
            The variance of the random variable.

        '''
        return self.random_variable_statistics()[1]

    def transfer(self, Y, x):
        '''
        Transfer from the random variable self to the random variable Y at
        points x.

        Parameters
        ----------
        Y : tensap.RandomVariable
            The target RandomVariable of the transfer.
        x : list or numpy.ndarray
            The input points.

        Returns
        -------
        y : numpy.ndarray
            The transfered points.

        '''
        assert isinstance(self, tensap.RandomVariable) and \
            isinstance(Y, tensap.RandomVariable), \
            'The first two arguments must be RandomVariable.'
        return Y.icdf(self.cdf(x))

    def truncated_support(self):
        '''
        Return the truncated support of the random variable.

        Returns
        -------
        sup : numpy.ndarray
            The truncated support of the random variable.

        '''

        sup = self.support()
        if sup[0] == -np.inf:
            sup[0] = self.mean() - 10*self.std()
        if sup[1] == np.inf:
            sup[1] = self.mean() + 10*self.std()
        return sup

    def pdf_plot(self, *args):
        '''
        Plot the probability density function (pdf) of the random variable.

        See also plot.

        Parameters
        ----------
        *args : tuple
            Additional parameters of the method plot.

        Returns
        -------
        None.

        '''
        self.plot('pdf', *args)

    def cdf_plot(self, *args):
        '''
        Plot the cumulative distribution function (cdf) of the random variable.

        See also plot.

        Parameters
        ----------
        *args : tuple
            Additional parameters of the method plot.

        Returns
        -------
        None.

        '''
        self.plot('cdf', *args)

    def icdf_plot(self, *args):
        '''
        Plot the inverse cumulative distribution function (icdf) of the random
        variable.

        See also plot.

        Parameters
        ----------
        *args : tuple
            Additional parameters of the method plot.

        Returns
        -------
        None.

        '''
        self.plot('icdf', *args)

    def plot(self, quantity, n_pts=100, bar=False, *args):
        '''
        Plot the desired quantity, chosen between 'pdf', 'cdf' or 'icdf'.

        Parameters
        ----------
        quantity : str
            The desired quantity, chosen between 'pdf', 'cdf' or 'icdf'.
        n_pts : int, optional
            The number of points used for the plot. The default is 100.
        bar : boolean, optional
            Determines if the method uses matplotlib.pyplot's function bar
            or plot. The default is False.
        *args : tuple
            Additional parameters for matplotlib.pyplot's function plot or bar.

        Raises
        ------
        ValueError
            If the provided argument quantity is wrong.

        Returns
        -------
        None.

        '''
        import matplotlib.pyplot as plt

        sup = self.truncated_support()
        if quantity == 'cdf':
            x = np.linspace(sup[0], sup[1], n_pts)
            P = self.cdf(x)
        elif quantity == 'icdf':
            x = np.linspace(0, 1, n_pts)
            P = self.icdf(x)
        elif quantity == 'pdf':
            x = np.linspace(sup[0], sup[1], n_pts)
            P = self.pdf(x)
        else:
            raise ValueError('Wrong argument value')

        if bar:
            plt.bar(x, P, *args)
        else:
            plt.plot(x, P, *args)
        plt.show()

    def get_parameters(self):
        '''
        Return the parameters of the random variable.

        '''
        raise NotImplementedError('No generic implementation of the method.')

    def random_variable_statistics(self):
        '''
        Return the mean and the variance of the random variable.

        Returns
        -------
        float
            The mean of the random variable.
        float
            The variance of the random variable.

        '''
        raise NotImplementedError('No generic implementation of the method.')

    def random(self, n):
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
        raise NotImplementedError('No generic implementation of the method.')

    def orthonormal_polynomials(self, *max_degree):
        '''
        Return the max_degree-1 first orthonormal polynomials associated with
        the RandomVariable.

        Parameters
        ----------
        max_degree : int, optional
            The maximum degree of the returned polynomials. The default is
            None, choosing the default maximum degree associated with the
            constructor of the polynomials.

        Returns
        -------
        poly : tensap.OrthonormalPolynomials
            The generated orthonormal polynomials.

        '''
        raise NotImplementedError('No generic implementation of the method.')
