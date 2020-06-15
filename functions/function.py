'''
Module function.

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

from abc import abstractmethod
import numpy as np
import tensap


class Function:
    '''
    Class Function.

    Attributes
    ----------
    dim : int
        The dimension of the input of the function.
    measure : tensap.Measure
        The measure associated with the function.
    output_shape : int or list or numpy.ndarray
        The shape of the output of the function.
    evaluation_at_multiple_points : bool
        Indicates if the function can be evaluated at multiple points at once.
    store : bool
        Indicates if the Function object should store the evaluations of the
        function.

    '''

    def __init__(self):
        '''
        Constructor for the class Function.

        Returns
        -------
        None.

        '''
        self.dim = []
        self.measure = None
        self.output_shape = 1
        self.evaluation_at_multiple_points = True
        self.store = False
        self._x_stored = []
        self._y_stored = []

    def __call__(self, x, return_f=False):
        if np.ndim(x) == 1:
            x = np.expand_dims(x, 1)

        if self.store:
            y = self.store_eval(x)
            if np.ndim(y) == 2 and np.shape(y)[1] == 1:
                y = np.squeeze(y, axis=1)
            return y
        else:
            y = self.eval(x)
            if np.ndim(y) == 2 and np.shape(y)[1] == 1:
                y = np.squeeze(y, axis=1)
            f = self
            if return_f:
                return y, f
            return y

    def store_eval(self, x):
        '''
        Evaluate the function, reuising previous evaluations if possible, and
        storing the new evaluations in self.

        Parameters
        ----------
        x : numpy.ndarray
            The input points.

        Returns
        -------
        y : numpy.ndarray
            The evaluations of the Function.
        tensap.Function
            The Function with stored evaluations, for future reuse.

        '''
        if np.ndim(x) < 2:
            x = np.reshape(x, [np.size(x), 1])

        if self.store and np.size(self._y_stored) != 0:
            ind_2, ind_1 = np.nonzero(np.all(x == self._x_stored[:,
                                                                 np.newaxis],
                                             axis=2))

            if np.prod(self.output_shape) != 1:
                self._y_stored = np.reshape(self._y_stored,
                                            (self._y_stored.shape[0], -1),
                                            order='F')

            y = np.zeros((x.shape[0], self._y_stored.shape[1]))
            y[ind_1, :] = self._y_stored[ind_2, :]

            x_new = x[np.setdiff1d(range(x.shape[0]), ind_1), :]
            if x_new.size != 0:
                y_new = self.eval(x_new)
                y[np.setdiff1d(range(x.shape[0]), ind_1), :] = \
                    np.reshape(y_new, (y_new.shape[0], -1))
                self._x_stored = np.vstack((self._x_stored, x_new))
                self._y_stored = np.vstack((self._y_stored,
                                            np.reshape(y_new,
                                                       (y_new.shape[0], -1))))
            y = np.reshape(y, np.concatenate(([y.shape[0]],
                                              np.atleast_1d(
                                                  self.output_shape))),
                           order='F')
            if np.prod(self.output_shape) != 1:
                self._y_stored = np.reshape(self._y_stored, np.concatenate(([
                    self._y_stored.shape[0]], np.atleast_1d(
                        self.output_shape))), order='F')
        else:
            y = self.eval(x)
            if np.ndim(y) < 2:
                y = np.reshape(y, [np.size(y), 1])

            self.store = True
            self._x_stored = x
            self._y_stored = y

        return np.squeeze(y), self

    def fplot(self, support=None, n_points=100, *args, **kwargs):
        '''
        Plot the function on a support using a given number of points.

        Parameters
        ----------
        support : list or numpy.ndarray, optional
            The support of the plot. The default is None, indicating to use
            the truncated_support of self.measure.
        n_points : int, optional
            The number of points used for the plot. The default is 100.
        *args : tuple
            Additional parameters used by the function matplotlib.pyplot.plot.

        Returns
        -------
        None.

        '''
        import matplotlib.pyplot as plt
        if support is None:
            support = self.measure.truncated_support()

        x = np.linspace(support[0], support[1], int(n_points))
        plt.plot(x, self.eval(x), *args, **kwargs)
        plt.show()

    def surf(self, n=None, *args):
        '''
        Surface plot of the bivariate function.

        Parameters
        ----------
        n : list or numpy.ndarray, optional
            The number of points used for the surface plot in each dimension.
            The default is [1000, 1000].
        *args : tuple
            Additional parameters used by matplotlib.pyplot's plot_surface
            function.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The surface plot as a matplotlib.axes._subplots.AxesSubplot object.

        '''
        assert self.measure is not None, 'Attribute measure is empty.'
        assert self.dim == 2, \
            ('The function should be a bivariate function, use the partial ' +
             'evaluation for higher-dimensional function.')

        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        if n is None:
            n = [1000, 1000]

        sup = self.measure.truncated_support()

        if np.size(n) == 1:
            n = np.tile(n, 2)

        grids = [np.linspace(sup[0][0], sup[0][1], n[0]),
                 np.linspace(sup[1][0], sup[1][1], n[1])]
        grids[0] = grids[0][1:-1]
        grids[1] = grids[1][1:-1]
        grids = [np.reshape(x, (x.size, -1)) for x in grids]

        grid = tensap.FullTensorGrid(grids)
        fg = self.eval_on_tensor_grid(grid)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.meshgrid(grids[0], grids[1])
        ax.plot_surface(x, y, fg.data, *args)
        plt.show()

        return ax

    def partial_evaluation(self, not_alpha, x_not_alpha):
        '''
        Return the partial evaluation of a function
        f(x) = f(x_alpha,x_not_alpha), a function
        f_alpha(.) = f(., x_not_alpha) for fixed values x_not_alpha of the
        variables with indices not_alpha.

        Parameters
        ----------
        not_alpha : list or numpy.ndarray
            The indices of the fixed variables.
        x_not_alpha : numpy.ndarray
            The points at which the function is evaluated in the dimensions
            not_alpha.

        Raises
        ------
        ValueError
            If the Function has an empty attribute dim.

        Returns
        -------
        f_alpha : tensap.UserDefinedFunction
            The partial evaluation of the Function.

        '''
        if self.dim is None or np.size(self.dim) == 0:
            raise ValueError('The Function has an empty attribute dim.')

        alpha = np.setdiff1d(range(self.dim), not_alpha)
        ind = [np.nonzero(np.concatenate((alpha, not_alpha)) == x)[0][0] for
               x in range(self.dim)]

        def fun(x_alpha):
            grid = tensap.FullTensorGrid([x_alpha, x_not_alpha])
            return self.eval(grid.array()[:, ind])

        f_alpha = tensap.UserDefinedFunction(fun, alpha.size)
        f_alpha.store = self.store
        f_alpha.evaluation_at_multiple_points = \
            self.evaluation_at_multiple_points
        f_alpha.measure = self.measure.marginal(alpha)

        return f_alpha

    def random(self, n=1, measure=None):
        '''
        Evaluates the function at n points x drawn randomly according to the
        ProbabilityMeasure in measure if provided, or in self.measure.

        Parameters
        ----------
        n : int, optional
            The number of random evaluations. The default is 1.
        measure : tensap.ProbabilityMeasure, optional
            The probability measure according to which the points x are drawn.
            The default is None, indicating to use self.measure.

        Raises
        ------
        ValueError
            If the provided measure is not a tensap.ProbabilityMeasure.

        Returns
        -------
        numpy.ndarray
            The evaluations of the function at the points x.
        x : numpy.ndarray
            The points at which the function is to be evaluated.

        '''
        if measure is None:
            if isinstance(self.measure, tensap.ProbabilityMeasure):
                measure = self.measure
            else:
                raise ValueError('Must provide a ProbabilityMeasure.')
        elif not isinstance(self.measure, tensap.ProbabilityMeasure):
            raise ValueError('Must provide a ProbabilityMeasure.')

        x = measure.random(n)
        return self.eval(x), x

    def eval_on_tensor_grid(self, x):
        '''
        Evaluate the Function on a grid x.

        Parameters
        ----------
        x : tensap.TensorGrid
            The tensap.ensorGrid used for the evaluation..

        Raises
        ------
        NotImplementedError
            If the method is not implemented.
        ValueError
            If x is not a tensap.TensorGrid object.

        Returns
        -------
        fx : numpy.ndarray
            The evaluation of the Function on the grid.

        '''
        x_a = x.array()
        fx = self.eval(x_a)

        if isinstance(x, tensap.SparseTensorGrid):
            if np.all(self.output_shape == 1):
                fx = tensap.SparseTensor(fx, x.indices, x.shape)
            else:
                raise NotImplementedError('Method not implemented.')
        elif isinstance(x, tensap.FullTensorGrid):
            if np.all(self.output_shape == 1):
                shape = x.shape
                if self.dim > 1:
                    fx = np.reshape(fx, shape, order='F')
            else:
                shape = np.concatenate((np.atleast_1d(x.shape),
                                        np.atleast_1d(self.output_shape)))
                fx = np.reshape(fx, shape, order='F')
            fx = tensap.FullTensor(fx, np.size(shape), shape)
        else:
            raise ValueError('A TensorGrid object must be provided.')

        return fx

    def test_error(self, g, n=1000, measure=None):
        '''
        Compute the test error associated with the function, using a function g
        or some of its evaluations as a reference. A measure can be provided
        to randomly draw the test input data.

        Parameters
        ----------
        g : tensap.Function or numpy.ndarray
            The reference function or evaluations of it.
        n : int or numpy.ndarray, optional
            The test sample size, or the test input data. The default is 1000
            test input points.
        measure : tap.ProbabilityMeasure, optional
            A probability measure used to draw the test input data. The default
            is None, indicating to either use self.measure, g.measure or the
            provided input points.

        Returns
        -------
        err_l2 : float
            The L2 error.
        err_linf : float
            The L-infinity error.

        '''
        if measure is not None:
            x_test = measure.random(n)
            g_x_test = g.eval(x_test)
            err_l2, err_linf = self.test_error(g_x_test, x_test)
        else:
            if isinstance(g, tensap.Function) and np.size(n) == 1:
                if self.measure is not None:
                    measure = self.measure
                else:
                    measure = g.measure
                err_l2, err_linf = self.test_error(g, n, measure)
            elif np.size(n) != 1:
                x_test = n
                n = np.shape(x_test)[0]
                f_x_test = self.eval(x_test)
                if isinstance(g, tensap.Function):
                    g_x_test = g.eval(x_test)
                else:
                    assert np.shape(g)[0] == n, \
                        ('The number of evaluations does not match the ' +
                         'number of points.')
                    g_x_test = np.array(g)

                f_x_test = np.reshape(f_x_test, (n, -1), order='F')
                g_x_test = np.reshape(g_x_test, (n, -1), order='F')
                err_l2 = np.linalg.norm(f_x_test - g_x_test) / \
                    np.linalg.norm(g_x_test)
                err_linf = np.linalg.norm(np.sqrt(np.sum((f_x_test -
                                                          g_x_test)**2, 1)),
                                          np.inf) / \
                    np.linalg.norm(np.sqrt(np.sum(g_x_test**2, 1)), np.inf)

        return err_l2, err_linf

    @abstractmethod
    def eval(self, x):
        '''
        Evaluate the function at the points x.

        Parameters
        ----------
        x : list or numpy.ndarray
            The points at which the function is to be evaluated.

        Returns
        -------
        numpy.ndarray
            The evaluations of the function at the points x.

        '''
