'''
Module multivariate_functions_benchmark

Copyright (c) 2020, Anthony Nouy, Erwan Grelier
This file is part of tensap (tensor approximation package).

tensap is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

tensap is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with tensap.  If not, see <https://www.gnu.org/licenses/>.

'''

import sys
import numpy as np
sys.path.insert(0, './../../../')
import tensap


def multivariate_functions_benchmark(case, *args):
    '''
    Generate different functions used to benchmark the package.

    Parameters
    ----------
    case : str
        The name of the function. Can be 'borehole', 'ishigami',
        'sin_of_a_sum', 'linear_additive', 'linear_rank_one',
        'quadratic_rank_one', 'canonical_rank_2', 'mixture', 'field',
        'oscillatory', 'product_peak', 'corner_peak', 'gaussian', 'continuous',
        'discontinuous', 'henon_heiles', 'sobol', 'anisotropic', 'polynomial'.
    *args : tuple
        Parameters of the function.

    Raises
    ------
    NotImplementedError
        If the function is not implemented.

    Returns
    -------
    function
        The asked function.
    tensap.RandomVector
        Input random variables.

    '''
    if case == 'borehole':
        X = np.empty(8, dtype=object)
        X[0] = tensap.NormalRandomVariable(0.1, 0.0161812)
        X[1] = tensap.NormalRandomVariable(0, 1)
        X[2] = tensap.UniformRandomVariable(63070, 115600)
        X[3] = tensap.UniformRandomVariable(990, 1110)
        X[4] = tensap.UniformRandomVariable(63.1, 116)
        X[5] = tensap.UniformRandomVariable(700, 820)
        X[6] = tensap.UniformRandomVariable(1120, 1680)
        X[7] = tensap.UniformRandomVariable(9855, 12045)
        X = tensap.RandomVector(X)

        def fun(x):
            return 2 * np.pi * x[:, 2] * (x[:, 3] - x[:, 5]) / \
                (np.log(np.exp(7.71 + 1.0056*x[:, 1]) / x[:, 0]) *
                 (1 + 2 * x[:, 6] * x[:, 2] /
                  np.log(np.exp(7.71 + 1.0056 * x[:, 1]) / x[:, 0]) /
                  x[:, 0]**2 / x[:, 7] + x[:, 2] / x[:, 4]))

    elif case == 'ishigami':
        if len(args) == 0:
            d = 3
        else:
            d = args[0]
        if len(args) <= 1:
            a = 7
        else:
            a = args[1]
        if len(args) <= 2:
            b = 0.1
        else:
            b = args[2]

        X = tensap.RandomVector(tensap.UniformRandomVariable(-np.pi, np.pi), d)

        def fun(x):
            return np.sin(x[:, 0]) + a*np.sin(x[:, 1])**2 + \
                b*x[:, 2]**4*np.sin(x[:, 0])

    elif case == 'sin_of_asum':
        if len(args) == 0:
            d = 3
        else:
            d = args[0]
        if len(args) <= 1:
            def fun(x):
                return np.sin(np.sum(x, 1))
        else:
            def fun(x):
                return np.sin(np.matmul(x, np.reshape(args[1], (-1, 1))))

        X = tensap.RandomVector(tensap.UniformRandomVariable(), d)

    elif case == 'linear_additive':
        if len(args) == 0:
            d = 3
        else:
            d = args[0]
        if len(args) <= 1:
            def fun(x):
                return np.sum(x, 1)
        else:
            def fun(x):
                return np.matmul(x, np.reshape(args[1], (-1, 1)))

        X = tensap.RandomVector(tensap.UniformRandomVariable(), d)

    elif case == 'linear_rank_one':
        if len(args) == 0:
            d = 3
        else:
            d = args[0]
        if len(args) <= 1:
            w = np.zeros((1, d))
        else:
            w = np.reshape(args[1], (1, -1))

        X = tensap.RandomVector(tensap.UniformRandomVariable(), d)

        def fun(x):
            return np.prod(np.tile(w, (np.shape(x)[0], 1)) + 2*x, 1)

    elif case == 'quadratic_rank_one':
        d = args[0]
        c = args[1]

        X = tensap.RandomVector(tensap.UniformRandomVariable(), d)

        def fun(x):
            return np.prod(np.tile(c[0, :], (np.shape(x)[0], 1)) +
                           np.tile(c[1, :], (np.shape(x)[0], 1))*x +
                           np.tile(c[2, :], (np.shape(x)[0], 1))*x**2, 1)

    elif case == 'canonical_rank_2':
        if len(args) == 0:
            d = 3
        else:
            d = args[0]

        X = tensap.RandomVector(tensap.UniformRandomVariable(), d)

        def fun(x):
            return x[:, 0] * x[:, 1] * x[:, 2] + x[:, 0]**2 + x[:, 1]

    elif case == 'mixture':
        if len(args) == 0:
            d = 6
        else:
            d = args[0]

        X = tensap.RandomVector(tensap.UniformRandomVariable(), d)

        def fun(x):
            return np.sin(x[:, 0] + x[:, 3])*np.exp(x[:, 4])*x[:, 5] + \
                np.sin(x[:, 2]*x[:, 3])*x[:, 5]

    elif case == 'field':
        if len(args) == 0:
            d = 6
        else:
            d = args[0]

        X = tensap.RandomVector(tensap.UniformRandomVariable(0, 1), d)

        def fun(x):
            return 1 + np.cos(x[:, 0])*x[:, 1] + np.sin(x[:, 0])*x[:, 2] + \
                np.exp(x[:, 0])*x[:, 3] + 1/(x[:, 0]+1)*x[:, 4] + \
                1/(2*x[:, 0]+3)*x[:, 5]

    elif case in ('oscillatory', 'product_peak', 'corner_peak', 'gaussian',
                  'continuous', 'discontinuous'):
        if len(args) == 0:
            d = 10
        else:
            d = args[0]

        w = np.random.rand(1, d)
        c = np.random.rand(1, d)

        X = tensap.RandomVector(tensap.UniformRandomVariable(0, 1), d)

        if case == 'oscillatory':
            def fun(x):
                return np.cos(w[0]*2*np.pi + np.matmul(x, np.transpose(c)))
        elif case == 'product_peak':
            def fun(x):
                return 1/np.prod(np.tile(x**(-2), (np.shape(x)[0], 1)) +
                                 (x + np.tile(w, (np.shape(x)[0], 1)))**2, 1)
        elif case == 'corner_peak':
            b = 185
            e = 2
            c = c*b/d**e/np.sum(c)

            def fun(x):
                return (1+np.matmul(x, np.transpose(c)))**(-d-1)
        elif case == 'gaussian':
            def fun(x):
                return np.matmul(np.exp(
                    -(x - np.tile(w, (np.shape(x)[0], 1)))**2,
                    np.transpose(c)**2))
        elif case == 'continuous':
            def fun(x):
                return np.matmul(np.exp(
                    -np.abs(x - np.tile(w, (np.shape(x)[0], 1))),
                    np.transpose(c)**2))
        elif case == 'discontinuous':
            raise NotImplementedError('Function not implemented.')

    elif case == 'henon_heiles':
        if len(args) == 0:
            d = 3
        else:
            d = args[0]

        X = tensap.RandomVector(tensap.NormalRandomVariable(), d)

        def fun(x):
            return 0.5*np.sum(x**2, 1) + \
                0.2*np.sum(x[:, :-1]*x[:, 1:]**2-x[:, :-1]**3, 1) + \
                0.2**2/16*np.sum((x[:, :-1]**2+x[:, 1:]**2)**2, 1)

    elif case == 'sobol':
        if len(args) == 0:
            d = 8
            a = [1, 2, 5, 10, 20, 50, 100, 500]
        else:
            d = args[0]
            if len(args) <= 1:
                a = 2**np.arange(0, d)
            else:
                a = args[1]

        X = tensap.RandomVector(tensap.UniformRandomVariable(0, 1), d)

        def fun(x):
            return np.prod((np.abs(4*x-2)+np.tile(a, (np.shape(x)[0], 1))) /
                           np.tile(1+a, (np.shape(x)[0], 1)), 1)

    elif case == 'anisotropic':
        if len(args) == 0:
            d = 16
        else:
            d = args[0]

        X = tensap.RandomVector(tensap.UniformRandomVariable(0, 1), d)

        def fun(x):
            return x[:, 2]*np.sin(x[:, 3] + x[:, 15])

    elif case == 'polynomial':
        if len(args) == 0:
            d = 16
        else:
            d = args[0]
        if len(args) <= 1:
            q = 2
        else:
            q = args[1]

        X = tensap.RandomVector(tensap.UniformRandomVariable(0, 1), d)

        def fun(x):
            return 1/(2**np.shape(x)[1])*np.prod(3*x**q+1, 1)

    else:
        raise NotImplementedError('Bad function name.')

    return fun, X
