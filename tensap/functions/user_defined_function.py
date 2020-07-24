'''
Module user_defined_function.

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


class UserDefinedFunction(tensap.Function):
    '''
    Class UserDefinedFunction.

    Attributes
    ----------
    fun : function or tensap.Function
        The function defining the UserDefinedFunction object.

    '''

    def __init__(self, fun, dim, shape=None):
        '''
        Constructor for the class UserDefinedFunction.

        Parameters
        ----------
        fun : function or tensap.Function
            The function defining the UserDefinedFunction object.
        dim : int
            The number of input variables.
        shape : list or numpy.ndarray, optional
            The shape of the output of the function. The default is 1.

        Returns
        -------
        None.

        '''
        tensap.Function.__init__(self)

        self.evaluation_at_multiple_points = False
        self.dim = dim
        if shape is not None:
            self.output_shape = shape

        if isinstance(fun, str):
            for i in np.arange(dim-1, -1, -1):
                s = ('x' + str(i))
                s_new = ('x[:, ' + str(i) + ']')
                fun = fun.replace(s, s_new)
            self.fun = lambda x: eval(fun)
        else:
            self.fun = fun

    def eval(self, x):
        if np.ndim(x) == 1:
            x = np.expand_dims(x, 1)

        n = x.shape[0]

        if hasattr(self.fun, 'eval'):
            def fun(x):
                return self.fun.eval(x)
        else:
            fun = self.fun

        if self.evaluation_at_multiple_points:
            y = fun(x)
            if isinstance(y, list):
                y = np.hstack([np.reshape(y_loc, [-1, 1]) for y_loc in y])
        else:
            y = np.zeros((n, np.prod(self.output_shape)))
            for i in range(n):
                y[i, :] = np.ravel(fun(np.atleast_2d(x[i, :])))

        return np.reshape(y, np.concatenate(
            ([n], np.atleast_1d(self.output_shape))))
