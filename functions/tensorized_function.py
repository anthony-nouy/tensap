'''
Module tensorized_function.

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


class TensorizedFunction(tensap.Function):
    '''
    Class TensorizedFunction.

    Attributes
    ----------
    fun : function or tensap.Function
        A function of (d+1)*dim variables.
    tens : tensap.Tensorizer
        A tensap.Tensorizer object.

    '''

    def __init__(self, fun, tens=None):
        '''
        Constructor for the class TensorizedFunction.

        Function g(x1, ..., xdim) identified with a function f(z) of
        (d+1)*dim variables using the Tensorizer tens (base b, resolution d,
        dimension dim).

        For a univariate function g(x) (dim=1),
        g(x) = f(i_1, ..., i_d, y) with y in [0,1] and i_k in {0, ..., b-1}
        where x = (i + y)b^(-d) with i in {0, ..., b^d-1} having the following
        representation in base b:
        i = sum_{k=1}^d i_k b^(k-1) in [0,b^d-1].

        For a bivariate function g(x1, x2) (dim=2)
            - if t.ordering_type == 1 then
            g(x) = f(i_1, ..., i_d, j_1, ...., j_d, y1, y2) with yk in [0,1]
            and x1 = (i + y1)b^(-d), x2 = (j + y2)b^(-d)
            - if t.ordering_type == 2 then
            g(x) = f(i_1, j_1, ..., i_d, j_d, y1, y2).

        Parameters
        ----------
        fun : function or tensap.Function
            A function of (d+1)*dim variables.
        tens : tensap.Tensorizer, optional
            A tensap.Tensorizer object. The default is None, raising an error.

        Returns
        -------
        None.

        '''
        assert tens is not None, 'Must provide a Tensorizer.'

        tensap.Function.__init__(self)

        self.fun = fun
        self.tens = tens
        self.dim = tens.dim

    def eval(self, z):
        z = np.array(z)
        if z.ndim == 1:
            z = np.reshape(z, [-1, 1])
        if z.shape[1] == self.tens.dim:
            z = self.tens.map(z)
        return self.fun(z)

    def domain(self):
        '''
        Return the domain of the function.

        Returns
        -------
        sup : numpy.ndarray
            The domain of the function.

        '''
        return self.tens.X.support()
