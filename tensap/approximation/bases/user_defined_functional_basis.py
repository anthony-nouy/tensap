'''
Module user_defined_functional_basis.

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


class UserDefinedFunctionalBasis(tensap.FunctionalBasis):
    '''
    Class UserDefinedFunctionalBasis.

    The basis is not L2-orthonormal a priori, hence the is_orthonormal
    attribute remains at its default value of False.

    Attributes
    ----------
    handle_fun : numpy.ndarray
        The functions making the basis.
    measure : tensap.Measure
        The measure associated with the basis. Can be a tensap.RandomVector or
        a tensap.RandomVariable to define a random generator and an
        expectation.
    input_dimension : int
        The dimension of the domain of the functions in handle_fun.

    '''

    def __init__(self, h_fun=None, measure=None, input_dim=None):
        '''
        Constructor for the class UserDefinedFunctionalBasis.

        The basis is not L2-orthonormal a priori, hence the is_orthonormal
        attribute remains at its default value of False.

        Parameters
        ----------
        h_fun : list or numpy.ndarray, optional
            The functions making the basis. The default is None.
        measure : tensap.Measure, optional
            The measure associated with the basis. Can be a tensap.RandomVector
            or a tensap.RandomVariable to define a random generator and an
            expectation. The default is None.
        input_dim : input_dimension, optional
            The dimension of the domain of the functions in handle_fun. The
            default is None, indicating to deduce if from measure.ndim().

        Returns
        -------
        None.

        '''
        tensap.FunctionalBasis.__init__(self)
        self.handle_fun = np.ravel(h_fun)

        if measure is not None:
            self.measure = deepcopy(measure)
        if input_dim is None and measure is not None:
            self.input_dimension = measure.ndim()
        elif input_dim is not None:
            self.input_dimension = input_dim
        else:
            self.input_dimension = 1

    def eval(self, x):
        x = np.atleast_1d(x)
        dim = len(self.handle_fun)
        out = np.zeros((np.shape(x)[0], dim))
        for mu in range(dim):
            out[:, mu] = np.ravel(self.handle_fun[mu](x))
        return out

    def domain(self):
        return self.measure.support()

    def ndim(self):
        return self.input_dimension

    def cardinal(self):
        return self.handle_fun.size
