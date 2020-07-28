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

'''
Module delta_functional_basis.

'''

from copy import deepcopy
import numpy as np
import tensap


class DeltaFunctionalBasis(tensap.FunctionalBasis):
    '''
    Class DeltaFunctionalBasis.

    Basis of univariate functions defined on a finite set of values in R^d.
    
    The is_orthonormal attribute remains at its default value of False.

    Attributes
    ----------
    values : numpy.ndarray of shape (n,d)
        contains the set of n values taken by the argument of the functions.
    measure : tensap.Measure
        The measure associated with the basis, by default a DiscreteMeasure. Can be a tensap.RandomVector or
        a tensap.RandomVariable to define a random generator and an
        expectation.

    '''

    def __init__(self, values=None, measure=None):
        '''
        Constructor for the class DeltaFunctionalBasis.

        The basis is not L2-orthonormal a priori, hence the is_orthonormal
        attribute remains at its default value of False.

        Parameters
        ----------
        values : numpy.ndarray of shape (n,d)
            contains the set of n values taken by the argument of the functions.
            The default is None.
        measure : tensap.Measure, optional
            The measure associated with the basis, by default a DiscreteMeasure. Can be a tensap.RandomVector or
            a tensap.RandomVariable to define a random generator and an
                expectation. The default is None.-

        Returns
        -------
        None.

        '''
        tensap.FunctionalBasis.__init__(self)
                   
        if values is not None:
            values = np.array(values)
            if values.ndim == 1:
                values = np.reshape(values, [-1, 1])
            self.values = values
        if measure is not None:
            self.measure = deepcopy(measure)
        elif values is not None:
            self.measure = tensap.DiscreteMeasure(values);
            
        if measure is not None:
            self.input_dimension = measure.ndim()
        else:
            self.input_dimension = 1

    def eval(self, x):
        x = np.atleast_1d(x)
        print(x)
        dim = self.values.shape[0]
        N = self.values.shape[0]
        out = np.zeros((np.shape(x)[0], dim))
        for mu in range(dim):
            for i in range(x.shape[0]):
                out[i, mu] = np.all(x[i,] == self.values[mu,])
        return out

    def domain(self):
        return self.measure.support()

    def ndim(self):
        return self.input_dimension

    def cardinal(self):
        return self.values.shape[1]
