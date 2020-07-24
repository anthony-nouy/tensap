'''
Module product_measure.

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


class ProductMeasure(tensap.Measure):
    '''
    Class ProductMeasure.

    Attributes
    ----------
    measures : list or tensap.RandomVector
        List of Measure objects.

    '''

    def __new__(cls, measures):
        if np.all([isinstance(x, tensap.ProbabilityMeasure) for
                   x in measures]):
            return tensap.RandomVector(measures)
        else:
            return object.__new__(cls)

    def __init__(self, measures):
        '''
        Constructor for the class ProductMeasure.

        Parameters
        ----------
        measures : list or tensap.RandomVector
            List of Measure objects.

        Raises
        ------
        ValueError
            If the provided measures are not of type Measure.

        Returns
        -------
        None.

        '''
        if isinstance(measures, tensap.RandomVector):
            if not isinstance(measures.copula, tensap.IndependentCopula):
                print('The given Copula is replaced by an IndependentCopula.')
            measures = measures.random_variables
        elif not isinstance(measures, list):
            raise ValueError('measures must be a list of Measure.')
        self.measures = measures

    def __repr__(self):
        return self.__class__.__name__

    def __eq__(self, measure_2):
        return np.all([x == y for x, y in zip(self.measures,
                                              measure_2.measures)])

    def random_vector(self):
        '''
        Return, if self is a ProbabilityMeasure, the associated RandomVector.

        Returns
        -------
        tensap.RandomVector
            The RandomVector associated with self.

        '''
        assert np.all([isinstance(x, tensap.ProbabilityMeasure) for x in
                       self.measures]), \
            'The measures should be ProbabilityMeasure.'

        return tensap.RandomVector(self.measures)

    def mass(self):
        return np.prod([x.mass() for x in self.measures])

    def ndim(self):
        return np.sum([x.ndim() for x in self.measures])

    def support(self):
        return [x.support() for x in self.measures]

    def truncated_support(self):
        '''
        Return the truncated support of the measures of the ProductMeasure.

        Returns
        -------
        list
            The truncated support of the measures of the ProductMeasure.

        '''
        return [x.truncated_support() for x in self.measures]

    def marginal(self, ind):
        return ProductMeasure([self.measures[x] for x in ind])

    def pdf(self, x):
        # TODO pdf
        raise NotImplementedError('Method not implemented.')

    def random(self, x):
        # TODO random
        raise NotImplementedError('Method not implemented.')

    def random_sequential(self, x):
        # TODO random_sequential
        raise NotImplementedError('Method not implemented.')

    @staticmethod
    def duplicate(measure, dim):
        '''
        Create a ProductMeasure by duplicating dim times the provided measure.

        Parameters
        ----------
        measure : tensap.Measure
            The measure to be duplicated.
        dim : int
            The number of times the provided measure is duplicated.

        Returns
        -------
        tensap.ProductMeasure
            The created ProductMeasure.

        '''
        return ProductMeasure([measure]*dim)
