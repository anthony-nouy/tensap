'''
Module measure.

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

from abc import ABC, abstractmethod


class Measure(ABC):
    '''
    Class Measure.

    '''

    @abstractmethod
    def __eq__(self, measure_2):
        pass

    @abstractmethod
    def mass(self):
        '''
        Return the mass of the Measure.

        Returns
        -------
        None.

        '''
        pass

    @abstractmethod
    def support(self):
        '''
        Return the support of the Measure.

        Returns
        -------
        None.

        '''
        pass

    @abstractmethod
    def ndim(self):
        '''
        Return the dimension of the Measure.

        Returns
        -------
        None.

        '''
        pass
