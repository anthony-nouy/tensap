'''
Module probability_measure.

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

import tensap


class ProbabilityMeasure(tensap.Measure):
    '''
    Class ProbabilityMeasure.

    '''

    @staticmethod
    def mass():
        return 1

    def random_rejection(self, n, Y, c, m):
        # TODO random_rejection
        raise NotImplementedError('Method not implemented.')
