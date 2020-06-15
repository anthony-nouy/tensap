'''
Module random_multi_indices.

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


def random_multi_indices(shape):
    '''
    Return a random variable uniformly distributed on I1 x ... x Id.

    If shape contains integers, the intervals are defined as
    Ij = np.arange(shape[j-1]), j = 1, ..., len(shape).

    Parameters
    ----------
    shape : list or numpy.ndarray
        The number of elements of each interval, or the interval themselves.

    Returns
    -------
    tensap.RandomVector
        The random variable uniformly distributed on I1 x ... x Id.

    '''
    order = len(shape)

    if np.all([isinstance(x, (list, np.ndarray)) for x in shape]):
        ind = shape
    else:
        ind = []
        for dim in range(order):
            ind.append(np.arange(shape[dim]))

    for dim in range(order):
        ind[dim] = tensap.DiscreteRandomVariable(ind[dim])

    return tensap.RandomVector(ind)
