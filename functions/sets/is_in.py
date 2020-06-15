'''
Module is_in.

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


def is_in(B, x):
    '''
    Return a boolean array of length x.shape[0] such that ok[k] = True if the
    point x[k, :] is in the box given by B.

    B is of shape (2, dim) and contains the two extreme points B[0, :] and
    B[1, :] of the box.

    Parameters
    ----------
    B : numpy.ndarray
        The box.
    x : numpy.ndarray
        The points.

    Returns
    -------
    ok : numpy.ndarray
        A boolean array indicating if the points x are in the box B.

    '''
    x = np.atleast_2d(x)
    B = np.atleast_2d(B)
    if B.size == 2:
        B = np.reshape(B, [-1, 1])

    ok = np.full(x.shape[0], True)
    for k in range(B.shape[1]):
        ok = np.logical_and(ok, x[:, k] >= B[0, k])
        ok = np.logical_and(ok, x[:, k] <= B[1, k])

    return ok
