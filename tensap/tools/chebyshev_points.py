'''
Module chebyshev_points.

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


def chebyshev_points(n, s=None):
    '''
    Return the first n Chebyshev points in [s[0], s[1]].

    Parameters
    ----------
    n : int
        The number of Chebyshev points.
    s : list or numpy.ndarray, optional
        The interval on which the points are computed. The default is [-1, 1].

    Returns
    -------
    x : numpy.ndarray
        The first n Chebyshev points in [s[0], s[1]].

    '''
    x = np.cos(np.pi*(2*np.arange(1, n+1)-1)/2/n)
    if s is not None:
        x = 0.5*(s[0]+s[1]) + 0.5*(s[1]-s[0])*x
    return np.expand_dims(x, 1)
