'''
Module magic_indices.

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


def magic_indices(F, n=None, option='left_right'):
    '''
    Return the set of n magic indices (i_1, j_1), ..., (i_n, j_n) constructed
    by a greedy algorithm.

    If option is 'left_right' (default value):
        [ik,jk] = arg max_{i,j} | F_ij - I_{k-1}(F)_ij |
        where I_{k-1}(F) is the rank-(k-1) matrix which interpolates F on the
        cross corresponding to rows (i_1, ..., i_{k-1}) and
        columns (j_1,..., j_{k-1}).

    If option is 'left':
        jk is equal to k and
        ik = arg max_{i} max_{1 <= j <= M} | F_ij - I_[k-1](F)_ij |.

    If option is 'right':
        ik is equal to k and
        jk = arg max_{j} max_{1 <= i <= N} | F_ij - I[k-1](F)_ij |

    Parameters
    ----------
    F : list or numpy.ndarray
        Input matrix.
    n : int, optional
        Number of magic indices. The default is None, indicating to choose
        n = np.min(F.shape).
    option : str, optional
        The selected option. The default is 'left_right'.

    Raises
    ------
    ValueError
        If the selected option is incorrect.

    Returns
    -------
    I, J : numpy.ndarray
        The magic indices.

    '''
    F = np.atleast_2d(F)
    if n is None:
        n = np.min(F.shape)

    if option == 'left_right':
        G = np.zeros(F.shape)
        I = []
        J = []
        for _ in range(n):
            i = np.argmax(np.max(np.abs(F - G), 1), 0)
            j = np.argmax(np.abs(F[i, :] - G[i, :]))
            I.append(i)
            J.append(j)
            G = np.matmul(F[:, J], np.linalg.solve(F[np.ix_(I, J)], F[I, :]))
    elif option == 'left':
        G = np.zeros(F.shape)
        I = []
        J = []
        for k in range(n):
            i = np.argmax(np.max(np.abs(F[:, :k+1] - G[:, :k+1]), 1), 0)
            I.append(i)
            J.append(k)
            G = np.matmul(F[:, J], np.linalg.solve(F[np.ix_(I, J)], F[I, :]))
    elif option == 'right':
        J, I = magic_indices(np.transpose(F), n, 'left')
    else:
        raise ValueError('Bad option.')
    return I, J
