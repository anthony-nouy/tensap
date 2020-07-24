'''
Tutorial on tensor truncation.

Copyright (c) 2020, Anthony Nouy, Erwan Grelier
This file is part of tensap (tensor approximation package).

tensap is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

tensap is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with tensap.  If not, see <https://www.gnu.org/licenses/>.

'''

import numpy as np
import tensap

# %% Definitions
ORDER = 6
NK = 8


def fun(inputs):
    '''
    Example function used for the tutorial.

    Parameters
    ----------
    inputs : numpy.ndarray
        The inputs points at which the function is to be evaluated.

    Returns
    -------
    numpy.ndarray
        The evaluations of the function at the input points.

    '''
    return 1/(3+inputs[:, 0]+inputs[:, 2]) + inputs[:, 1] + \
        np.cos(inputs[:, 3] + inputs[:, 4])


GRID = tensap.FullTensorGrid(np.linspace(0, 1, NK), ORDER)
ARRAY = GRID.array()

U = fun(ARRAY)
U = tensap.FullTensor(U, ORDER, np.full(ORDER, NK))

# %% Higher Order SVD - truncation in Tucker format
TR = tensap.Truncator()
TR.tolerance = 1e-8
UR = TR.hosvd(U)
print('Error = %2.5e' % ((UR.full()-U).norm()/U.norm()))
print('Storage = %d' % UR.storage())
print('Dimension of spaces = %s\n' % UR.tensors[0].shape)

# %% Tree-based format
ARITY_INTERVAL = [2, 3]
TREE = tensap.DimensionTree.random(ORDER, ARITY_INTERVAL)
TR = tensap.Truncator()
TR.tolerance = 1e-8
UR = TR.hsvd(U, TREE)

print('Error = %2.5e' % ((UR.full()-U).norm()/U.norm()))
print('Storage = %d' % UR.storage())
print('Ranks = %s\n' % UR.ranks)

# %% Truncation in tensor-train format
TR = tensap.Truncator()
TR.tolerance = 1e-8
UR = TR.ttsvd(U)
print(UR)

SIN_VAL = UR.singular_values()

print('Error = %2.5e' % ((UR.full()-U).norm()/U.norm()))
print('Storage = %d' % UR.storage())
print('TT-rank = %s\n' % UR.ranks)

# %% Truncation in tensor-train format after permutation
PERM = np.random.permutation(ORDER)
U_PERM = U.transpose(PERM)

TR = tensap.Truncator()
TR.tolerance = 1e-10
TR.max_rank = 200

UR = TR.ttsvd(U_PERM)
print('With permutation...')
print('\tOrdering = %s' % PERM)
print('\tError = %2.5e' % ((UR.full()-U_PERM).norm()/U_PERM.norm()))
print('\tStorage = %d' % UR.storage())
print('\tRanks = %s\n' % UR.ranks)

# %% Tensor-train tree optimization
print('Tree optimization...')
UR_PERM = UR.optimize_leaves_permutations(1e-10, 10)
print('\tInitial storage = %d' % UR.storage())
print('\tError = %2.5e' % ((UR_PERM.full()-U_PERM).norm() / U.norm()))
print('\tFinal storage = %d' % UR_PERM.storage())
print('\tRanks = %s' % UR_PERM.ranks)

UR.tree.plot_dims(title='Dimensions before tree optimization')
UR_PERM.tree.plot_dims(title='Dimensions before after optimization')
