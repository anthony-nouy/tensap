'''
Tutorial on MultiIndices.

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

# %% Product set
I = tensap.MultiIndices.product_set([[1, 2, 3], [1, 2]])
print('I.array = \n%s\n' % I.array)

# %% Set of indices with bounded p-norm
m = 6
p = 1/2
d = 2
Ip = tensap.MultiIndices.with_bounded_norm(d, p, m)
print('Ip.array = \n%s\n' % Ip.array)

# %% Set of indices bounded by an multi-index
p = [3, 2]
Ip = tensap.MultiIndices.bounded_by(p)
print('Ip.array = \n%s\n' % Ip.array)

# %% Operations on indices
I1 = tensap.MultiIndices.bounded_by([1, 3])
I2 = tensap.MultiIndices.bounded_by([3, 1])

# Adding two sets
I = I1.add_indices(I2)
print('I1.add_indices(I2).array = \n%s\n' % I.array)

# Substracting a set
I = I1.remove_indices(I2)
print('I1.remove_indices(I2).array = \n%s\n' % I.array)

# Adding (or substracting) an integer
I = I+2
print('(I+2).array = \n%s\n' % I.array)

# %% Maximal elements, margin, reduced margin
I1 = tensap.MultiIndices.bounded_by([1, 3])
I2 = tensap.MultiIndices.bounded_by([3, 1])
I = I1.add_indices(I2)
print('I.array = I1.add_indices(I2).array = \n%s\n' % I.array)
I_marg = I.get_margin()
print('I.get_margin().array = \n%s\n' % I_marg.array)
I_red = I.get_reduced_margin()
print('I.get_reduced_margin().array = \n%s\n' % I_red.array)
I_max = I.get_maximal_indices()
print('I.get_maximal_indices().array = \n%s\n' % I_max.array)

# %% Check whether a set is downward closed
dim = 2
I = tensap.MultiIndices.with_bounded_norm(dim,1,4)
print('I.is_downward_closed() = %s\n' % I.is_downward_closed())

J = I.add_indices(tensap.MultiIndices([2, 4]))
print('J.is_downward_closed() = %s\n' % J.is_downward_closed())

# %% When indices represent subindices of an nd array of shape sz
# Obtaining the position of multi-indices in the nd array
I = tensap.MultiIndices.bounded_by([3, 4, 2], 1)
e = I.sub2ind([5, 5, 5])
# Creating a multi-index set associated with entries of a multi-array
I = tensap.MultiIndices.ind2sub([5, 5, 5], e)
print('I.array = \n%s\n' % I.array)

# %% Reduced margin of a sum of two tensap.MultiIndices
I1 = tensap.MultiIndices.bounded_by([3, 5, 7, 4, 3])
I2 = tensap.MultiIndices.bounded_by([7, 5, 3, 4, 2])
I = I1.add_indices(I2)
I_marg = I.get_reduced_margin()
print('I1.add_indices(I2).get_reduced_margin().array = \n%s\n' % I_marg.array)
