'''
Tutorial on tensor interpolation.

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
D = 3
P = 10

F = tensap.UserDefinedFunction('1+x0+(2+x0)/(2+x1)+0.04*(x2-x2**3)', D)
F.evaluation_at_multiple_points = True
V = tensap.UniformRandomVariable(-1, 1)
X = tensap.RandomVector(V, D)
BASIS = tensap.PolynomialFunctionalBasis(tensap.LegendrePolynomials(),
                                         range(P+1))
BASES = tensap.FunctionalBases.duplicate(BASIS, D)

# %% Sparse tensor product functional basis
IND = tensap.MultiIndices.with_bounded_norm(D, 1, P)
H = tensap.SparseTensorProductFunctionalBasis(BASES, IND)

# %% Interpolation on a magic grid
G, _, _ = H.magic_points(X.random(1000))
IF = H.interpolate(F, G)

X_TEST = X.random(1000)
ERR = np.linalg.norm(F(X_TEST)-IF(X_TEST)) / np.linalg.norm(F(X_TEST))
print('Test error = %2.5e' % ERR)

# %% Interpolation on a structured magic grid
GRIDS = X.random(1000)
G, _ = BASES.magic_points([GRIDS[:, i] for i in range(GRIDS.shape[1])])
G = tensap.SparseTensorGrid(G, H.indices)
IF = H.interpolate(F, G.array())

X_TEST = X.random(1000)
ERR = np.linalg.norm(F(X_TEST)-IF(X_TEST)) / np.linalg.norm(F(X_TEST))
print('Test error = %2.5e' % ERR)

# %% Interpolation on a structured magic grid (alternative)
GRIDS = X.random(1000)
G, _ = BASES.magic_points([GRIDS[:, i] for i in range(GRIDS.shape[1])])
IF, OUTPUT = H.tensor_product_interpolation(F, G)

X_TEST = X.random(1000)
ERR = np.linalg.norm(F(X_TEST)-IF(X_TEST)) / np.linalg.norm(F(X_TEST))
print('Test error = %2.5e' % ERR)
