'''
Tutorial on magic points and interpolation.

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
import matplotlib.pyplot as plt
import tensap

# %% Magic points associated with a Functional Basis (1D)
P1 = tensap.PolynomialFunctionalBasis(tensap.LegendrePolynomials(), range(21))
DOM = P1.domain()
G1 = np.linspace(DOM[0], DOM[1], 200)
MAGIC_POINTS, IND, OUTPUT = P1.magic_points(G1)

plt.figure()
plt.plot(G1, np.zeros(G1.shape), 'k.', markersize=1)
plt.plot(MAGIC_POINTS, np.zeros(MAGIC_POINTS.shape), 'ro', fillstyle='none')
plt.show()

# %% Magic points associated with a FullTensorProductFunctionalBasis
# Tensorization of 1D uniform grids for the selection of magic points in
# dimension d
D = 2
P1 = tensap.PolynomialFunctionalBasis(tensap.LegendrePolynomials(), range(21))
BASES = tensap.FunctionalBases.duplicate(P1, D)
P = tensap.FullTensorProductFunctionalBasis(BASES)
G1 = np.linspace(DOM[0], DOM[1], 30)
G = tensap.FullTensorGrid(G1, D).array()
MAGIC_POINTS, IND, OUTPUT = P.magic_points(G)

if D == 2:
    plt.figure()
    plt.plot(G[:, 0], G[:, 1], 'k.', markersize=1)
    plt.plot(MAGIC_POINTS[:, 0], MAGIC_POINTS[:, 1], 'ro', fillstyle='none')
    plt.show()

# %% Magic points associated with a SparseTensorProductFunctionalBasis
# Selection of magic points in dimension d in
# - a tensorization of 1D uniform grids
# - or a tensorization of 1D magic points
TENSORIZATION_OF_MAGIC_POINTS = True
D = 2
P1 = tensap.PolynomialFunctionalBasis(tensap.LegendrePolynomials(), range(21))
BASES = tensap.FunctionalBases.duplicate(P1, D)
W = [1, 2]  # Weights for the anisotropic sparsity
IND = tensap.MultiIndices.with_bounded_weighted_norm(2, 1, P1.cardinal()-1, W)
P = tensap.SparseTensorProductFunctionalBasis(BASES, IND)

if not TENSORIZATION_OF_MAGIC_POINTS:
    G1 = np.linspace(DOM[0], DOM[1], 100)
    G = tensap.FullTensorGrid(G1, D).array()
else:
    G1 = np.linspace(DOM[0], DOM[1], 1000)
    M1 = P1.magic_points(G1)[0]
    G = tensap.FullTensorGrid(np.ravel(M1), D).array()

MAGIC_POINTS, IND, OUTPUT = P.magic_points(G)

if D == 2:
    plt.figure()
    plt.plot(G[:, 0], G[:, 1], 'k.', markersize=1)
    plt.plot(MAGIC_POINTS[:, 0], MAGIC_POINTS[:, 1], 'ro', fillstyle='none')
    plt.show()

# %% Interpolation of a function using magic points
D = 2


def F(x):
    return np.cos(x[:, 0]) + x[:, 1]**6 + x[:, 0]**2*x[:, 1]


IF = P.interpolate(F, MAGIC_POINTS)

X = tensap.RandomVector(tensap.UniformRandomVariable(), D)
N_TEST = 1000
X_TEST = X.random(N_TEST)
ERR = np.linalg.norm(IF(X_TEST) - F(X_TEST)) / np.linalg.norm(F(X_TEST))
print('Test error = %2.5e' % ERR)
