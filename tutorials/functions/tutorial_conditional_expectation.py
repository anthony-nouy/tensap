'''
Tutorial on conditional expectations

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

# %% Approximation of a function F(X)
DIM = 6
FUN = tensap.UserDefinedFunction('x0 + x0*x1 + x0*x2**2 + x3**3 + x4 + x5',
                                 DIM)
FUN.evaluationAtMultiplePoints = True

DEGREE = 3
X = tensap.RandomVector(tensap.NormalRandomVariable(0, 1), DIM)
P = tensap.PolynomialFunctionalBasis(tensap.HermitePolynomials(),
                                     range(DEGREE+1))
BASES = tensap.FunctionalBases.duplicate(P, DIM)
GRIDS = X.random(1000)

G = tensap.FullTensorGrid([np.reshape(GRIDS[:, i], [-1, 1]) for
                           i in range(DIM)])
H = tensap.FullTensorProductFunctionalBasis(BASES)

F, OUTPUT = H.tensor_product_interpolation(FUN, G)

X_TEST = X.random(1000)
F_X_TEST = F(X_TEST)
Y_TEST = FUN(X_TEST)
ERR = np.linalg.norm(Y_TEST-F_X_TEST) / np.linalg.norm(Y_TEST)
print('Mean squared error = %2.5e\n' % ERR)

# %% Conditional expectations
# E(F | X0) = 2*X0
DIMS = 0
F_CE = F.conditional_expectation(DIMS)
FUN_CE = tensap.UserDefinedFunction('2*x0', np.size(DIMS))
X_TEST = np.random.randn(10, np.size(DIMS))
ERR = np.linalg.norm(FUN_CE(X_TEST)-F_CE(X_TEST)) / \
    np.linalg.norm(FUN_CE(X_TEST))
print('Dim %s:          error = %2.5e\n' % (DIMS, ERR))

# E(F | X0,X1) = 2*X0 + X0*X1
DIMS = [0, 1]
F_CE = F.conditional_expectation(DIMS)
FUN_CE = tensap.UserDefinedFunction('2*x0+x0*x1', np.size(DIMS))
X_TEST = np.random.randn(10, np.size(DIMS))
ERR = np.linalg.norm(FUN_CE(X_TEST)-F_CE(X_TEST)) / \
    np.linalg.norm(FUN_CE(X_TEST))
print('Dims %s:    error = %2.5e\n' % (DIMS, ERR))

# E(F | X0,X2) = X0 + X0*X2^2
DIMS = [0, 2]
F_CE = F.conditional_expectation(DIMS)
FUN_CE = tensap.UserDefinedFunction('x0+x0*x1**2', np.size(DIMS))
X_TEST = np.random.randn(10, np.size(DIMS))
ERR = np.linalg.norm(FUN_CE(X_TEST)-F_CE(X_TEST)) / \
    np.linalg.norm(FUN_CE(X_TEST))
print('Dims %s:    error = %2.5e\n' % (DIMS, ERR))

# E(F | X0,X3) = 2*X0 + X3^3
DIMS = [0, 3]
F_CE = F.conditional_expectation(DIMS)
FUN_CE = tensap.UserDefinedFunction('2*x0+x1**3', np.size(DIMS))
X_TEST = np.random.randn(10, np.size(DIMS))
ERR = np.linalg.norm(FUN_CE(X_TEST)-F_CE(X_TEST)) / \
    np.linalg.norm(FUN_CE(X_TEST))
print('Dims %s:    error = %2.5e\n' % (DIMS, ERR))

# E(F | X0,X2,X3) = X0 + X0*X1^2 + X3^3
DIMS = [0, 2, 3]
F_CE = F.conditional_expectation(DIMS)
FUN_CE = tensap.UserDefinedFunction('x0+x0*x1**2+x2**3', np.size(DIMS))
X_TEST = np.random.randn(10, np.size(DIMS))
ERR = np.linalg.norm(FUN_CE(X_TEST)-F_CE(X_TEST)) / \
    np.linalg.norm(FUN_CE(X_TEST))
print('Dims %s: error = %2.5e\n' % (DIMS, ERR))
