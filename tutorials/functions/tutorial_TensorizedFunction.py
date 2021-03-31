# Copyright (c) 2020, Anthony Nouy, Erwan Grelier
# This file is part of tensap (tensor approximation package).

# tensap is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# tensap is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with tensap.  If not, see <https://www.gnu.org/licenses/>.

'''
Tutorial on Tensorizer and TensorizedFunction.

'''

import numpy as np
import tensap

# %% Identification of a function f(x) on (0,1) with a function g(i1, ...,id , y)
# x is identified with (i_1, ..., i_d, y) through a Tensorizer
# First interpolate and then truncate

L = 10  # Resolution
B = 3  # Scaling factor

T = tensap.Tensorizer(B, L, 1)

def FUN(x):
    return np.sqrt(x)
FUN = tensap.UserDefinedFunction(FUN, 1)
FUN.evaluation_at_multiple_points = True

TENSORIZED_FUN = T.tensorize(FUN)
TENSORIZED_FUN.fun.evaluation_at_multiple_points = True

# Interpolation of the function in the tensor product feature space 
DEGREE = 3
BASES = T.tensorized_function_functional_bases(DEGREE)
H = tensap.FullTensorProductFunctionalBasis(BASES)
FUN_INTERP, _ = H.tensor_product_interpolation(TENSORIZED_FUN)
TENSORIZED_FUN_INTERP = tensap.TensorizedFunction(FUN_INTERP, T)
X_TEST = T.X.random(100)
F_X_TEST = TENSORIZED_FUN_INTERP(X_TEST)
Y_TEST = FUN(X_TEST)
ERR_L2 = np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST)
print('Mean squared error for the interpolation = %2.5e' % ERR_L2)

# Truncation in tensor train format
TR = tensap.Truncator()
tens = TENSORIZED_FUN_INTERP.fun.tensor
for k in range(1,9):
    TR.tolerance = 10**(-k)
    print('Tolerance =%s' % TR.tolerance)
    TENSORIZED_FUN_TT = TENSORIZED_FUN_INTERP
    TENSORIZED_FUN_TT.fun.tensor = TR.ttsvd(tens)
    print('Representation ranks = %s' % TENSORIZED_FUN_TT.fun.tensor.representation_rank)
    print('Complexity = %s' % TENSORIZED_FUN_TT.fun.tensor.storage())
    X_TEST = T.X.random(1000)
    F_X_TEST = TENSORIZED_FUN_TT(X_TEST)
    Y_TEST = FUN(X_TEST)
    ERR_L2 = np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST)
    print('Mean squared error = %2.5e' % ERR_L2)

# %% Identification of a bivariate function f(x1, x2) on (0,1)^2 with a function
# g(i1,j1, ..., id,jd, y1, y2)
# x1 and x2 are identified with (i_1, ,..., i_d, y1) and (j_1, ...., j_d, y2)
# through a Tensorizer
# First interpolate and then truncate

DIM = 2
L = 8  # Resolution
B = 2  # Scaling factor

T = tensap.Tensorizer(B, L, DIM)
T.ordering_type = 2
FUN = tensap.UserDefinedFunction('1/(1+x0+x1)', DIM)
FUN.evaluation_at_multiple_points = True
TENSORIZED_FUN = T.tensorize(FUN)

# Interpolation of the function in the tensor product feature space
DEGREE = 2
BASES = T.tensorized_function_functional_bases(DEGREE)

H = tensap.FullTensorProductFunctionalBasis(BASES)
FUN_INTERP, _ = H.tensor_product_interpolation(TENSORIZED_FUN)
TENSORIZED_FUN_INTERP = tensap.TensorizedFunction(FUN_INTERP, T)
X_TEST = T.X.random(100)
F_X_TEST = TENSORIZED_FUN_INTERP(X_TEST)
Y_TEST = FUN(X_TEST)
ERR_L2 = np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST)
print('Mean squared error for the interpolation = %2.5e' % ERR_L2)

# Truncation in tensor train format
TR = tensap.Truncator()
tens = TENSORIZED_FUN_INTERP.fun.tensor
for k in range(1,9):
    TR.tolerance = 10**(-k)
    print('Tolerance =%s' % TR.tolerance)
    TENSORIZED_FUN_TT = TENSORIZED_FUN_INTERP
    TENSORIZED_FUN_TT.fun.tensor = TR.ttsvd(tens)
    print('Representation ranks = %s' % TENSORIZED_FUN_TT.fun.tensor.representation_rank)
    print('Complexity = %s' % TENSORIZED_FUN_TT.fun.tensor.storage())
    X_TEST = T.X.random(1000)
    F_X_TEST = TENSORIZED_FUN_TT(X_TEST)
    Y_TEST = FUN(X_TEST)
    ERR_L2 = np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST)
    print('Mean squared error = %2.5e' % ERR_L2)
 
    