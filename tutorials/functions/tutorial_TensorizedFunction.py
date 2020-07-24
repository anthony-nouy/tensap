'''
Tutorial on Tensorizer and TensorizedFunction.

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

# %% Identification of a function f(x) with a function g(i1, ...,id , y)
# x is identified with (i_1, ..., i_d, y) through a Tensorizer
D = 6  # Resolution
B = 3  # Scaling factor

X = tensap.UniformRandomVariable(0, 1)
Y = tensap.UniformRandomVariable(0, 1)

T = tensap.Tensorizer(B, D, 1, X, Y)


def FUN(x):
    return np.sqrt(x)


TENSORIZED_FUN = T.tensorize(FUN)
TENSORIZED_FUN.fun.evaluation_at_multiple_points = True

DEGREE = 4
P = tensap.PolynomialFunctionalBasis(Y.orthonormal_polynomials(),
                                     range(DEGREE+1))
BASES = T.tensorized_function_functional_bases(P)

H = tensap.FullTensorProductFunctionalBasis(BASES)
GRIDS, _ = BASES.magic_points(BASES.measure.random(100))
G = tensap.FullTensorGrid(GRIDS)

FUN_INTERP, _ = H.tensor_product_interpolation(TENSORIZED_FUN, G)
TR = tensap.Truncator()
TR.tolerance = 1e-9
FUN_INTERP.tensor = TR.ttsvd(FUN_INTERP.tensor)
TENSORIZED_FUN_INTERP = tensap.TensorizedFunction(FUN_INTERP, T)

X_TEST = X.random(1000)
F_X_TEST = TENSORIZED_FUN_INTERP(X_TEST)
Y_TEST = FUN(X_TEST)
ERR_L2 = np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST)
print('Mean squared error = %2.5e' % ERR_L2)

# %% Identification of a bivariate function f(x1, x2) with a function
# g(i1, ..., id, j1,..., jd, y1, y2)
# x1 and x2 are identified with (i_1, ,..., i_d, y1) and (j_1, ...., j_d, y2)
# through a Tensorizer

DIM = 2
D = 6  # Resolution
B = 2  # Scaling factor

T = tensap.Tensorizer(B, D, DIM)
X = T.X
Y = T.Y
T.ordering_type = 1
FUN = tensap.UserDefinedFunction('1/(1+x0+x1)', DIM)
FUN.evaluation_at_multiple_points = True
TENSORIZED_FUN = T.tensorize(FUN)

DEGREE = 1
P = [tensap.PolynomialFunctionalBasis(y.orthonormal_polynomials(),
                                      range(DEGREE+1)) for
     y in Y.random_variables]
BASES = T.tensorized_function_functional_bases(tensap.FunctionalBases(P))

H = tensap.FullTensorProductFunctionalBasis(BASES)
GRIDS, _ = BASES.magic_points(BASES.measure.random(100))
G = tensap.FullTensorGrid(GRIDS)

FUN_INTERP, _ = H.tensor_product_interpolation(TENSORIZED_FUN, G)
TR = tensap.Truncator()
TR.tolerance = 1e-9
FUN_INTERP.tensor = TR.ttsvd(FUN_INTERP.tensor)
TENSORIZED_FUN_INTERP = tensap.TensorizedFunction(FUN_INTERP, T)
print('Representation ranks = %s' % FUN_INTERP.tensor.representation_rank)

X_TEST = X.random(1000)
F_X_TEST = TENSORIZED_FUN_INTERP(X_TEST)
Y_TEST = FUN(X_TEST)
ERR_L2 = np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST)
print('Mean squared error = %2.5e' % ERR_L2)
