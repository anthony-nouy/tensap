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
Tutorial on Tensorizer, TensorizedFunction and FunctionalTensorPrincipalComponentAnalysis.

'''

import numpy as np
import tensap


# %% Identification of a multivariate function f(x1, ..., xd) on (0,1)^d with a function
# g(i_1^1,...,i_1^d, ..., i_L^1,...,i_L^d, y1, ...,yd)
# x1 and x2 are identified with (i_1^1, ,..., i_L^1, y1) and (i_1^d, ...., i_L^d, yd)
# through a Tensorizer

DIM = 4
L = 13  # Resolution
B = 2  # Scaling factor

T = tensap.Tensorizer(B, L, DIM)
T.ordering_type = 2
FUN = tensap.UserDefinedFunction('1/(1+x0+2*x1+3*x2+4*x3)', DIM)
FUN.evaluation_at_multiple_points = True
TENSORIZED_FUN = T.tensorize(FUN)

# Interpolation of the function in the tensor product feature space
DEGREE = 2
BASES = T.tensorized_function_functional_bases(DEGREE)



# %% Learning with PCA based algorithm
FPCA = tensap.FunctionalTensorPrincipalComponentAnalysis()
FPCA.pca_sampling_factor = 20
FPCA.bases = BASES
FPCA.display = False
TREE = tensap.DimensionTree.balanced(TENSORIZED_FUN.fun.dim)
FPCA.tol = 1e-8
FPCA.max_rank = np.inf
FUN_TB, OUTPUT = FPCA.tree_based_approximation(TENSORIZED_FUN, TREE)
TENSORIZED_FUN_TB = tensap.TensorizedFunction(FUN_TB, T)
X_TEST = T.X.random(1000)
F_X_TEST = TENSORIZED_FUN_TB(X_TEST)
Y_TEST = FUN(X_TEST)
ERR_L2 = np.linalg.norm(Y_TEST - F_X_TEST) / np.linalg.norm(Y_TEST)
print('Mean squared error = %2.5e' % ERR_L2)


