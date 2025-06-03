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

"""
Tutorial on tensor interpolation.

"""

import numpy as np
import tensap

# %% Definitions
D = 2
P = 10

FUN = tensap.UserDefinedFunction("(1+x0)**2+(2+x0)/(2+x1)", D)
FUN.evaluation_at_multiple_points = True
V = tensap.UniformRandomVariable(-1.5, 2)
X = tensap.RandomVector(V, D)
BASIS = tensap.PolynomialFunctionalBasis(V.orthonormal_polynomials(), range(P + 1))
BASES = tensap.FunctionalBases.duplicate(BASIS, D)

# %% Sparse tensor product functional basis
IND = tensap.MultiIndices.with_bounded_norm(D, 1, P)
H = tensap.SparseTensorProductFunctionalBasis(BASES, IND)

# Training and Test sample
X_TRAIN = X.random(50)
Y_TRAIN = FUN(X_TRAIN)
X_TEST = X.random(1000)
Y_TEST = FUN(X_TEST)

# Solver
SOLVER = tensap.LinearModelLearningSquareLoss()
SOLVER.basis = H
SOLVER.training_data = [X_TRAIN, Y_TRAIN]

# Standard least-squares
SOLVER.regularization = False
SOLVER.basis_adaptation = False

F, OUTPUT = SOLVER.solve()
ERR = np.linalg.norm(F(X_TEST) - Y_TEST) / np.linalg.norm(Y_TEST)
print("Standard least-squares")
print("Test error = %2.5e" % ERR)

# Least-squares with l2 regularization
SOLVER.regularization = True
SOLVER.regularization_type = "l2"
SOLVER.regularization_options = {"alpha": 0.1}

F, OUTPUT = SOLVER.solve()
ERR = np.linalg.norm(F(X_TEST) - Y_TEST) / np.linalg.norm(Y_TEST)
print("Least-squares with l2 regularization")
print("Test error = %2.5e" % ERR)

# Least-squares with l1 regularization
SOLVER.regularization = True
SOLVER.regularization_type = "l1"
SOLVER.regularization_options = {"alpha": 0.1}
SOLVER.model_selection = True

F, OUTPUT = SOLVER.solve()
ERR = np.linalg.norm(F(X_TEST) - Y_TEST) / np.linalg.norm(Y_TEST)
print("Least-squares with l1 regularization")
print("Test error = %2.5e" % ERR)

#  Least-squares with l1 regularization and selection of optimal pattern from
# l1 solution path
SOLVER.regularization = True
SOLVER.regularization_type = "l1"
SOLVER.regularization_options = {"alpha": 0.0}
SOLVER.model_selection = True
SOLVER.basis_adaptation = False

F, OUTPUT = SOLVER.solve()
ERR = np.linalg.norm(F(X_TEST) - Y_TEST) / np.linalg.norm(Y_TEST)
print("Least-squares with l1 regularization and selection of optimal pattern")
print("Test error = %2.5e" % ERR)

# Standard least-squares with basis adaptation :
# Computes least-squares approximations for a
# sequence of subspaces (by default, use hierarchical subspaces
# determined by the natural ordering of basis functions
SOLVER.regularization = False
SOLVER.model_selection = True
SOLVER.basis_adaptation = True
SOLVER.model_selection = True

F, OUTPUT = SOLVER.solve()
ERR = np.linalg.norm(F(X_TEST) - Y_TEST) / np.linalg.norm(Y_TEST)
print("Standard least-squares with basis adaptation")
print("Test error = %2.5e" % ERR)

# The sequences of subspaces can be specified by providing the property
# SOLVER.basis_adaptation_path (a boolean matrix whose columns specify
# the different subspaces)
# Here polynomial spaces V_i with total degre <=i, for i=0...P
B = np.zeros((H.cardinal(), P + 1), dtype=bool)
for i in range(P + 1):
    B[:, i] = np.sum(IND.array, axis=1) <= i

SOLVER.basis_adaptation_path = B
F, OUTPUT = SOLVER.solve()
ERR = np.linalg.norm(F(X_TEST) - Y_TEST) / np.linalg.norm(Y_TEST)
print("Standard least-squares with basis adaptation (specific sequence of spaces)")
print("Test error = %2.5e" % ERR)
