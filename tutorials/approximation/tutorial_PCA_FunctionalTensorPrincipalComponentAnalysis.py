'''
Tutorial on functional PCA for low rank approximation of multivariate
functions.

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

# %% Choice of the function to approximate
CHOICE = 2
if CHOICE == 1:
    print('Henon-Heiles')
    D = 5
    FUN, X = tensap.multivariate_functions_benchmark('henon_heiles', D)

    DEGREE = 4
    BASES = [tensap.PolynomialFunctionalBasis(x.orthonormal_polynomials(),
                                              range(DEGREE+1)) for
             x in X.random_variables]

elif CHOICE == 2:
    print('Anisotropic function')
    D = 6
    X = tensap.RandomVector(tensap.UniformRandomVariable(-1, 1), D)

    def FUN(x):
        return 1/(10 + 2*x[:, 0] + x[:, 2] + 2*x[:, 3] - x[:, 4])**2

    DEGREE = 13
    BASES = [tensap.PolynomialFunctionalBasis(x.orthonormal_polynomials(),
                                              range(DEGREE+1)) for
             x in X.random_variables]

elif CHOICE == 3:
    print('Sinus of a sum')
    D = 10
    X = tensap.RandomVector(tensap.UniformRandomVariable(-1, 1), D)

    def FUN(x):
        return np.sin(np.sum(x[:, :10], 1))

    DEGREE = 17
    BASES = [tensap.PolynomialFunctionalBasis(x.orthonormal_polynomials(),
                                              range(DEGREE+1)) for
             x in X.random_variables]

elif CHOICE == 4:
    print('Composition of functions')
    D = 6
    X = tensap.RandomVector(tensap.UniformRandomVariable(-1, 1), D)

    TREE = tensap.DimensionTree.balanced(D)

    def FUN(x1, x2):
        return 1 / (1 + x1**2 + x2**2)
    FUN = tensap.CompositionalModelFunction(TREE, FUN, X)

    DEGREE = 4
    BASES = [tensap.PolynomialFunctionalBasis(x.orthonormal_polynomials(),
                                              range(DEGREE+1)) for
             x in X.random_variables]

elif CHOICE == 5:
    print('Borehole function')
    FUN, X = tensap.multivariate_functions_benchmark('borehole')
    D = X.size

    DEGREE = 14
    BASES = [tensap.PolynomialFunctionalBasis(x.orthonormal_polynomials(),
                                              range(DEGREE+1)) for
             x in X.random_variables]

elif CHOICE == 6:
    print('Tensorized function')
    R = 11  # Resolution
    B = 2  # Scaling factor
    D = R+1

    X = tensap.UniformRandomVariable(0, 1)
    Y = tensap.UniformRandomVariable(0, 1)

    def IFUN(x):
        return 1/(1+x)

    T = tensap.Tensorizer(B, R, 1, X, Y)
    FUN = T.tensorize(IFUN)
    FUN.fun.evaluation_at_multiple_points = True
    DEGREE = 1
    H = tensap.PolynomialFunctionalBasis(Y.orthonormal_polynomials(),
                                         range(DEGREE+1))
    BASES = T.tensorized_function_functional_bases(H)
    X = tensap.RandomVector(BASES.measure)

else:
    raise ValueError('Bad function choice.')

# %% HOPCA (PCA for each dimension, provides reduced spaces)
# PCA for each dimension to get principal subspaces
print('--- Higher order PCA ---')
FPCA = tensap.FunctionalTensorPrincipalComponentAnalysis()
FPCA.bases = BASES
FPCA.pca_sampling_factor = 1
FPCA.tol = 1e-10
SUB_BASES, OUTPUTS = FPCA.hopca(FUN)
print('Number of evaluations = \n%s' % [x['number_of_evaluations'] for x
                                        in OUTPUTS])
print('Ranks {1, ..., d} = \n%s' % [x.cardinal() for x in SUB_BASES])

# %% Approximation in Tucker Format
# PCA for each dimension to get principal subspaces and interpolation on the
# tensor product of principal subspaces
print('\n--- Approximation in Tucker format ---')
FPCA = tensap.FunctionalTensorPrincipalComponentAnalysis()
FPCA.pca_sampling_factor = 1
FPCA.pca_adaptive_sampling = True
FPCA.bases = BASES

print('\nPrescribed ranks')
FPCA.tol = np.inf
FPCA.max_rank = np.random.randint(1, 5, D)
F, OUTPUT = FPCA.tucker_approximation(FUN)

print('Number of evaluations = %i' % OUTPUT['number_of_evaluations'])
print('Storage = %i' % F.storage())
print('Prescribed ranks = %s' % FPCA.max_rank)
print('Ranks = %s' % F.tensor.ranks[F.tensor.tree.dim2ind-1])
X_TEST = X.random(1e4)
F_X_TEST = F(X_TEST)
Y_TEST = FUN(X_TEST)
print('Error = %2.5e' % (np.linalg.norm(Y_TEST - F_X_TEST) /
                         np.linalg.norm(Y_TEST)))

print('\nPrescribed tolerance')
FPCA.tol = 1e-10
FPCA.max_rank = np.inf
F, OUTPUT = FPCA.tucker_approximation(FUN)

print('Number of evaluations = %i' % OUTPUT['number_of_evaluations'])
print('Ranks = %s' % F.tensor.ranks[F.tensor.tree.dim2ind-1])
X_TEST = X.random(1e4)
F_X_TEST = F(X_TEST)
Y_TEST = FUN(X_TEST)
print('Error = %2.5e' % (np.linalg.norm(Y_TEST - F_X_TEST) /
                         np.linalg.norm(Y_TEST)))

# %% Approximation in tree based format
print('\n--- Approximation in tree based format ---')
FPCA = tensap.FunctionalTensorPrincipalComponentAnalysis()
FPCA.pca_sampling_factor = 20
FPCA.bases = BASES
TREE = tensap.DimensionTree.balanced(D)

print('\nPrescribed ranks')
FPCA.tol = np.inf
FPCA.max_rank = np.random.randint(1, 9, TREE.nb_nodes)
FPCA.max_rank[TREE.root-1] = 1
F, OUTPUT = FPCA.tree_based_approximation(FUN, TREE)

print('Number of evaluations = %i' % OUTPUT['number_of_evaluations'])
print('Storage = %i' % F.storage())
print('Prescribed ranks = %s' % FPCA.max_rank)
print('Ranks = %s' % F.tensor.ranks)
X_TEST = X.random(1e3)
F_X_TEST = F(X_TEST)
Y_TEST = FUN(X_TEST)
print('Error = %2.5e' % (np.linalg.norm(Y_TEST - F_X_TEST) /
                         np.linalg.norm(Y_TEST)))

print('\nPrescribed tolerance')
FPCA.tol = 1e-10
FPCA.max_rank = np.inf
F, OUTPUT = FPCA.tree_based_approximation(FUN, TREE)

print('Number of evaluations = %i' % OUTPUT['number_of_evaluations'])
print('Storage = %i' % F.storage())
print('Ranks = %s' % F.tensor.ranks)
X_TEST = X.random(1e3)
F_X_TEST = F(X_TEST)
Y_TEST = FUN(X_TEST)
print('Error = %2.5e' % (np.linalg.norm(Y_TEST - F_X_TEST) /
                         np.linalg.norm(Y_TEST)))

# %% Approximation in Tensor Train format
print('\n--- Approximation in tensor train format ---')
FPCA = tensap.FunctionalTensorPrincipalComponentAnalysis()
FPCA.projection_type = 'interpolation'
FPCA.pca_sampling_factor = 3
FPCA.pca_adaptive_sampling = True
FPCA.bases = BASES

print('\nPrescribed ranks')
FPCA.tol = np.inf
FPCA.max_rank = np.random.randint(1, 9, D-1)
F, OUTPUT = FPCA.tt_approximation(FUN)

print('Number of evaluations = %i' % OUTPUT['number_of_evaluations'])
print('Storage = %i' % F.storage())
print('Prescribed ranks = %s' % FPCA.max_rank)
TT_RANKS = np.flip(F.tensor.ranks[F.tensor.is_active_node])
print('TT-ranks = %s' % TT_RANKS[:-1])
X_TEST = X.random(1e3)
F_X_TEST = F(X_TEST)
Y_TEST = FUN(X_TEST)
print('Error = %2.5e' % (np.linalg.norm(Y_TEST - F_X_TEST) /
                         np.linalg.norm(Y_TEST)))

print('\nPrescribed tolerance')
FPCA.tol = 1e-4
FPCA.max_rank = np.inf
F, OUTPUT = FPCA.tt_approximation(FUN)

print('Number of evaluations = %i' % OUTPUT['number_of_evaluations'])
print('Storage = %i' % F.storage())
TT_RANKS = np.flip(F.tensor.ranks[F.tensor.is_active_node])
print('TT-ranks = %s' % TT_RANKS[:-1])
X_TEST = X.random(1e3)
F_X_TEST = F(X_TEST)
Y_TEST = FUN(X_TEST)
print('Error = %2.5e' % (np.linalg.norm(Y_TEST - F_X_TEST) /
                         np.linalg.norm(Y_TEST)))
