'''
Tutorial on principal component analysis for low rank approximation of tensors.

See the following article:
Anthony Nouy. Higher-order principal component analysis for the approximation
of tensors in tree-based low-rank formats. Numerische Mathematik,
141(3):743--789, Mar 2019.

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

# %% Creating a function providing the entries of a tensor
D = 5
N = 10
SIZE = np.full(D, N)


def fun(i):
    return np.cos(i[:, 0]/SIZE[0]) + 1/(1+(i[:, 1]/SIZE[1])**2 +
                                        (i[:, 2]/SIZE[2])**4) + i[:, 2]/SIZE[2]


X = tensap.random_multi_indices(SIZE)
TOL = 1e-8

# %% HOPCA (PCA for each dimension, provides reduced spaces)
print('--- Higher order PCA ---')
TPCA = tensap.TensorPrincipalComponentAnalysis()
TPCA.pca_sampling_factor = 1
TPCA.pca_adaptive_sampling = False

print('\nPrescribed tolerance')
TPCA.tol = TOL
SUB_BASES, OUTPUTS = TPCA.hopca(fun, SIZE)
print('Number of evaluations = \n%s' % [x['number_of_evaluations'] for x
                                        in OUTPUTS])
print('Ranks {1, ..., d} = \n%s' % [x.shape[1] for x in SUB_BASES])

print('\nPrescribed ranks')
TPCA.tol = np.inf
TPCA.max_rank = np.random.randint(1, 5, D)
SUB_BASES, OUTPUTS = TPCA.hopca(fun, SIZE)
print('Number of evaluations = \n%s' % [x['number_of_evaluations'] for x
                                        in OUTPUTS])
print('Ranks {1, ..., d} = \n%s' % [x.shape[1] for x in SUB_BASES])

# %% Approximation in Tucker Format
print('\n--- Approximation in Tucker format ---')
TPCA = tensap.TensorPrincipalComponentAnalysis()
TPCA.pca_sampling_factor = 1
TPCA.pca_adaptive_sampling = False

print('\nPrescribed ranks')
TPCA.tol = np.inf
TPCA.max_rank = np.random.randint(1, 5, D)
F, OUTPUT = TPCA.tucker_approximation(fun, SIZE)

print('Number of evaluations = %i' % OUTPUT['number_of_evaluations'])
print('Prescribed ranks = %s' % TPCA.max_rank)
print('Ranks = %s' % F.ranks[F.tree.dim2ind-1])
X_TEST = X.random(1e4)
F_X_TEST = F.eval_at_indices(X_TEST)
Y_TEST = fun(X_TEST)
print('Error = %2.5e' % (np.linalg.norm(Y_TEST - F_X_TEST) /
                         np.linalg.norm(Y_TEST)))

print('\nPrescribed tolerance')
TPCA.tol = TOL
TPCA.max_rank = np.inf
F, OUTPUT = TPCA.tucker_approximation(fun, SIZE)

print('Number of evaluations = %i' % OUTPUT['number_of_evaluations'])
print('Ranks = %s' % F.ranks[F.tree.dim2ind-1])
X_TEST = X.random(1e4)
F_X_TEST = F.eval_at_indices(X_TEST)
Y_TEST = fun(X_TEST)
print('Error = %2.5e' % (np.linalg.norm(Y_TEST - F_X_TEST) /
                         np.linalg.norm(Y_TEST)))

# %% Approximation in tree based format
print('\n--- Approximation in tree based format ---')
TPCA = tensap.TensorPrincipalComponentAnalysis()
TPCA.pca_sampling_factor = 1
TPCA.pca_adaptive_sampling = False
TREE = tensap.DimensionTree.balanced(D)

print('\nPrescribed ranks')
TPCA.tol = np.inf
TPCA.max_rank = np.random.randint(1, 9, TREE.nb_nodes)
TPCA.max_rank[TREE.root-1] = 1
F, OUTPUT = TPCA.tree_based_approximation(fun, SIZE, TREE)

print('Number of evaluations = %i' % OUTPUT['number_of_evaluations'])
print('Storage = %i' % F.storage())
print('Prescribed ranks = %s' % TPCA.max_rank)
print('Ranks = %s' % F.ranks)
X_TEST = X.random(1e3)
F_X_TEST = F.eval_at_indices(X_TEST)
Y_TEST = fun(X_TEST)
print('Error = %2.5e' % (np.linalg.norm(Y_TEST - F_X_TEST) /
                         np.linalg.norm(Y_TEST)))

print('\nPrescribed tolerance')
TPCA.tol = TOL
TPCA.max_rank = np.inf
F, OUTPUT = TPCA.tree_based_approximation(fun, SIZE, TREE)

print('Number of evaluations = %i' % OUTPUT['number_of_evaluations'])
print('Storage = %i' % F.storage())
print('Ranks = %s' % F.ranks)
X_TEST = X.random(1e3)
F_X_TEST = F.eval_at_indices(X_TEST)
Y_TEST = fun(X_TEST)
print('Error = %2.5e' % (np.linalg.norm(Y_TEST - F_X_TEST) /
                         np.linalg.norm(Y_TEST)))

print('\nPrescribed tolerance (adaptive sampling)')
TPCA.tol = TOL
TPCA.pca_adaptive_sampling = True
TPCA.pca_sampling_factor = 1.2
TPCA.max_rank = np.inf
F, OUTPUT = TPCA.tree_based_approximation(fun, SIZE, TREE)

print('Number of evaluations = %i' % OUTPUT['number_of_evaluations'])
print('Storage = %i' % F.storage())
print('Ranks = %s' % F.ranks)
X_TEST = X.random(1e3)
F_X_TEST = F.eval_at_indices(X_TEST)
Y_TEST = fun(X_TEST)
print('Error = %2.5e' % (np.linalg.norm(Y_TEST - F_X_TEST) /
                         np.linalg.norm(Y_TEST)))

# %% Approximation in Tensor Train format
print('\n--- Approximation in tensor train format ---')
TPCA = tensap.TensorPrincipalComponentAnalysis()
TPCA.pca_sampling_factor = 1
TPCA.pca_adaptive_sampling = False

print('\nPrescribed tolerance')
TPCA.tol = TOL
TPCA.max_rank = np.inf
F, OUTPUT = TPCA.tt_approximation(fun, SIZE)

print('Number of evaluations = %i' % OUTPUT['number_of_evaluations'])
print('Storage = %i' % F.storage())
TT_RANKS = np.flip(F.ranks[F.is_active_node])
print('TT-ranks = %s' % TT_RANKS[:-1])
X_TEST = X.random(1e3)
F_X_TEST = F.eval_at_indices(X_TEST)
Y_TEST = fun(X_TEST)
print('Error = %2.5e' % (np.linalg.norm(Y_TEST - F_X_TEST) /
                         np.linalg.norm(Y_TEST)))

print('\nPrescribed ranks')
TPCA.tol = np.inf
TPCA.max_rank = np.random.randint(1, 9, D-1)
F, OUTPUT = TPCA.tt_approximation(fun, SIZE)

print('Number of evaluations = %i' % OUTPUT['number_of_evaluations'])
print('Storage = %i' % F.storage())
print('Prescribed ranks = %s' % TPCA.max_rank)
TT_RANKS = np.flip(F.ranks[F.is_active_node])
print('TT-ranks = %s' % TT_RANKS[:-1])
X_TEST = X.random(1e3)
F_X_TEST = F.eval_at_indices(X_TEST)
Y_TEST = fun(X_TEST)
print('Error = %2.5e' % (np.linalg.norm(Y_TEST - F_X_TEST) /
                         np.linalg.norm(Y_TEST)))
