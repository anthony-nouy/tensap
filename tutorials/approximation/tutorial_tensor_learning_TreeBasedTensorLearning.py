'''
Tutorial on learning in tree-based tensor format.

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

from time import time
import numpy as np
import tensap

# %% Function to approximate
CHOICE = 2
if CHOICE == 1:
    ORDER = 5
    X = tensap.RandomVector(tensap.NormalRandomVariable(), ORDER)

    def fun(x):
        return 1 / (10 + x[:, 0] + 0.5*x[:, 1])**2
elif CHOICE == 2:
    fun, X = tensap.multivariate_functions_benchmark('borehole')
    ORDER = X.size
else:
    raise ValueError('Bad function choice.')

# %% Approximation basis
DEGREE = 8
ORTHONORMAL_BASES = True
if ORTHONORMAL_BASES:
    BASES = [tensap.PolynomialFunctionalBasis(
        x.orthonormal_polynomials(), range(DEGREE+1)) for
             x in X.random_variables]
else:
    BASES = [tensap.PolynomialFunctionalBasis(
        tensap.CanonicalPolynomials(), range(DEGREE+1)) for
             x in X.random_variables]
BASES = tensap.FunctionalBases(BASES)

# %% Training and test samples
NUM_TRAIN = 100
X_TRAIN = X.random(NUM_TRAIN)
Y_TRAIN = fun(X_TRAIN)

NUM_TEST = 10000
X_TEST = X.random(NUM_TEST)
Y_TEST = fun(X_TEST)

# %% Tree-based tensor format
# Tensor format
# 1 - Random tree and active nodes
# 2 - Tensor-Train
# 3 - Hierarchial Tensor-Train
# 4 - Binary tree
CHOICE = 3
if CHOICE == 1:
    print('Random tree with active nodes')
    ARITY = [2, 4]
    TREE = tensap.DimensionTree.random(ORDER, ARITY)
    IS_ACTIVE_NODE = np.full(TREE.nb_nodes, True)
    SOLVER = tensap.TreeBasedTensorLearning(TREE, IS_ACTIVE_NODE,
                                            tensap.SquareLossFunction())
elif CHOICE == 2:
    print('Tensor-train format')
    SOLVER = tensap.TreeBasedTensorLearning.tensor_train(
        ORDER, tensap.SquareLossFunction())
elif CHOICE == 3:
    print('Tensor Train Tucker')
    SOLVER = tensap.TreeBasedTensorLearning.tensor_train_tucker(
        ORDER, tensap.SquareLossFunction())
elif CHOICE == 4:
    print('Binary tree')
    TREE = tensap.DimensionTree.balanced(ORDER)
    IS_ACTIVE_NODE = np.full(TREE.nb_nodes, True)
    SOLVER = tensap.TreeBasedTensorLearning(TREE, IS_ACTIVE_NODE,
                                            tensap.SquareLossFunction())
else:
    raise NotImplementedError('Not implemented.')

# %% Random shuffling of the dimensions associated to the leaves
RANDOMIZE = True
if RANDOMIZE:
    SOLVER.tree.dim2ind = np.random.permutation(SOLVER.tree.dim2ind)
    SOLVER.tree = SOLVER.tree.update_dims_from_leaves()

# %% Learning in tree-based tensor format
SOLVER.bases = BASES
SOLVER.bases_eval = BASES.eval(X_TRAIN)
SOLVER.training_data = [None, Y_TRAIN]

SOLVER.tolerance['on_stagnation'] = 1e-6
SOLVER.tolerance['on_error'] = 1e-6

SOLVER.initialization_type = 'canonical'

SOLVER.linear_model_learning.regularization = False
SOLVER.linear_model_learning.basis_adaptation = True
SOLVER.linear_model_learning.error_estimation = True

SOLVER.test_error = True
SOLVER.test_data = [X_TEST, Y_TEST]
# SOLVER.bases_eval_test = BASES.eval(X_TEST)

SOLVER.rank_adaptation = True
SOLVER.rank_adaptation_options['max_iterations'] = 20
SOLVER.rank_adaptation_options['theta'] = 0.8
SOLVER.rank_adaptation_options['early_stopping'] = True
SOLVER.rank_adaptation_options['early_stopping_factor'] = 10

SOLVER.tree_adaptation = True
SOLVER.tree_adaptation_options['max_iterations'] = 1e2
# SOLVER.tree_adaptation_options['force_rank_adaptation'] = True

SOLVER.alternating_minimization_parameters['stagnation'] = 1e-10
SOLVER.alternating_minimization_parameters['max_iterations'] = 50

SOLVER.display = True
SOLVER.alternating_minimization_parameters['display'] = False

SOLVER.model_selection = True
SOLVER.model_selection_options['type'] = 'test_error'

# DMRG (inner rank adaptation)
DMRG = False
if DMRG:
    # Rank of the initialization
    SOLVER.rank = 3
    # Type of rank adaptation: can be 'dmrg' to perform classical DMRG (using
    # a truncation of the factor to adapt the rank) or 'dmrg_low_rank' to learn
    # the factor using a rank-adaptive learning
    # algorithm
    SOLVER.rank_adaptation_options['type'] = 'dmrg_low_rank'
    # Maximum alpha-rank for all the nodes of the tree
    SOLVER.rank_adaptation_options['max_rank'] = 10
    # If True: perform an alternating minimization with fixed tree and ranks
    # after the rank adaptation using DMRG
    SOLVER.rank_adaptation_options['post_alternating_minimization'] = False
    # Model selection type when type is 'dmrg_low_rank': can be 'cv_error' to
    # use a cross-validation estimator of the error or 'test_error' to use
    # the error on a test sample
    SOLVER.rank_adaptation_options['model_selection_type'] = 'cv_error'

T0 = time()
F, OUTPUT = SOLVER.solve()
T1 = time()
print(T1-T0)

# %% Displays
TEST_ERROR = SOLVER.loss_function.test_error(F, [X_TEST, Y_TEST])
print('Ranks: %s' % F.tensor.ranks)
print('Loo error = %2.5e' % OUTPUT['error'])
print('Test error = %2.5e' % TEST_ERROR)

F.tensor.plot(title='Active nodes')
F.tensor.tree.plot_dims(title='Dimensions associated to the leaf nodes')
F.tensor.tree.plot_with_labels_at_nodes(F.tensor.representation_rank,
                                        title='Representation ranks')
