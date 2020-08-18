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
Tutorial on tensor completion with tree-based tensor format.

'''

from enum import Enum
from time import time
import numpy as np
import tensap

np.random.seed(0)

# %% Generation of the tensor to recover
sz = [5, 8, 4, 3, 6]  # Size of the tensor
ORDER = len(sz)  # Order of the tensor
T = tensap.DimensionTree.linear(ORDER)
ranks = [1, 3, 3, 4, 3, 6, 3, 2, 4]
np.random.seed(0)
FTBT = tensap.TreeBasedTensor.randn(T, ranks, sz)
F = FTBT.full()

# %% Training and test samples
p = 0.3  # Proportion of known entries of the tensor

N = np.prod(sz)
NUM_TRAIN = np.int64(np.round(p*N))
loc_TRAIN = np.random.choice(N,NUM_TRAIN,replace=False)  # Random selection of n entries
indices_TRAIN = tensap.MultiIndices.ind2sub(sz, loc_TRAIN)
Y_TRAIN = F.eval_at_indices(indices_TRAIN.array)

NUM_TEST = N
loc_TEST = range(N)
indices_TEST = tensap.MultiIndices.ind2sub(sz, loc_TEST)
Y_TEST = F.eval_at_indices(indices_TEST.array)
print('Tensor to recover with ',p*100,'% of its entries known (',NUM_TRAIN,' entries):')
print(FTBT)

# %% Features creation: matrices containing non zero entries if the relative entry is known
FEATURES_TRAIN = []
for i in range(ORDER):
    M = np.zeros((NUM_TRAIN,sz[i]))
    np.put_along_axis(M, indices_TRAIN.array[:, i:i+1], 1.0, axis=1)
    FEATURES_TRAIN.append(M)


FEATURES_TEST = []
for i in range(ORDER):
    M = np.zeros((NUM_TEST,sz[i]))
    np.put_along_axis(M, indices_TEST.array[:, i:i+1], 1.0, axis=1)
    FEATURES_TEST.append(M)


# %% Tree-based tensor format
# Tensor format

class TensorFormat(str, Enum):
    RANDON_TREE_AND_ACTIVE_NODES = 'Random tree and active nodes'
    TENSOR_TRAIN = 'Tensor-Train'
    HIERARCHICAL_TENSOR_TRAIN = 'Hierarchial Tensor-Train'
    BINARY_TREE = 'Binary tree'


CHOICE = TensorFormat.HIERARCHICAL_TENSOR_TRAIN
if CHOICE == TensorFormat.RANDON_TREE_AND_ACTIVE_NODES:
    print('Random tree with active nodes')
    ARITY = [2, 4]
    TREE = tensap.DimensionTree.random(ORDER, ARITY)
    IS_ACTIVE_NODE = np.full(TREE.nb_nodes, True)
    SOLVER = tensap.TreeBasedTensorLearning(TREE, IS_ACTIVE_NODE,
                                            tensap.SquareLossFunction())
elif CHOICE == TensorFormat.TENSOR_TRAIN:
    print('Tensor-train format')
    SOLVER = tensap.TreeBasedTensorLearning.tensor_train(
        ORDER, tensap.SquareLossFunction())
elif CHOICE == TensorFormat.HIERARCHICAL_TENSOR_TRAIN:
    print('Tensor Train Tucker')
    SOLVER = tensap.TreeBasedTensorLearning.tensor_train_tucker(
        ORDER, tensap.SquareLossFunction())
elif CHOICE == TensorFormat.BINARY_TREE:
    print('Binary tree')
    TREE = tensap.DimensionTree.balanced(ORDER)
    IS_ACTIVE_NODE = np.full(TREE.nb_nodes, True)
    SOLVER = tensap.TreeBasedTensorLearning(TREE, IS_ACTIVE_NODE,
                                            tensap.SquareLossFunction())
else:
    raise NotImplementedError('Not implemented.')

TREE=SOLVER.tree
IS_ACTIVE_NODE=SOLVER.is_active_node

# %% Random shuffling of the dimensions associated to the leaves
RANDOMIZE = False
if RANDOMIZE:
    SOLVER.tree.dim2ind = np.random.permutation(SOLVER.tree.dim2ind)
    SOLVER.tree = SOLVER.tree.update_dims_from_leaves()

# %% Initial guess: known entries in a rank-1 tree-based tensor
guess = np.zeros(np.prod(sz))
guess[loc_TRAIN]=Y_TRAIN
guess = guess.reshape(sz,order='F')
guess = tensap.FullTensor(guess,order=ORDER,shape=sz)
tr = tensap.Truncator(tolerance=0, max_rank = 1)
guess = tr.hsvd(guess, SOLVER.tree,SOLVER.is_active_node)


# %% Learning in tree-based tensor format
SOLVER.bases_eval = FEATURES_TRAIN
SOLVER.training_data = [None, Y_TRAIN]

SOLVER.tolerance['on_stagnation'] = 1e-8
SOLVER.tolerance['on_error'] = 1e-8

SOLVER.initialization_type = 'canonical'

SOLVER.linear_model_learning.regularization = False
SOLVER.linear_model_learning.basis_adaptation = True
SOLVER.linear_model_learning.error_estimation = True

SOLVER.test_error = True
SOLVER.test_data = [None, Y_TEST]
SOLVER.bases_eval_test = FEATURES_TEST

SOLVER.rank_adaptation = True
SOLVER.rank_adaptation_options['max_iterations'] = 20
SOLVER.rank_adaptation_options['theta'] = 0.8
SOLVER.rank_adaptation_options['early_stopping'] = False
SOLVER.rank_adaptation_options['early_stopping_factor'] = 10

SOLVER.tree_adaptation = False
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
Fpred, OUTPUT = SOLVER.solve()
T1 = time()
print(T1-T0)

# ESTIMATED TENSOR
Fpred = Fpred.tensor

# %% Displays
TEST_ERROR = SOLVER.loss_function.test_error(Fpred.eval_at_indices(indices_TEST.array), [None, Y_TEST])
print('Ranks: %s' % Fpred.ranks)
print('Loo error = %2.5e' % OUTPUT['error'])
print('Test error = %2.5e' % TEST_ERROR)

Fpred.plot(title='Active nodes')
Fpred.tree.plot_dims(title='Dimensions associated to the leaf nodes')
Fpred.tree.plot_with_labels_at_nodes(Fpred.representation_rank,
                                        title='Representation ranks')
