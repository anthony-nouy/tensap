'''
Tutorial on tree-based tensor approximation of multivariate probability density
functions using least-squares.

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
from scipy.stats import multivariate_normal
import tensap

# %% Function to approximate: multivariate gaussian distribution
ORDER = 6
# Covariance matrix
S = np.array([[2,   0,   0.5, 1,   0,   0.5],
              [0,   1,   0,   0,   0.5, 0],
              [0.5, 0,   2,   0,   0,   1],
              [1,   0,   0,   3,   0,   0],
              [0,   0.5, 0,   0,   1,   0],
              [0.5, 0,   1,   0,   0,   2]])
# Reference measure
REF = tensap.RandomVector([tensap.UniformRandomVariable(-x, x)
                          for x in 5*np.diag(S)])


# Density, defined with respect to the reference measure
def U(x):
    return multivariate_normal.pdf(x, np.zeros(ORDER), S) * \
        np.prod(2*5*np.diag(S))


# %% Training and test samples
NUM_TRAIN = 1e5  # Training sample size
X_TRAIN = multivariate_normal.rvs(np.zeros(ORDER), S, int(NUM_TRAIN))

NUM_TEST = 1e4  # Test sample size
X_TEST = multivariate_normal.rvs(np.zeros(ORDER), S, int(NUM_TEST))

NUM_REF = 1e4  # Size of the sample used to compute the L2 error
XI = REF.random(NUM_REF)

# %% Approximation basis
DEGREE = 20
# Orthonormal bases in each dimension, with respect to the reference measure
BASES = [tensap.PolynomialFunctionalBasis(x.orthonormal_polynomials(),
                                          range(DEGREE+1)) for
         x in REF.random_variables]
BASES = tensap.FunctionalBases(BASES)

# %% Tree-based tensor learning parameters
SOLVER = tensap.TreeBasedTensorLearning.tensor_train_tucker(
        ORDER, tensap.DensityL2LossFunction())

RANDOMIZE = True
if RANDOMIZE:
    SOLVER.tree.dim2ind = np.random.permutation(SOLVER.tree.dim2ind)
    SOLVER.tree = SOLVER.tree.update_dims_from_leaves()

SOLVER.bases = BASES
SOLVER.bases_eval = BASES.eval(X_TRAIN)
SOLVER.training_data = X_TRAIN

SOLVER.tolerance['on_stagnation'] = 1e-6
# In density estimation, the error is the risk, which is negative
SOLVER.tolerance['on_error'] = -np.inf

SOLVER.initialization_type = 'random'

SOLVER.linear_model_learning.regularization = False
SOLVER.linear_model_learning.basis_adaptation = True
SOLVER.linear_model_learning.error_estimation = True

SOLVER.test_error = True
SOLVER.test_data = X_TEST
# SOLVER.bases_eval_test = BASES.eval(X_TEST)

SOLVER.rank_adaptation = True
SOLVER.rank_adaptation_options['max_iterations'] = 20
SOLVER.rank_adaptation_options['theta'] = 0.8
SOLVER.rank_adaptation_options['early_stopping'] = True
# early_stopping_factor < 1 because we the risk is negative
SOLVER.rank_adaptation_options['early_stopping_factor'] = 0.1

SOLVER.tree_adaptation = True
# For the tree adaptation in density estimation, a tolerance must be provided
SOLVER.tree_adaptation_options['tolerance'] = 1e-6

SOLVER.alternating_minimization_parameters['stagnation'] = 1e-6
SOLVER.alternating_minimization_parameters['max_iterations'] = 50

SOLVER.display = True
SOLVER.alternating_minimization_parameters['display'] = False

SOLVER.model_selection = True
SOLVER.model_selection_options['type'] = 'test_error'

# %% Density estimation
T0 = time()
F, OUTPUT = SOLVER.solve()
T1 = time()
print(T1-T0)

L2_ERR = np.linalg.norm(U(XI) - F(XI)) / np.linalg.norm(U(XI))
print('Risk leave-one-out estimation =       %2.5e' % OUTPUT['error'])
print('Risk estimation using a test sample = %2.5e' % OUTPUT['test_error'])
print('L2 relative error estimation =         %2.5e' % L2_ERR)

F.tensor.plot(title='Active nodes')
F.tensor.tree.plot_dims(title='Dimensions associated to the leaf nodes')
F.tensor.tree.plot_with_labels_at_nodes(F.tensor.representation_rank,
                                        title='Representation ranks')
