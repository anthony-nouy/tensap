'''
Tutorial on learning in tree-based tensor format a tensorized function, with
an example of model selection using a slope heuristic instead of the error on a
test sample.

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
import matplotlib.pyplot as plt
import tensap

# %% Function to approximate: identification of a function f(x) with a function
# g(i1,...,id,y) (see also tutorial_TensorizedFunction)
R = 4  # Resolution
B = 5  # Scaling factor
ORDER = R+1

X = tensap.UniformRandomVariable(0, 1)
Y = tensap.UniformRandomVariable(0, 1)

CHOICE = 1
if CHOICE == 1:
    def FUN(x):
        return np.sin(10*np.pi*(2*x+0.5))/(4*x+1) + (2*x-0.5)**4
elif CHOICE == 2:
    def FUN(x):
        return (np.sin(4*np.pi*x) + 0.2*np.cos(16*np.pi*x))**(x < 0.5) + \
            (2*x-1)*(x >= 0.5)
else:
    raise ValueError('Bad function choice.')

T = tensap.Tensorizer(B, R, 1, X, Y)
TENSORIZED_FUN = T.tensorize(FUN)
TENSORIZED_FUN.fun.evaluation_at_multiple_points = True

# %% Approximation basis
DEGREE = 5
H = tensap.PolynomialFunctionalBasis(Y.orthonormal_polynomials(),
                                     range(DEGREE+1))
BASES = T.tensorized_function_functional_bases(H)

# %% Training and test samples
NUM_TRAIN = 200
X_TRAIN = X.random(NUM_TRAIN)
Y_TRAIN = FUN(X_TRAIN)
X_TRAIN = T.map(X_TRAIN)  # Identification of X_TRAIN with (i_1,...,i_d,y)

NUM_TEST = 1000
X_TEST = X.random(NUM_TEST)
Y_TEST = FUN(X_TEST)
X_TEST = T.map(X_TEST)

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

# Let the solver go to the maximum number of iterations, to perform model
# selection a posteriori using the test error and a slope heuristic
SOLVER.tolerance['on_stagnation'] = 0
SOLVER.tolerance['on_error'] = 0
SOLVER.rank_adaptation_options['early_stopping'] = False

SOLVER.initialization_type = 'canonical'

SOLVER.linear_model_learning.regularization = False
SOLVER.linear_model_learning.basis_adaptation = True
SOLVER.linear_model_learning.error_estimation = True

SOLVER.test_error = True
SOLVER.test_data = [X_TEST, Y_TEST]
# SOLVER.bases_eval_test = BASES.eval(X_TEST)

SOLVER.rank_adaptation = True
SOLVER.rank_adaptation_options['max_iterations'] = 50
SOLVER.rank_adaptation_options['theta'] = 0.8

SOLVER.tree_adaptation = True
SOLVER.tree_adaptation_options['max_iterations'] = 100
# SOLVER.tree_adaptation_options['force_rank_adaptation'] = True

SOLVER.alternating_minimization_parameters['stagnation'] = 1e-10
SOLVER.alternating_minimization_parameters['max_iterations'] = 50

SOLVER.display = True
SOLVER.alternating_minimization_parameters['display'] = False

SOLVER.model_selection = True
SOLVER.model_selection_options['type'] = 'test_error'

T0 = time()
F, OUTPUT = SOLVER.solve()
T1 = time()
print(T1-T0)

# %% Displays
TEST_ERROR = SOLVER.loss_function.test_error(F, [X_TEST, Y_TEST])
print('\nRanks: %s' % F.tensor.ranks)
print('Loo error = %2.5e' % OUTPUT['error'])
print('Test error = %2.5e' % TEST_ERROR)

F.tensor.plot(title='Active nodes')
F.tensor.tree.plot_dims(title='Dimensions associated to the leaf nodes')
F.tensor.tree.plot_with_labels_at_nodes(F.tensor.representation_rank,
                                        title='Representation ranks')

plt.figure()
x_lin = np.linspace(0, 1, 1000)
plt.plot(x_lin, FUN(x_lin), x_lin, F(T.map(x_lin)))
plt.legend(('True function', 'Approximation'))
plt.show()

# %% Model selection using a slope heuristic instead of the test error
SEL = tensap.ModelSelection()
# Complexity of each model: standard complexity based on the method storage
SEL.data['complexity'] = SEL.complexity(OUTPUT['iterates'],
                                        fun='storage',
                                        c_type='standard')
# Empirical risk associated with each model
SEL.data['empirical_risk'] = [SOLVER.loss_function.risk_estimation(
    x, (X_TRAIN, Y_TRAIN)) for x in OUTPUT['iterates']]
# Shape of the penalization function
SEL.pen_shape = lambda x: x / NUM_TRAIN
# Gap factor used in the slope heuristic
SEL.gap_factor = 2

LAMBDA_HAT, M_HAT, LAMBDA_PATH, M_PATH = SEL.slope_heuristic()

print('Model selected using a test sample:         %i, test error = %2.5e.' %
      (OUTPUT['selected_model_number'], OUTPUT['test_error']))
print('Model selected using the slope heuristic:   %i, test error = %2.5e.' %
      (M_HAT, OUTPUT['test_error_iterations'][M_HAT]))
