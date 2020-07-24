'''
Tutorial on learning in canonical tensor format.

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

# %% Approximation basis
DEGREE = 8
BASES = [tensap.PolynomialFunctionalBasis(X_TRAIN.orthonormal_polynomials(),
                                       range(DEGREE+1)) for
         X_TRAIN in X.random_variables]
BASES = tensap.FunctionalBases(BASES)

# %% Training and test samples
NUM_TRAIN = 1000
X_TRAIN = X.random(NUM_TRAIN)
Y_TRAIN = fun(X_TRAIN)

NUM_TEST = 10000
X_TEST = X.random(NUM_TEST)
Y_TEST = fun(X_TEST)

# %% Learning in canonical tensor format
SOLVER = tensap.CanonicalTensorLearning(ORDER, tensap.SquareLossFunction())
SOLVER.rank_adaptation = True
SOLVER.initialization_type = 'mean'
SOLVER.tolerance['on_error'] = 1e-6
SOLVER.alternating_minimization_parameters['stagnation'] = 1e-8
SOLVER.alternating_minimization_parameters['max_iterations'] = 100
SOLVER.linear_model_learning.regularization = False
SOLVER.linear_model_learning.basis_adaptation = True
SOLVER.bases = BASES
SOLVER.training_data = [X_TRAIN, Y_TRAIN]
SOLVER.display = True
SOLVER.alternating_minimization_parameters['display'] = False
SOLVER.test_error = True
SOLVER.test_data = [X_TEST, Y_TEST]
SOLVER.alternating_minimization_parameters['one_by_one_factor'] = False
SOLVER.alternating_minimization_parameters['inner_loops'] = 2
SOLVER.alternating_minimization_parameters['random'] = False
SOLVER.rank_adaptation_options['max_iterations'] = 20
SOLVER.model_selection = True
SOLVER.model_selection_options['type'] = 'test_error'

F, OUTPUT = SOLVER.solve()

TEST_ERROR = SOLVER.loss_function.test_error(F, [X_TEST, Y_TEST])
print('\nCanonical rank = %i, test error = %2.5e' % (len(F.tensor.core.data),
                                                   TEST_ERROR))
