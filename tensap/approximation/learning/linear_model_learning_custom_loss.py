'''
Module linear_model_learning_custom_loss.

Copyright (c) 2020, Anthony Nouy, Erwan Grelier
This file is part of tensap (tensor approximation package).

tensap is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

tensap is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with tensap.  If not, see <https://www.gnu.org/licenses/>.

'''

from numpy import arange, finfo
import numpy as np
import tensap
try:
    import tensorflow as tf
    cond = True
except ImportError:
    cond = False


class LinearModelLearningCustomLoss(tensap.LinearModelLearning):
    '''
    Class LinearModelLearningCustomLoss.

    Attributes
    ----------
    optimizer : tensorflow.keras.optimizers.Optimizer
        The optimizer used to solve the learning problem. The default is Adam.
    initial_guess : numpy.ndarray or tensorflow.Tensor
        The initial guess used as a starting point of the optimization
        algorithm. The default is a tensor with components drawn according
        to a standard normal random variable.
    options : dict
        Options for the optimizer:
            - max_iter: the maximum number of iterations of an iterative
            minimization algorithm,
            - stagnation: the value of a stopping criterion based on the
            relative stagnation between two iterates,

    '''

    def __init__(self, custom_loss):
        '''
        Constructor for the class LinearModelLearningCustomLoss.

        Parameters
        ----------
        custom_loss : tap.CustomLossFunction
            The loss function.

        Returns
        -------
        None.

        '''
        if not cond:
            raise ImportError('Package tensorflow must be installed to ' +
                              'use LinearModelLearningCustomLoss.')
        super().__init__(custom_loss)
        self.optimizer = tf.keras.optimizers.Adam()
        self.initial_guess = None

        self.options = {'max_iterations': 1e3,
                        'stagnation': finfo(float).eps}

    def solve(self):
        '''
        Solution of the minimization problem.

        Returns
        -------
        numpy.ndarray or tensap.FunctionalBasisArray
            The solution of the minimization problem.
        dict
            Outputs of the algorithm.
        '''
        self.initialize()

        if self.initial_guess is None:
            self.initial_guess = tf.random.normal([self.basis_eval.shape[1]],
                                                  dtype=tf.float64)
        else:
            self.initial_guess = tf.convert_to_tensor(self.initial_guess,
                                                      dtype=tf.float64)

        basis_eval = self.basis_eval
        training_data = self.training_data

        def risk():
            fun_eval = tf.squeeze(tf.tensordot(basis_eval, var, [1, 0]))
            out = self.loss_function.risk_estimation(fun_eval,
                                                     training_data)
            return out

        var = tf.Variable(self.initial_guess, dtype=tf.float64)
        for it in arange(self.options['max_iterations']):
            var0 = var.numpy()
            self.optimizer.minimize(risk, var_list=[var])

            stagnation = (tf.linalg.norm(var0 - var) /
                          tf.linalg.norm(var0)).numpy()
            if stagnation < self.options['stagnation']:
                break

        sol = var.numpy()

        output = {}
        if self.test_error:
            f_eval = np.matmul(self.basis_eval_test, sol)
            test_error = self.loss_function.test_error(
                f_eval, self.test_data)
            if isinstance(test_error, tf.Tensor):
                test_error = test_error.numpy()
            output['test_error'] = test_error

        if self.basis is not None:
            if np.ndim(sol) == 1:
                sol = tensap.FunctionalBasisArray(sol, self.basis)
            else:
                sol = tensap.FunctionalBasisArray(sol, self.basis,
                                                  sol.shape[1])

        return sol, output
