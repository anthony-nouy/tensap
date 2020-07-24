'''
Module canonical_tensor_learning.

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

from copy import deepcopy
import numpy as np
import tensap


class CanonicalTensorLearning(tensap.TensorLearning):
    '''
    Class CanonicalTensorLearning.

    See also tensap.TensorLearning.

    Attributes
    ----------
    order : int
        The order of the tensor.

    '''

    def __init__(self, order, *args):
        '''
        Constructor for the class CanonicalTensorLearning.

        Parameters
        ----------
        order : int
            The order of the tensor.
        *args : tuple
            Additional parameters.

        Returns
        -------
        None.

        '''
        super().__init__(*args)
        self.order = order
        self.initialization_type = 'mean'
        self.truncate_initialization = True

# %% Standard solver methods
    def initialize(self):
        if self.tree_adaptation:
            print('tree_adaptation not defined for CanonicalTensorLearning.')
            self.tree_adaptation = False

        if 'one_by_one_factor' in self.alternating_minimization_parameters and\
            self.alternating_minimization_parameters['one_by_one_factor'] and \
                self.rank != 1:
            self._exploration_strategy = np.arange(
                1, self.alternating_minimization_parameters['inner_loops'] *
                self.rank*self.order+2)
            self._number_of_parameters = self._exploration_strategy.size
        else:
            self.alternating_minimization_parameters['one_by_one_factor'] = \
                False
            self._exploration_strategy = np.arange(1, self.order+2)
            self._number_of_parameters = self.order + 1

        shape = [x.shape[1] for x in self.bases_eval]
        if self.initialization_type == 'random':
            f = tensap.CanonicalTensor.randn(self.rank, shape)
        elif self.initialization_type == 'ones':
            f = tensap.CanonicalTensor.ones(self.rank, shape)
        elif self.initialization_type == 'initial_guess':
            f = self.initial_guess
        elif self.initialization_type == 'mean' or \
                self.initialization_type == 'mean_randomized':
            if not isinstance(self.training_data, list) or \
                    (isinstance(self.training_data, list) and
                     len(self.training_data) == 1):
                raise NotImplementedError('Initialization type not ' +
                                          'implemented in unsupervised ' +
                                          'learning.')
            if isinstance(self.bases, tensap.FunctionalBases):
                means = self.bases.mean()
            else:
                means = [np.mean(x, 0) for x in self.bases_eval]
            if self.initialization_type == 'mean_randomized':
                means = [x + 0.01*np.random.randn(*x.shape) for x in means]
            means = [np.reshape(x, [-1, 1]) for x in means]

            f = tensap.CanonicalTensor(
                means, np.atleast_1d(np.mean(self.training_data[1])))
        elif self.initialization_type == 'greedy':
            s_ini = deepcopy(self)
            s_ini.rank_adaptation = False
            s_ini.algorithm = 'greedy'
            s_ini.initialization_type = 'mean'
            s_ini.alternating_minimization_parameters['display'] = False
            s_ini.linear_model_learning.error_estimation = False
            s_ini.test_error = False
            s_ini.display = False
            f, output_ini = s_ini.solve()

            if self.display and 'error' in output_ini:
                print('Greedy initialization: rank = %i, error = %2.5e' %
                      (len(f.tensor.core.data), output_ini['error']))
        else:
            raise ValueError('Wrong initialization type.')

        if isinstance(f, tensap.FunctionalTensor):
            f = f.tensor

        if self.rank > len(f.core.data):
            fx = f.tensor_matrix_product(self.bases_eval).eval_diag().data

            s_ini = deepcopy(self)
            s_ini.rank_adaptation = False
            s_ini.algorithm = 'standard'
            s_ini.initialization_type = 'greedy'
            s_ini.rank = self.rank - len(f.core.data)
            s_ini.alternating_minimization_parameters['display'] = False
            s_ini.linear_model_learning.error_estimation = False
            s_ini.test_error = False
            s_ini.display = False
            if isinstance(s_ini.training_data, list) and \
                    len(s_ini.training_data) == 2:
                s_ini.training_data[1] -= fx
            elif isinstance(s_ini.loss_function, tensap.DensityL2LossFunction):
                s_ini.training_data = [s_ini.training_data, fx]

            f_add = s_ini.solve()[0]
            if isinstance(f_add, tensap.FunctionalTensor):
                f += f_add.tensor
            else:
                f += f_add

        if self.truncate_initialization:
            if self.order == 2:
                tr = tensap.Truncator()
                f = tr.truncate(f)
                f = tensap.CanonicalTensor(f.space, f.core.data)
                self.rank = len(f.core.data)
        return self, f

    def pre_processing(self, f):
        return self, f

    def randomize_exploration_strategy(self):
        if 'one_by_one_factor' not \
            in self.alternating_minimization_parameters or not \
                self.alternating_minimization_parameters['one_by_one_factor']:
            strategy = np.concatenate((
                np.random.permutation(self._number_of_parameters-1)+1,
                [self._number_of_parameters]))
        else:
            strat_mu = np.random.permutation(self.order)
            strat_i = np.random.permutation(self.rank)
            strategy = np.reshape(self._exploration_strategy[:-1],
                                  [-1, self.order], order='F')
            strategy[:, strat_mu] = np.array(strategy)
            strategy = np.reshape(np.reshape(strategy, [1, -1], order='F'),
                                  [self.rank, -1], order='F')
            strategy[strat_i, :] = np.array(strategy)
            strategy = np.concatenate((np.reshape(strategy, -1, order='F'),
                                       [self._number_of_parameters]))
        return strategy

    def prepare_alternating_minimization_system(self, f, mu):
        assert isinstance(self.loss_function, tensap.SquareLossFunction), \
            'Method not implemented for this loss function.'
        y = self.training_data[1]
        N = self.bases_eval[0].shape[0]

        if mu != self._number_of_parameters:
            if self.alternating_minimization_parameters['one_by_one_factor']:
                ind = mu
                mu = int(np.ceil(ind /
                                 self.alternating_minimization_parameters[
                                     'inner_loops'] / self.rank))
                ind = int(ind - self.rank * (np.ceil(ind / self.rank) - 1))

                coef = f.tensor.space[mu-1][:, ind-1]

                fH = f.tensor.tensor_matrix_product(self.bases_eval)

                fH_mu = np.ones((N, self.rank))
                no_mu = np.setdiff1d(np.arange(1, f.tensor.order+1), mu)
                for nu in no_mu:
                    fH_mu *= fH.space[nu-1]

                B = fH_mu * fH.space[mu-1]
                no_i = np.setdiff1d(np.arange(1, self.rank+1), ind)
                b = y - np.matmul(B[:, no_i-1], fH.core.data[no_i-1])

                A = self.bases_eval[mu-1] * np.transpose(
                    np.tile(fH_mu[:, ind-1], (f.tensor.shape[mu-1], 1)))
                if self.linear_model_learning[mu-1].basis_adaptation:
                    self.linear_model_learning[mu-1].basis_adaptation_path = \
                        self.bases_adaptation_path[mu-1]
            else:
                coef = f.tensor.space[mu-1]
                grad = f.parameter_gradient_eval(mu).transpose([0, 2, 1])
                A = np.reshape(grad.data, [grad.shape[0], -1], order='F')

                if self.linear_model_learning[mu-1].basis_adaptation:
                    self.linear_model_learning[mu-1].basis_adaptation_path = \
                        np.tile(self.bases_adaptation_path[mu-1],
                                (self.rank, 1))
                elif self.rank > 1:
                    self.linear_model_learning[mu-1].options[
                        'non_zero_blocks'] = np.empty(self.rank, dtype=object)
                    for kk in range(self.rank):
                        shape_mu = f.tensor.shape[mu-1]
                        self.linear_model_learning[mu-1].options[
                            'non_zero_blocks'][kk] = \
                            kk * shape_mu + np.arange(shape_mu)
                b = y
        else:
            coef = f.tensor.core.data
            if self.alternating_minimization_parameters['one_by_one_factor']:
                mu = self.order + 1
            grad = f.parameter_gradient_eval(mu)
            A = np.reshape(grad.data, [grad.shape[0], -1], order='F')
            b = y
            self.linear_model_learning[mu-1].basis_adaptation = False

        self.linear_model_learning[mu-1].initial_guess = np.reshape(coef, -1)

        return self, A, b, f

    def set_parameter(self, f, mu, coef):
        if mu != self._number_of_parameters:
            if not self.alternating_minimization_parameters[
                    'one_by_one_factor']:
                coef = np.reshape(coef, [f.tensor.shape[mu-1], self.rank],
                                  order='F')
                norm_coef = np.sqrt(np.sum(coef**2, 0))
                ind = norm_coef != 0
                if not np.all(ind):
                    print('Degenerate case: one factor is zero.')
                coef[:, ind] = np.matmul(coef[:, ind],
                                         np.diag(1/norm_coef[ind]))
                f.tensor.space[mu-1] = coef
                f.tensor.core.data = np.ravel(norm_coef)

                if len(f.tensor.space) == 2 and mu == 1:
                    f.tensor.space[0] = np.linalg.qr(
                        f.tensor.space[0])[0]
            else:
                ind = mu
                mu = int(np.ceil(ind /
                                 self.alternating_minimization_parameters[
                                     'inner_loops'] / self.rank))
                ind = int(ind - self.rank * (np.ceil(ind / self.rank) - 1))
                norm_coef = np.linalg.norm(coef)
                coef /= norm_coef
                f.tensor.space[mu-1][:, ind-1] = coef
                f.tensor.core.data[ind-1] = norm_coef
        else:
            f.tensor.core.data = coef
        return f

    def stagnation_criterion(self, f, f0):
        norm_f = f.tensor.norm()
        norm_f0 = f0.tensor.norm()
        return 2 * np.abs(norm_f - norm_f0) / (norm_f + norm_f0)

    def final_display(self, f):
        print('Rank = %i' % len(f.tensor.core.data), end='')

# %% Rank adaptation solver methods
    def local_solver(self):
        s_local = deepcopy(self)
        s_local.display = False
        s_local.rank_adaptation = False
        s_local.test_error = False
        s_local.algorithm = 'standard'
        s_local.initialization_type = 'mean'
        s_local.model_selection = False
        return s_local

    def new_rank_selection(self, f):
        return len(f.tensor.core.data) + 1, 1, deepcopy(f)

    def initial_guess_new_rank(self, s_local, f, *args):
        s_local.initialization_type = 'initial_guess'
        s_local.initial_guess = f
        return s_local

    def adaptation_display(self, f, *args):
        print('\tRank = %i' % len(f.tensor.core.data))

# %% Greedy solver
    def _solve_greedy(self):
        '''
        Greedy solver in canonical tensor format.

        Raises
        ------
        ValueError
            If the number of LinearModelLearning objects is not equal to
            _numberOfParameters.

        Returns
        -------
        f : tensap.FunctionalTensor
            The learned approximation.
        output : dict
            The outputs of the solver.

        '''
        assert isinstance(self.loss_function, tensap.SquareLossFunction), \
            'Method not implemented for this loss function.'

        bases_eval = self.bases_eval
        output = {}
        output['sequence'] = np.empty(self.rank, dtype=object)

        # Replication of the LinearModelLearning objects
        if self.linear_model_learning_parameters[
                'identical_for_all_parameters'] and \
                not isinstance(self.linear_model_learning, (list, np.ndarray)):
            self.linear_model_learning = list(map(deepcopy, [
                self.linear_model_learning] * self._number_of_parameters))
        elif isinstance(self.linear_model_learning, (list, np.ndarray)) and \
                len(self.linear_model_learning) != self._number_of_parameters:
            raise ValueError('Must provide self._numberOfParameters ' +
                             'LinearModelLearning objects.')

        # Working set paths
        if isinstance(self.linear_model_learning, (list, np.ndarray)) and \
            np.any([x.basis_adaptation for
                    x in self.linear_model_learning]) and \
                self.bases_adaptation_path is None:
            self.bases_adaptation_path = self.bases.adaptation_path()

        y = self.training_data[1]

        s_local = deepcopy(self)
        s_local.algorithm = 'standard'
        s_local.rank = 1
        s_local.display = False
        s_local.test_error = False
        s_local.model_selection = False

        f = tensap.CanonicalTensor.zeros(0, [x.shape[1] for x in bases_eval])
        f_0 = deepcopy(f)
        stagnation = np.zeros((1, self.rank))

        ls_local = deepcopy(s_local.linear_model_learning
                            [self._number_of_parameters-1])

        is_error = False
        error = np.ones((1, self.rank))
        pure_greedy = False
        if not pure_greedy:
            for linear_solver in self.linear_model_learning:
                setattr(linear_solver, 'error_estimation', False)

        for k in np.arange(1, self.rank+1):
            s_local.training_data[1] = \
                y - f.tensor_matrix_product(bases_eval).eval_diag().data
            f_add, output_greedy = s_local.solve()
            if isinstance(f_add, tensap.FunctionalTensor):
                f += f_add.tensor
            else:
                f += f_add

            if not pure_greedy:
                f_H = f.tensor_matrix_product(bases_eval)
                A = np.ones((bases_eval[0].shape[0], len(f_H.core.data)))
                for space in f_H.space:
                    A *= space
                ls_local.basis_adaptation = False
                ls_local.basis = None
                ls_local.basis_eval = A
                ls_local.training_data = [None, y]
                coef, output_greedy = ls_local.solve()
                f.core.data = np.ravel(coef)

            stagnation[k-1] = 2*(f - f_0).norm() / (f.norm() + f_0.norm())
            current_rank = len(f.core.data)
            output['sequence'][k-1] = f

            if 'error' in output_greedy:
                error[k-1] = output_greedy['error']
                is_error = True
                if self.alternating_minimization_parameters['display']:
                    print('Alternating minimization (greedy): rank = %i, ' +
                          'error = %2.5e, stagnation = %2.5e', current_rank,
                          error[k-1], stagnation[k-1])
            else:
                if self.alternating_minimization_parameters['display']:
                    print('Alternating minimization (greedy): rank = %i, ' +
                          'stagnation = %2.5e', current_rank, stagnation[k-1])

            if error[k-1] < self.tolerance['on_error'] or \
                stagnation[k-1] < self.tolerance['on_stagnation'] or \
                    (k > 2 and error[k-1] > error[k-2] and
                     error[k-2] > error[k-3]):
                break

            f_0 = deepcopy(f)
            if self.test_error:
                f_eval_test = tensap.FunctionalTensor(f, self.bases_eval_test)
                output['test_error'] = self.loss_function.test_error(
                    f_eval_test, self.test_data)
                output['test_error_iterations'].append(output['test_error'])
                if self.display:
                    print('Greedy: iteration %i, test error = %2.5e', k,
                          output['test_error'])

        output['stagnation'] = stagnation[:k-1]
        if is_error:
            output['errors'] = error[:k-1]
            K = np.argmin(output['errors'])
            f = output['sequence'][K]
            output['selected_iterate'] = K
            output['error'] = output['errors'][K]

        if isinstance(self.bases, tensap.FunctionalBases):
            f = tensap.FunctionalTensor(f, self.bases)

        return f, output
