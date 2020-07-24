'''
Module tensor_learning.

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

from abc import abstractmethod
from copy import deepcopy
import pprint
import numpy as np
import tensap


class TensorLearning(tensap.Learning):
    '''
    Class TensorLearning.

    See also tensap.Learning.

    Attributes
    ----------
    order : int
        Order of the tensor.
    bases : list or numpy.ndarray
        The functional bases used for by the approximation.
    bases_eval : list or numpy.ndarray
        The evaluations of the bases on the training data.
    bases_eval_test : list or numpy.ndarray
        The evaluations of the bases on the test data.
    algorithm : str
        The choice of algorithm.
    initialization_type : str
        The type of initialization.
    initial_guess : tensap.Tensor
        An initial guess for the solver.
    tree_adaptation : bool
        Boolean indicating if tree adaptation is to be performed.
    tree_adaptation_options : dict
        Options for the tree adaptation.
    rank_adaptation : bool
        Boolean indicating if rank adaptation is to be performed.
    rank_adaptation_options : dict
        Options for the rank adaptation.
    tolerance : dict
        Tolerances for the solver.
    linear_model_learning_parameters : dict
        Parameters for the linear solvers.
    alternating_minimization_parameters : dict
        Parameters for the alternating minimization.
    bases_adaptation_path : list or numpy.ndarray
        Paths for each basis for the basis adaptation.
    store_iterates : bool
        Boolean indicating if the iterates are to be stored.
    rank : int or list or numpy.ndarray
        The rank of the expected approximation.
    output_dimension : int
        The dimension of the outputs of the function to be approximated.
    _number_of_parameters : int
        The number of parameters of the approximation.
    _exploration_strategy : numpy.ndarray
        The exploration strategy.
    _warnings : dict
        Dictionnary containing the state of some warnings to be displayed.

    '''

    def __init__(self, loss):
        '''
        Constructor for the class TensorLearning.

        See also tensap.Learning.

        Parameters
        ----------
        loss : tensap.LossFunction
            The loss function associated with the solver.

        Returns
        -------
        None.

        '''
        super().__init__(loss)
        self.linear_model_learning = tensap.Learning.linear_model(loss)
        self.order = None
        self.bases = None
        self.bases_eval = None
        self.bases_eval_test = None
        self.algorithm = 'standard'
        self.initialization_type = None
        self.initial_guess = None
        self.tree_adaptation = False
        self.tree_adaptation_options = {'tolerance': None,
                                        'max_iterations': 100,
                                        'force_rank_adaptation': True}
        self.rank_adaptation = False
        self.rank_adaptation_options = {'max_iterations': 10,
                                        'early_stopping': False,
                                        'early_stopping_factor': 10}
        self.tolerance = {'on_error': 1e-6, 'on_stagnation': 1e-6}
        self.linear_model_learning_parameters = \
            {'identical_for_all_parameters': True}
        self.alternating_minimization_parameters = {'display': False,
                                                    'max_iterations': 30,
                                                    'stagnation': 1e-6,
                                                    'random': False}
        self.bases_adaptation_path = None
        self.store_iterates = True
        self.rank = 1
        self.output_dimension = 1

        self.model_selection = False
        # Possible choices: 'test_error', 'cv_error'
        self.model_selection_options['type'] = 'test_error'

        self._number_of_parameters = None
        self._exploration_strategy = None
        self._warnings = {'orthonormality_warning_display': True,
                          'empty_bases_warning_display': True}

    def __repr__(self):
        return pprint.pformat(self.__dict__, indent=4, width=1)

    def solve(self, *args):
        '''
        Solver for the learning problem with tensor formats.

        Parameters
        ----------
        *args : misc
            Additional arguments, if needed.

        Raises
        ------
        NotImplementedError
            If the required solver is not implemented.
        ValueError
            If the required data is not provided.

        Returns
        -------
        f : tensap.FunctionalTensor
            The learned approximation.
        output : dict
            The outputs of the solver.

        '''
        # If possible, deduce from training_data the output dimension
        if isinstance(self.training_data, list) and \
                len(self.training_data) == 2 and \
                np.ndim(self.training_data[1]) == 2:
            self.output_dimension = self.training_data[1].shape[1]
        if self.output_dimension > 1 and \
            (not isinstance(self, tensap.TreeBasedTensorLearning) or
             not isinstance(self.loss_function, (tensap.SquareLossFunction,
                                                 tensap.CustomLossFunction))):
            raise NotImplementedError(
                'Solver not implemented for vector-valued functions ' +
                'approximation, use TreeBasedTensorLearning with a ' +
                'SquareLossFunction or a CustomLossFunction instead.')

        if self.output_dimension > 1 and \
            'type' in self.rank_adaptation_options and \
                self.rank_adaptation_options['type'] == 'inner':
            print('Inner rank adaptation not implemented for ' +
                  'output_dimension greater than 1, disabling it.')
            del self.rank_adaptation_options['type']

        if self._warnings['orthonormality_warning_display'] and \
            (self.bases is None or (self.bases is not None and not
                                    np.all([x.is_orthonormal for x in
                                            self.bases.bases]))):
            self._warnings['orthonormality_warning_display'] = False
            print('The implemented learning algorithms are designed ' +
                  'for orthonormal bases. These algorithms work with ' +
                  'non-orthonormal bases, but without some guarantees ' +
                  'on their results.')

        # If no bases are provided, warn that the returned functions are
        # evaluated on the training data
        if self.bases is None and \
                self._warnings['empty_bases_warning_display']:
            self._warnings['empty_bases_warning_display'] = False
            print('The returned functions are evaluated on the training ' +
                  'data. To evaluate them at other points, assign to the ' +
                  'FunctionalTensor a nonempty field bases and set the ' +
                  'attribute evaluatedBases to False.')

        # If the test error cannot be computed, it is disabled
        if self.test_error and \
            not isinstance(self.bases, tensap.FunctionalBases) and \
                self.bases_eval_test is None:
            print('The test error cannot be computed.')
            self.test_error = False

        # Assert if basis adaptation can be performed
        if self.bases_adaptation_path is None and \
                not hasattr(self.bases, 'adaptation_path'):
            if isinstance(self.linear_model_learning, (list, np.ndarray)) and \
                    np.any([x.basis_adaptation for
                            x in self.linear_model_learning]):
                print('Cannot perform basis adaptation, disabling it.')
                for linear_solver in self.linear_model_learning:
                    setattr(linear_solver, 'basis_adaptation', False)

            elif not isinstance(self.linear_model_learning,
                                (list, np.ndarray)) and \
                    self.linear_model_learning.basis_adaptation:
                print('Cannot perform basis adaptation, disabling it.')
                self.linear_model_learning.basis_adaptation = False

        # Bases evaluation
        if hasattr(self.bases, 'eval'):
            if self.training_data is not None and self.bases_eval is None:
                if isinstance(self.training_data, list) and \
                        self.training_data[0] is not None:
                    self.bases_eval = self.bases.eval(self.training_data[0])
                elif not isinstance(self.training_data, list) and \
                        self.training_data is not None:
                    self.bases_eval = self.bases.eval(self.training_data)
                else:
                    raise ValueError('Must provide input training data.')

            if self.test_error and self.test_data is not None and \
                    self.bases_eval_test is None:
                if isinstance(self.test_data, list) and \
                        self.test_data[0] is not None:
                    self.bases_eval_test = self.bases.eval(self.test_data[0])
                elif not isinstance(self.test_data, list) and \
                        self.test_data is not None:
                    self.bases_eval_test = self.bases.eval(self.test_data)
                else:
                    raise ValueError('Must provide input test data.')

        if self.model_selection:
            self.store_iterates = True
            if self.model_selection_options['type'] == 'test_error' and \
                    not self.test_error:
                print('Cannot perform test error based model selection, ' +
                      'disabling it.')
                self.model_selection = False
            elif self.model_selection_options['type'] == 'cv_error' and \
                    not self.error_estimation:
                self.error_estimation = True

        if self.rank_adaptation:
            if 'type' not in self.rank_adaptation_options or \
                    self.rank_adaptation_options['type'] == 'standard':
                fun, output = self._solve_adaptation()
            elif isinstance(self.rank_adaptation_options['type'], str):
                adapt_type = self.rank_adaptation_options['type']
                if adapt_type == 'dmrg' or adapt_type == 'dmrg_low_rank':
                    adapt_type = 'dmrg'

                # Call the method corresponding to the rank adaptation option
                fun, output = eval('self._solve_' + adapt_type.lower() +
                                   '_rank_adaptation()')
            else:
                raise ValueError('The rank_adaptation_options attribute ' +
                                 'must be either inexistant or a string.')
        elif self.algorithm == 'standard':
            fun, output = self._solve_standard()
        else:
            fun, output = eval('self._solve_' + self.algorithm.lower() + '()')

        if self.model_selection:
            if self.model_selection_options['type'] == 'test_error':
                ind = np.argmin(output['test_error_iterations'])
                fun = output['iterates'][ind]
                output['selected_model_number'] = ind
                if 'error_iterations' in output and \
                    np.size(output['error_iterations']) > 0:
                    output['error'] = output['error_iterations'][ind]
                output['test_error'] = output['test_error_iterations'][ind]
                if self.display:
                    print('\nModel selection using the test error: model ' +
                          '#%i selected' % ind)
            elif self.model_selection_options['type'] == 'cv_error':
                ind = np.argmin(output['error_iterations'])
                fun = output['iterates'][ind]
                output['selected_model_number'] = ind
                if 'test_error_iterations' in output and \
                    np.size(output['test_error_iterations']) > 0:
                    output['test_error'] = output['test_error_iterations'][ind]
                output['error'] = output['error_iterations'][ind]
                if self.display:
                    print('\nModel selection using the cross-validation ' +
                          ' error: model #%i selected' % ind)
            else:
                print('Wrong model selection type, returning the last ' +
                      'iterate.')

            if self.display:
                if self.alternating_minimization_parameters['display']:
                    print('')
                self.final_display(fun)
                if 'error' in output:
                    print(', error = %2.5e' % output['error'], end='')
                if 'test_error' in output:
                    print(', test error = %2.5e' % output['test_error'],
                          end='')
                print('')

        return fun, output

    def _solve_standard(self):
        '''
        Solver for the learning problem with tensor formats using the standard
        algorithm (without adaptation).

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
        output = {'flag': 0}

        self, f = self.initialize()
        f = tensap.FunctionalTensor(f, self.bases_eval)

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

        if self.error_estimation:
            for x in self.linear_model_learning:
                setattr(x, 'error_estimation', True)

        # Working set paths
        if isinstance(self.linear_model_learning, (list, np.ndarray)) and \
            np.any([x.basis_adaptation for
                    x in self.linear_model_learning]) and \
                self.bases_adaptation_path is None:
            self.bases_adaptation_path = self.bases.adaptation_path()

        if self.alternating_minimization_parameters['max_iterations'] == 0:
            return f, output

        output['stagnation_criterion'] = []
        output['iterates'] = []
        output['error_iterations'] = []
        output['test_error_iterations'] = []

        # Alternating minimization loop
        for iteration in range(self.alternating_minimization_parameters
                               ['max_iterations']):
            self, f = self.pre_processing(f)
            f0 = deepcopy(f)

            if self.alternating_minimization_parameters['random']:
                # Randomize the exploration strategy
                alpha_list = self.randomize_exploration_strategy()
            else:
                alpha_list = self._exploration_strategy

            for alpha in alpha_list:
                self, A, b, f = \
                    self.prepare_alternating_minimization_system(f, alpha)
                self.linear_model_learning[alpha-1].training_data = [None, b]
                self.linear_model_learning[alpha-1].basis = None
                self.linear_model_learning[alpha-1].basis_eval = A

                coef, output_tmp = self.linear_model_learning[alpha-1].solve()
                if coef is None or np.count_nonzero(coef) == 0 or \
                        not np.all(np.isfinite(coef)):
                    print('Empty, zero or NaN solution, returning to the ' +
                          'previous iteration.')
                    output['flag'] = -2
                    output['error'] = np.inf
                    break

                f = self.set_parameter(f, alpha, coef)

            stagnation = self.stagnation_criterion(f, f0)
            output['stagnation_criterion'].append(stagnation)

            if self.store_iterates:
                if isinstance(self.bases, tensap.FunctionalBases):
                    output['iterates'].append(tensap.FunctionalTensor(
                        f.tensor, self.bases))
                else:
                    output['iterates'].append(f)

            if 'error' in output_tmp:
                output['error'] = output_tmp['error']
                output['error_iterations'].append(output['error'])

            if self.test_error:
                f_eval_test = tensap.FunctionalTensor(f, self.bases_eval_test)
                output['test_error'] = self.loss_function.test_error(
                    f_eval_test, self.test_data)
                output['test_error_iterations'].append(output['test_error'])

            if self.alternating_minimization_parameters['display']:
                print('\tAlt. min. iteration %s: stagnation = %2.5e' %
                      (str(iteration).
                       zfill(len(str(self.alternating_minimization_parameters
                                     ['max_iterations']-1))),
                       stagnation), end='')
                if 'error' in output:
                    print(', error = %2.5e' % output['error'], end='')
                if self.test_error:
                    if not np.isscalar(output['test_error']):
                        output['test_error'] = output['test_error'].numpy()
                    print(', test error = %2.5e' % output['test_error'],
                          end='')
                print('')

            if iteration > 0 and stagnation < \
                    self.alternating_minimization_parameters['stagnation']:
                output['flag'] = 1
                break

            if self.test_error and \
                    output['test_error'] < self.tolerance['on_error']:
                output['flag'] = 1
                break

        if isinstance(self.bases, tensap.FunctionalBases):
            f = tensap.FunctionalTensor(f.tensor, self.bases)
        output['iter'] = iteration

        if self.display and not self.model_selection:
            if self.alternating_minimization_parameters['display']:
                print('')
            self.final_display(f)
            if 'error' in output:
                print(', error = %2.5e' % output['error'], end='')
            if 'test_error' in output:
                print(', test error = %2.5e' % output['test_error'], end='')
            print('')

        return f, output

    def _solve_adaptation(self):
        '''
        Solver for the learning problem with tensor formats using the adaptive
        algorithm.

        Returns
        -------
        f : tensap.FunctionalTensor
            The learned approximation.
        output : dict
            The outputs of the solver.

        '''

        flag = 0
        output = {'enriched_nodes_iterations':
                  np.empty(self.rank_adaptation_options['max_iterations'],
                           dtype=object)}
        tree_adapt = False

        f = None
        errors = np.zeros(self.rank_adaptation_options['max_iterations'])
        test_errors = np.zeros(self.rank_adaptation_options['max_iterations'])
        iterates = np.empty(self.rank_adaptation_options['max_iterations'],
                            dtype=object)

        # new_rank = s_local.rank
        new_rank = self.rank
        s_local = self.local_solver()
        s_local.model_selection = False

        enriched_nodes = np.array([])
        for iteration in range(self.rank_adaptation_options['max_iterations']):
            s_local.bases = self.bases
            s_local.bases_eval = self.bases_eval
            s_local.bases_eval_test = self.bases_eval_test
            s_local.training_data = self.training_data
            s_local.test_data = self.test_data
            s_local.rank = new_rank

            f_old = deepcopy(f)
            f, output_local = s_local.solve()
            s_local = self.local_solver()

            if 'error' in output_local:
                errors[iteration] = output_local['error']
                if np.isinf(errors[iteration]):
                    print('Infinite error, returning to the previous iterate.')
                    f = f_old
                    iteration -= 1
                    flag = -2
                    break

            if self.test_error:
                f_eval_test = tensap.FunctionalTensor(
                    f, self.bases_eval_test)
                test_errors[iteration] = self.loss_function.test_error(
                    f_eval_test, self.test_data)

            if self.store_iterates:
                if isinstance(self.bases, tensap.FunctionalBases):
                    iterates[iteration] = tensap.FunctionalTensor(f.tensor,
                                                                  self.bases)
                else:
                    iterates[iteration] = f

            if self.display:
                if self.alternating_minimization_parameters['display']:
                    print('')
                print('\nRank adaptation, iteration %i:' % (iteration))
                self.adaptation_display(f, enriched_nodes)
                print('\tStorage complexity = %i' % f.tensor.storage())

                if errors[iteration] != 0:
                    print('\tError      = %2.5e' % errors[iteration])
                if test_errors[iteration] != 0:
                    print('\tTest error = %2.5e' % test_errors[iteration])
                if self.alternating_minimization_parameters['display']:
                    print('')

            if iteration == self.rank_adaptation_options['max_iterations']-1:
                break

            if (self.test_error and test_errors[iteration] <
                    self.tolerance['on_error']) or \
                ('error' in output_local and errors[iteration] <
                 self.tolerance['on_error']):
                flag = 1
                break

            fac = self.rank_adaptation_options['early_stopping_factor']
            cond = iteration > 0 and (self.test_error and
                                      (np.isnan(test_errors[iteration]) or
                                       fac * np.min(test_errors[:iteration]) <
                                       test_errors[iteration]) or
                                      ('error' in output_local and
                                       (np.isnan(errors[iteration]) or
                                        fac * np.min(errors[:iteration]) <
                                        errors[iteration])))
            if self.rank_adaptation_options['early_stopping'] and cond:
                print('Early stopping', end='')
                if 'error' in output_local:
                    print(', error = %2.5e' % errors[iteration], end='')
                if self.test_error:
                    print(', test error = %2.5e' % test_errors[iteration],
                          end='')
                print('\n')
                iteration -= 1
                f = f_old
                flag = -1
                break

            adapted_tree = False
            if s_local.tree_adaptation and iteration > 0 and \
                    (not self.tree_adaptation_options['force_rank_adaptation']
                     or not tree_adapt):
                C_old = f.tensor.storage()
                self, f, output = self.adapt_tree(f, errors[iteration],
                                                  None, output, iteration)
                adapted_tree = output['adapted_tree']
                if adapted_tree:
                    if self.display:
                        print('\t\tStorage complexity before permutation ' +
                              '= %i' % C_old)
                        print('\t\tStorage complexity after permutation ' +
                              '= %i' % f.tensor.storage())
                    if self.test_error:
                        f_eval_test = tensap.FunctionalTensor(
                            f, self.bases_eval_test)
                        test_errors[iteration] = self.loss_function.test_error(
                            f_eval_test, self.test_data)
                        if self.display:
                            print('\t\tTest error after permutation ' +
                                  '= %2.5e' % test_errors[iteration])

                    if self.alternating_minimization_parameters['display']:
                        print('')

            if not self.tree_adaptation or not adapted_tree:
                if iteration > 0 and not tree_adapt:
                    stagnation = self.stagnation_criterion(
                        tensap.FunctionalTensor(f.tensor, self.bases_eval),
                        tensap.FunctionalTensor(f_old.tensor, self.bases_eval))
                    if stagnation < self.tolerance['on_stagnation'] or \
                            np.isnan(stagnation):
                        break
                tree_adapt = False
                new_rank, enriched_nodes, tensor_for_initialization = \
                    self.new_rank_selection(f)
                output['enriched_nodes_iterations'][iteration] = enriched_nodes
                s_local = self.initial_guess_new_rank(
                    s_local, tensor_for_initialization, new_rank)
            else:
                tree_adapt = True
                enriched_nodes = []
                new_rank = f.tensor.ranks
                s_local.initialization_type = 'initial_guess'
                s_local.initial_guess = f.tensor

        if isinstance(self.bases, tensap.FunctionalBases):
            f = tensap.FunctionalTensor(f.tensor, self.bases)

        if self.store_iterates:
            output['iterates'] = iterates[:iteration+1]

        output['flag'] = flag
        output['enriched_nodes_iterations'] = \
            output['enriched_nodes_iterations'][:iteration+1]
        if 'error' in output_local:
            output['error_iterations'] = errors[:iteration+1]
            output['error'] = errors[iteration]

        if self.test_error:
            output['test_error_iterations'] = test_errors[:iteration+1]
            output['test_error'] = test_errors[iteration]

        if 'adapted_tree' in output:
            del output['adapted_tree']

        return f, output

    def adapt_tree(self, f, cv_error, test_error, output, *args):
        '''
        Tree adaptation algorithm.

        Parameters
        ----------
        f : tensap.Tensor
            The current tensor approximation.
        cv_error : float
            The current cross-validation error.
        test_error : float
            The current test error.
        output : dict
            The outputs of the solver.
        args : tuple
            Additional parameters.

        Returns
        -------
        tensap.TensorLearning
            The TensorLearning object.
        f : tensap.Tensor
            The current tensor approximation, perhaps with an adapted tree.
        output : dict
            The outputs of the solver.

        '''
        return self, f, output

    @abstractmethod
    def initialize(self):
        '''
        Initialization of the learning algorithm.

        Returns
        -------
        tensap.TensorLearning
            The TensorLearning object.
        f : tensap.FunctionalTensor
            The initialization of the solver.

        '''

    @abstractmethod
    def pre_processing(self, f):
        '''
        Initialization of the alternating minimization algorithm.

        Parameters
        ----------
        f : tensap.Tensor
            The current tensor approximation.

        Returns
        -------
        tensap.TensorLearning
            The TensorLearning object.
        f : tensap.Tensor
            The pre-processed tensor approximation.

        '''

    @abstractmethod
    def randomize_exploration_strategy(self):
        '''
        Randomization of the exploration strategy.

        Returns
        -------
        alpha_list : numpy.ndarray
            The randomized exploration strategy.

        '''

    @abstractmethod
    def prepare_alternating_minimization_system(self, f, mu):
        '''
        Preparation of the alternating minimization algorithm.

        Parameters
        ----------
        f : tensap.FunctionalTensor
            The current approximation.
        mu : int
            The number of the parameter to be optimized.

        Returns
        -------
        tensap.TensorLearning
            The TensorLearning object.
        A : numpy.ndarray
            The matricized partial evaluation of f.
        b : numpy.ndarray
            The target training data.
        f : tensap.FunctionalTensor
            The current approximation.

        '''

    @abstractmethod
    def set_parameter(self, f, mu, coef):
        '''
        Update of the parameter mu of the tensor f.

        Parameters
        ----------
        f : tensap.FunctionalTensor
            The current approximation.
        mu : int
            The number of the optimized parameter.
        coef : numpy.ndarray
            The new value of the parameter mu of f.

        Returns
        -------
        f : tensap.FunctionalTensor
            The updated approximation.

        '''

    @abstractmethod
    def stagnation_criterion(self, f, f0):
        '''
        Computation of the stagnation criterion.

        Parameters
        ----------
        f : tensap.FunctionalTensor
            The current approximation.
        f0 : tensap.FunctionalTensor
            The previous approximation.

        Returns
        -------
        stagnation : float
            The value of the stagnation criterion.

        '''

    @abstractmethod
    def final_display(self, f):
        '''
        Display at the end of the computation.

        Parameters
        ----------
        f : tensap.FunctionalTensor
            The current approximation.

        Returns
        -------
        None.

        '''

    @abstractmethod
    def local_solver(self):
        '''
         Extraction of the solver for the adaptive algorithm

        Returns
        -------
        s_local : tensap.TensorLearning
            The local solver.

        '''

    @abstractmethod
    def new_rank_selection(self, f):
        '''
        Selection of a new rank in the adaptive algorithm.

        Parameters
        ----------
        f : tensap.FunctionalTensor
            The current approximation.

        Returns
        -------
        new_rank : numpy.ndarray
            The new tensor rank.
        enriched_nodes : numpy.ndarray
            The enriched parameters.
        tensor_for_initialization : tensap.Tensor
            A tensor used for the initialization of the next iteration.

        '''

    @abstractmethod
    def initial_guess_new_rank(self, s_local, f, new_rank):
        '''
        Computation of the initial guess with the new selected rank.

        Parameters
        ----------
        s_local : tensap.TensorLearning
            The local solver.
        f : tensap.FunctionalTensor
            The current approximation.
        new_rank : numpy.ndarray
            The new tensor rank.

        Returns
        -------
        s_local : tensap.TensorLearning
            The local solver with the initial guess in s_local.initial_guess.

        '''

    @abstractmethod
    def adaptation_display(self, f, enriched_nodes):
        '''
        Display during the adaptation.

        Parameters
        ----------
        f : tensap.FunctionalTensor
            The current approximation.
        enriched_nodes : numpy.ndarray
            The enriched parameters.

        Returns
        -------
        None.

        '''
        return
