'''
Module linear_model_learning_density_l2.

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

import numpy as np
import tensap


class LinearModelLearningDensityL2(tensap.LinearModelLearning):
    '''
    Class LinearModelLearningDensityL2.

    Attributes
    ----------
    is_basis_orthonormal : bool, optional
        Boolean indicating if the basis used to compute the approximation
        is orthonormal according to some reference measure. The default is
        True.

    '''

    def __init__(self):
        '''
        Constructor for the class LinearModelLearningDensityL2.

        Returns
        -------
        None.

        '''
        super().__init__(tensap.SquareLossFunction())

        self.is_basis_orthonormal = True

    def solve(self):
        '''
        Solution (Ordinary or Regularized) of the minimization problem and
        cross-validation procedure.

        Returns
        -------
        sol : numpy.ndarray  or tensap.FunctionalBasisArray
            The solution of the minimization problem.
        output : dict
            Outputs of the algorithm.

        '''
        self.initialize()

        if self.basis_adaptation:
            sol, output = self._solve_basis_adaptation()
        elif self.regularization:
            sol, output = self._solve_regularized()
        else:
            sol, output = self._solve_standard()

        if self.test_error:
            f_eval = np.matmul(self.basis_eval_test, sol)
            output['test_error'] = self.loss_function.test_error(
                f_eval, self.test_data, sol.norm()**2)

        if self.basis is not None:
            sol = tensap.FunctionalBasisArray(sol, self.basis)

        return sol, output

    def _solve_standard(self):
        '''
        Solution of the minimization problem and cross-validation procedure
        (if requested).

        Returns
        -------
        sol : numpy.ndarray
            The solution of the minimization problem.
        output : dict
            Outputs of the algorithm.

        '''
        assert self.is_basis_orthonormal, \
            'Only implemented for orthonormal bases.'

        A = self.basis_eval
        if np.ndim(A) == 3:
            A = np.squeeze(A, axis=2)

        sol = np.mean(A, axis=0)
        if isinstance(self.training_data, list) and \
            len(self.training_data) == 2 and \
                np.size(self.training_data[1]):
            b = np.reshape(self.training_data[1], sol.shape)

            sol -= b
        else:
            b = []

        output = {}
        if self.error_estimation and self.error_estimation_type == 'leave_out':
            N = A.shape[0]
            if np.size(b) == 0:
                output['error'] = (-N**2)/(1-N)**2*np.linalg.norm(sol)**2 + \
                    (2*N-1)/(N*(N-1)**2)*np.sum(np.ravel(A)**2)
            else:
                output['error'] = (N**2-2*N)/(N-1)**2*np.linalg.norm(sol)**2 +\
                    1/(N-1)**2*np.linalg.norm(b)**2 - \
                    2/(N-1)**2*np.sum(np.matmul(A, b)) + \
                    (2*N-1)/(N*(N-1)**2)*np.sum(np.ravel(A)**2) - \
                    2/(N-1)*np.sum(np.matmul(A, sol)) + \
                    2*np.sum(sol*b)

        return sol, output

    def _solve_regularized(self):
        '''
        Solution of the regularized minimization problem and cross-validation
        procedure.

        Returns
        -------
        sol : numpy.ndarray
            The solution of the minimization problem.
        output : dict
            Outputs of the algorithm.

        '''
        sol_standard, _ = self._solve_standard()

        A = self.basis_eval
        if np.ndim(A) == 3:
            A = np.squeeze(A, axis=2)
        N = A.shape[0]

        if isinstance(self.training_data, list) and \
            len(self.training_data) == 2 and \
                np.size(self.training_data[1]):
            b = np.reshape(self.training_data[1], sol_standard.shape)
        else:
            b = []

        A_square_sum = np.sum(A**2, 0)

        list_sort = np.argsort(-np.abs(sol_standard))

        if 'included_coefficients' in self.regularization_options:
            incl_coef = self.regularization_options['included_coefficients']
            list_sort = list_sort[np.logical_not(np.in1d(list_sort,
                                                         incl_coef))]

            sol_incl_coef = np.array(sol_standard)
            sol_incl_coef[np.setdiff1d(range(len(sol_incl_coef)),
                                       incl_coef)] = 0

            if np.size(b) == 0:
                err_incl_coef = \
                    (-N**2)/(1-N)**2*np.linalg.norm(sol_incl_coef)**2 + \
                    (2*N-1)/(N*(N-1)**2)*np.sum(
                        np.ravel(A_square_sum)[incl_coef])
            else:
                b_incl_coef = b[incl_coef]
                err_incl_coef = \
                    (N**2-2*N)/(N-1)**2*np.linalg.norm(sol_incl_coef)**2 +\
                    1/(N-1)**2*np.linalg.norm(b_incl_coef)**2 - \
                    2/(N-1)**2*np.sum(
                        np.matmul(A[:, incl_coef], b_incl_coef)) + \
                    (2*N-1)/(N*(N-1)**2)*np.sum(
                        np.ravel(A_square_sum)[incl_coef]) -\
                    2/(N-1)*np.sum(np.matmul(A, sol_incl_coef)) + \
                    2*np.sum(sol_incl_coef*b_incl_coef)

        if list_sort.size == 0:
            return sol_standard, {'error': np.nan}

        n = list_sort.size
        sol = np.zeros((sol_standard.shape[0], n))
        err = np.zeros(n)

        for i in range(n):
            ind = list_sort[:i+1]
            if 'included_coefficients' in self.regularization_options:
                ind = np.concatenate((incl_coef, ind))
            sol_red = sol_standard[ind]

            if np.size(b) == 0:
                err[i] = \
                    (-N**2)/(1-N)**2*np.linalg.norm(sol_red)**2 + \
                    (2*N-1)/(N*(N-1)**2)*np.sum(
                        np.ravel(A_square_sum)[ind])
            else:
                err[i] = \
                    (N**2-2*N)/(N-1)**2*np.linalg.norm(sol_red)**2 +\
                    1/(N-1)**2*np.linalg.norm(b[ind])**2 - \
                    2/(N-1)**2*np.sum(np.matmul(A[:, ind], b[ind])) + \
                    (2*N-1)/(N*(N-1)**2)*np.sum(np.ravel(A_square_sum)[ind]) -\
                    2/(N-1)*np.sum(np.matmul(A[:, ind], sol_red)) + \
                    2*np.sum(sol_red*b[ind])

            sol[ind, i] = np.squeeze(sol_red)

        if 'included_coefficients' in self.regularization_options:
            sol = np.hstack((np.reshape(sol_incl_coef, (-1, 1)), sol))
            err = np.hstack((err_incl_coef, err))

        ind = np.argmin(err)
        pattern = sol != 0

        output = {}
        output['error'] = err[ind]
        output['error_path'] = err
        output['ind'] = ind
        output['pattern'] = pattern[:, ind]
        output['pattern_path'] = pattern
        output['optimal_solution'] = sol[:, ind]
        output['solution_path'] = sol

        return sol[:, ind], output

    def _solve_basis_adaptation(self):
        '''
        Solution of the minimization problem with working-set and
        cross-validation procedure.

        Returns
        -------
        sol : numpy.ndarray
            The solution of the minimization problem.
        output : dict
            Outputs of the algorithm.

        '''
        if self.basis_adaptation_path is None:
            solpath = np.triu(np.full([self.basis_eval.shape[1]]*2, True))
        else:
            solpath = self.basis_adaptation_path

        sol, output = self._select_optimal_path(solpath)
        output['flag'] = 2
        return sol, output

    def _select_optimal_path(self, solpath):
        '''
        Selection of a solution using leave-one-out cross-validation error.

        The dictionnary output contains:
            - error: the leave-one-out cross-validation error estimate,
            - error_path: the regularization path of leave-one-out
            cross-validation error estimates,
            - ind: the index of the optimal pattern,
            - pattern: the optimal sparsity pattern,
            - pattern_path: the regularization path of sparsity patterns,
            - solution_path: the regularization path of the solutions,
            - optimal_solution: the optimal solution.

        Parameters
        ----------
        solpath : numpy.ndarray
            Array of shape (P, m) whose columns give m potential solutions with
            different sparsity patterns.

        Returns
        -------
        numpy.ndarray
            The optimal solution of the minimization problem.
        dict
            Outputs of the algorithm.

        '''
        sol_standard, output = self._solve_standard()

        A = self.basis_eval
        if np.ndim(A) == 3:
            A = np.squeeze(A, axis=2)

        if isinstance(self.training_data, list) and \
            len(self.training_data) == 2 and \
                np.size(self.training_data[1]):
            b = np.reshape(self.training_data[1], sol_standard.shape)
        else:
            b = []

        A_square_sum = np.sum(A**2, 0)
        if np.size(b) != 0:
            A_sum = np.sum(A, 0)

        pattern = solpath != 0
        rep = np.any(pattern, 0)
        pattern = pattern[:, rep]

        _, ind = np.unique(pattern, return_index=True, axis=1)
        ind = np.sort(ind)
        pattern = pattern[:, ind]

        ind = np.sum(pattern != 0, 0) <= A.shape[0]
        pattern = pattern[:, ind]

        if np.linalg.norm(pattern) == 0:
            return sol_standard, output

        sol = np.zeros(pattern.shape)
        err = np.zeros(pattern.shape[1])

        N = A.shape[0]

        for i in range(pattern.shape[1]):
            ind = pattern[:, i]
            sol_red = sol_standard[ind]

            if np.size(b) == 0:
                err[i] = (-N**2)/(1-N)**2*np.linalg.norm(sol_red)**2 + \
                    (2*N-1)/(N*(N-1)**2)*np.sum(np.ravel(A_square_sum)[ind])
            else:
                b_red = b[ind]
                err[i] = (N**2-2*N)/(N-1)**2*np.linalg.norm(sol_red)**2 +\
                    1/(N-1)**2*np.linalg.norm(b_red)**2 - \
                    2/(N-1)**2*np.sum(A_sum[ind]*b_red) + \
                    (2*N-1)/(N*(N-1)**2)*np.sum(np.ravel(A_square_sum)[ind]) -\
                    2/(N-1)*np.sum(A_sum[ind]*sol_red) + \
                    2*np.sum(sol_red*b_red)
            sol[ind, i] = np.squeeze(sol_red)

            if i > 1 and \
                self.model_selection_options['stop_if_error_increase'] and \
                    err[i] > 2*err[i-1]:
                print('stop_if_error_increase')
                err[i+1:] = np.inf

        ind = np.argmin(err)

        output['error'] = err[ind]
        output['error_path'] = err
        output['ind'] = ind
        output['pattern'] = pattern[:, ind]
        output['pattern_path'] = pattern
        output['optimal_solution'] = sol[:, ind]
        output['solution_path'] = sol

        return sol[:, ind], output
