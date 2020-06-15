'''
Module linear_model_learning_square_loss.

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
from scipy.sparse import diags
import tensap


class LinearModelLearningSquareLoss(tensap.LinearModelLearning):
    '''
    Class LinearModelLearningSquareLoss.

    Attributes
    ----------
    weights : list or numpy.ndarray, optional
        The arrays for the weighted least-squares minimization. The default is
        None.
    linear_solver : str
        The selected solver: 'solve' to directly solve the normal equation
        using numpy.solve, 'qr' to perform a QR decomposition, and solve
        the resulting system using numpy.solve. The default is 'qr'.
    shared_coefficients : bool
        When approximating a vector-valued function, indicates if the
        coefficients of the approximation are common to all the outputs. The
        default is True.

    '''

    def __init__(self, weights=None):
        '''
        Constructor for the class LinearModelLearningSquareLoss.

        When approximating a vector-valued function, setting sharedCoefficients
        to false will independently compute y.shape[1] sets of coefficients,
        whereas setting it to true will compute one set of coefficients shared
        across all the outputs. In that case, basisEval should be an array of
        shape (n-by-N-by-D), with n the size of the dataset, N the number of
        basis functions and D the number of outputs.

        Parameters
        ----------
        weights : list or numpy.ndarray, optional
            The arrays for the weighted least-squares minimization. The default
            is None.

        Returns
        -------
        None.

        '''
        super().__init__(tensap.SquareLossFunction())
        self.weights = weights
        self.linear_solver = 'qr'
        self.shared_coefficients = True

    def solve(self):
        '''
        Solution (Ordinary or Regularized) of the Least-Squares problem and
        cross-validation procedure.

        Returns
        -------
        sol : numpy.ndarray or tensap.FunctionalBasisArray
            The solution of the minimization problem.
        output : dict
            Outputs of the algorithm.

        '''
        self.initialize()

        A = self.basis_eval
        y = self.training_data[1]

        if self.shared_coefficients:
            if np.ndim(A) == 3:
                assert (np.ndim(y) == 1 and A.shape[2] == 1) or \
                    (np.ndim(y) == 2 and A.shape[2] == y.shape[1]), \
                    'A.shape[2] should be equal to y.shape[1].'

                A = np.transpose(A, [0, 2, 1])
                A = np.reshape(A, [-1, A.shape[2]], order='F')
                y = np.reshape(y, [-1, 1], order='F')
                self.basis_eval = A
                self.training_data[1] = y
        if np.ndim(y) == 2:
            n = y.shape[1]
        else:
            n = 1

        output = {}

        if self.weights is not None:
            weights = diags(np.sqrt(self.weights))
            A = np.matmul(weights, A)
            y = np.matmul(weights, y)

        if not self.basis_adaptation:
            if not self.regularization:
                sol, output = self._solve_ols()
            else:
                if n == 1:
                    sol, output = self._solve_regularized_ls()
                else:
                    sol = np.zeros([A.shape[1], n])
                    output['error'] = np.zeros(n)
                    output['outputs'] = np.empty(n, dtype=object)
                    for ind in range(n):
                        self.training_data[1] = y[:, ind]
                        sol[:, ind], output_tmp = self._solve_regularized_ls()
                        output['error'][ind] = output_tmp['error']
                        output['outputs'][ind] = output_tmp
        else:
            if n == 1:
                sol, output = self._solve_basis_adaptation()
            else:
                sol = np.zeros([A.shape[1], n])
                output['error'] = np.zeros(n)
                output['outputs'] = np.empty(n, dtype=object)
                for ind in range(n):
                    self.training_data[1] = y[:, ind]
                    sol[:, ind], output_tmp = self._solve_basis_adaptation()
                    output['error'][ind] = output_tmp['error']
                    output['outputs'][ind] = output_tmp

        if self.test_error:
            if n == 1:
                f_eval = np.matmul(self.basis_eval_test, sol)
                output['test_error'] = self.loss_function.test_error(
                    f_eval, self.test_data)
            else:
                output['test_error'] = np.zeros([1, n])
                f_eval = np.matmul(self.basis_eval_test, sol)
                for ind in range(n):
                    test_data = [self.test_data[0], self.test_data[1][:, ind]]
                    output['test_error'][ind] = self.loss_function.test_error(
                        f_eval, test_data)

        if self.basis is not None:
            sol = tensap.FunctionalBasisArray(sol, self.basis, n)

        return sol, output

    def _solve_ols(self):
        '''
        Solution of the Ordinary Least-Squares (OLS) problem and
        cross-validation procedure (if requested).

        Returns
        -------
        sol : numpy.ndarray
            The solution of the minimization problem.
        output : dict
            Outputs of the algorithm.

        '''
        A = self.basis_eval
        y = self.training_data[1]
        output = {}

        if self.linear_solver == 'solve':
            B = np.matmul(np.transpose(A), A)
            try:
                sol = np.linalg.solve(B, np.matmul(np.transpose(A), y))
            except Exception:
                sol = np.linalg.lstsq(B, np.matmul(np.transpose(A), y),
                                      None)[0]
        elif self.linear_solver == 'qr':
            q_A, r_A = np.linalg.qr(A)
            try:
                sol = np.linalg.solve(r_A, np.matmul(np.transpose(q_A), y))
            except Exception:
                sol = np.linalg.lstsq(r_A, np.matmul(np.transpose(q_A), y),
                                      None)[0]

        if self.error_estimation:
            second_moment = np.var(y, 0) + np.mean(y, 0)**2
            if self.error_estimation_type == 'residuals':
                delta = y - np.matmul(A, sol)
                error = np.mean(delta**2, 0) / second_moment
            else:
                if self.linear_solver == 'solve':
                    try:
                        B_inv = np.linalg.inv(B)
                    except Exception:
                        B_inv = np.linalg.pinv(B)
                    error, delta = self._compute_cv_error(y, second_moment,
                                                          A, sol,
                                                          self.linear_solver,
                                                          B_inv)
                elif self.linear_solver == 'qr':
                    error, delta = self._compute_cv_error(y, second_moment,
                                                          A, sol,
                                                          self.linear_solver,
                                                          q_A, r_A)
            output['error'] = error
            output['delta'] = delta

        output['flag'] = 2
        return sol, output

    def _solve_regularized_ls(self):
        '''
        Solution of the Regularized Least-Squares problem and cross-validation
        procedure.

        Returns
        -------
        sol : numpy.ndarray
            The solution of the minimization problem.
        output : dict
            Outputs of the algorithm.

        '''
        from sklearn import linear_model

        A = self.basis_eval
        y = np.squeeze(self.training_data[1])
        output = {}

        N, P = A.shape
        solpath = []

        options = self.regularization_options
        if self.regularization_type == 'l0':
            options_loc = {i: output[i] for i in output if i != 'alpha'}
            D = np.linalg.norm(A, axis=0)
            A_omp = np.matmul(A, diags(1/D))
            solpath = linear_model.orthogonal_mp(A_omp, y,
                                                 copy_X=True,
                                                 n_nonzero_coefs=np.min((N,
                                                                         P)),
                                                 return_path=True,
                                                 **options_loc)
            solpath = np.atleast_2d(solpath)
            sol = np.matmul(diags(1/D), solpath[:, -1])
        elif self.regularization_type == 'l1':
            reg = linear_model.LassoLars(copy_X=True, fit_intercept=False,
                                         **options)
            reg.fit(A, y)
            sol = reg.coef_
            solpath = reg.coef_path_
        elif self.regularization_type == 'l2':
            reg = linear_model.Ridge(copy_X=True, fit_intercept=False,
                                     **options)
            reg.fit(A, y)
            sol = reg.coef_
        else:
            raise ValueError('Regularization technique not implemented.')

        if self.model_selection and np.linalg.norm(solpath) != 0:
            if 'non_zero_blocks' in self.options:
                rep = np.true(solpath.shape[1])
                rep = np.logical_and(rep, np.any(solpath, 0))
                for block in self.options['non_zero_blocks']:
                    rep = np.logical_and(rep, np.any(solpath[block, :], 0))
                solpath = solpath[:, rep]

            sol, output = self._select_optimal_path(solpath)
        elif self.model_selection:
            # print('solpath does not exist or is empty.')
            pass

        if self.error_estimation and 'error' not in output:
            second_moment = np.var(y, 0) + np.mean(y, 0)**2
            if self.error_estimation_type == 'residuals':
                delta = y - np.matmul(A, sol)
                error = np.mean(delta**2, 0) / second_moment
            else:
                ind = np.nonzero(sol)[0]
                A_red = A[:, ind]
                if self.linear_solver == 'solve':
                    B = np.matmul(np.transpose(A_red), A_red)
                    try:
                        B_inv = np.linalg.inv(B)
                    except Exception:
                        B_inv = np.linalg.pinv(B)
                    error, delta = self._compute_cv_error(y, second_moment,
                                                          A_red, sol[ind],
                                                          self.linear_solver,
                                                          B_inv)
                elif self.linear_solver == 'qr':
                    q_A, r_A = np.linalg.qr(A_red)
                    error, delta = self._compute_cv_error(y, second_moment,
                                                          A_red, sol[ind],
                                                          self.linear_solver,
                                                          q_A, r_A)
            output['error'] = error
            output['delta'] = delta

        output['flag'] = 2
        return sol, output

    def _solve_basis_adaptation(self):
        '''
        Solution of the Least-Squares problem with working-set and
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

    def _select_optimal_path(self, solpath=None):
        '''
        Selection of a solution using (corrected) relative (leave-one-out or
        k-fold) cross-validation error.

        The dictionnary output contains:
            - error: the (corrected) relative (leave-one-out or k-fold)
              cross-validation error estimate,
            - error_path: the regularization path of (corrected) relative
              (leave-one-out or k-fold) cross-validation error estimates,
            - ind: the index of the optimal pattern,
            - pattern: the optimal sparsity pattern,
            - pattern_path: the regularization path of sparsity patterns,
            - solution_path: the regularization path of potential non-zero
              solutions with different sparsity patterns,
            - solution_path_OLS: the regularization path of the OLS solutions,
            - optimal_solution: the optimal solution,
            - delta: the optimal residual,
            - delta_path: the regularization paths of the residuals.

        Parameters
        ----------
        solpath : numpy.ndarray, optional
            Array of shape (P, m) whose columns give m potential solutions with
            different sparsity patterns. The default is None, returning the
            solution of the OLS problem without basis adaptation.

        Returns
        -------
        numpy.ndarray
            The optimal solution of the minimization problem.
        dict
            Outputs of the algorithm.

        '''
        if solpath is None:
            return self._solve_ols()

        A = self.basis_eval
        y = self.training_data[1]
        output = {}
        linear_solver = self.linear_solver

        pattern = solpath != 0
        rep = np.any(pattern, 0)
        pattern = pattern[:, rep]
        solpath = solpath[:, rep]

        _, ind = np.unique(pattern, return_index=True, axis=1)
        ind = np.sort(ind)
        solpath = solpath[:, ind]
        pattern = pattern[:, ind]

        sol = np.zeros(pattern.shape)

        if 'gram_matrix' in self.error_estimation_options:
            gram_matrix = self.error_estimation_options['gram_matrix']

        second_moment = np.var(y, 0) + np.mean(y, 0)**2

        ind = np.sum(pattern != 0, 0) <= A.shape[0]
        err = np.zeros(np.count_nonzero(ind))
        delta = np.zeros([y.shape[0], np.count_nonzero(ind)])
        pattern = pattern[:, ind]
        solpath = solpath[:, ind]

        if np.linalg.norm(solpath) == 0:
            return self._solve_ols()

        for i in range(pattern.shape[1]):
            ind = pattern[:, i]

            A_red = A[:, ind]
            if 'gram_matrix' in self.error_estimation_options:
                self.error_estimation_options['gram_matrix'] =  \
                    gram_matrix[ind, ind]

            if A_red.shape[1] > A_red.shape[0]:
                self.linear_solver = 'solve'
            else:
                self.linear_solver = linear_solver

            if self.linear_solver == 'qr':
                q_A, r_A = np.linalg.qr(A_red)
                try:
                    sol_red = np.linalg.solve(r_A,
                                              np.matmul(np.transpose(q_A), y))
                except Exception:
                    sol_red = np.linalg.lstsq(r_A,
                                              np.matmul(np.transpose(q_A), y),
                                              None)[0]

                err[i], delta[:, i] = self._compute_cv_error(y, second_moment,
                                                             A_red, sol_red,
                                                             'qr', q_A, r_A)
            elif self.linear_solver == 'solve':
                try:
                    C_red = np.linalg.inv(np.matmul(np.transpose(A_red),
                                                    A_red))
                except Exception:
                    C_red = np.linalg.pinv(np.matmul(np.transpose(A_red),
                                                     A_red))

                sol_red = np.matmul(C_red, np.matmul(np.transpose(A_red), y))
                err[i], delta[:, i] = self._compute_cv_error(y, second_moment,
                                                             A_red, sol_red,
                                                             'solve', C_red)

            sol[ind, i] = np.squeeze(sol_red)

            if i > 1 and \
                self.model_selection_options['stop_if_error_increase'] and \
                    err[i] > 2*err[i-1]:
                print('stop_if_error_increase')
                err[i+1:] = np.inf
                delta[:, i+1:] = np.inf

        ind = np.argmin(err)

        output['error'] = err[ind]
        output['error_path'] = err
        output['ind'] = ind
        output['pattern'] = pattern[:, ind]
        output['pattern_path'] = pattern
        output['solution_path'] = solpath
        output['optimal_solution'] = sol[:, ind]
        output['solution_path_OLS'] = sol
        output['delta'] = delta[:, ind]
        output['delta_path'] = delta

        return sol[:, ind], output

    def _compute_cv_error(self, y, second_moment, A, sol, linear_solver=None,
                          *args):
        '''
        Relative cross-validation error estimates and residuals.

        For self.error_estimation_type = 'leave_out': compute the relative
        leave-one-out cross-validation error for the coefficients matrix sol
        using the fast leave-one-out cross-validation procedure
        [Cawlet & Talbot 2004] based on the Bartlett matrix inversion formula
        (special case of the Sherman-Morrison-Woodbury formula).
        If self.error_estimation_type.correction == True, compute the corrected
        relative leave-one-out cross-validation error for the coefficients
        matrix sol.

        For self.error_estimation_type = 'k_fold': compute the  k-fold
        cross-validation error for the coefficients matrix sol using the fast
        k-fold cross-validation procedure based on the
        Sherman-Morrison-Woodbury formula.
        self.error_estimation_options['number_of_folds']: number of folds
        (only for the k-fold cross-validation procedure), min(10, N) by default
        where N is the number of samples.
        If self.error_estimation_type.correction == True, compute the corrected
        relative k-fold cross-validation error for the coefficients matrix sol.


        Parameters
        ----------
        y : numpy.ndarray
            The evaluations of response vector.
        second_moment : float or numpy.ndarray
            The empirical second moment of y.
        A : numpy.ndarray
            The evaluations of the basis functions.
        sol : numpy.ndarray
            The coefficients of the approximation.
        linear_solver : str, optional
            The selected linear solver. The default is None, selecting 'solve'.
        *args : tuple
            Additional arguments (if required).

        Raises
        ------
        NotImplementedError
            If the requested cross-validation estimator is not implemented.

        Returns
        -------
        numpy.ndarray
            The (corrected) relative (leave-one-out or k-fold) cross-validation
            error estimate.
        delta : numpy.ndarray
            The residuals.

        '''
        if np.ndim(y) == 2:
            N, n = y.shape
        else:
            N = y.shape[0]
            n = 1

        P = A.shape[1]
        err = np.zeros(n)
        delta = np.zeros(y.shape)

        if self.error_estimation_type == 'k_fold':
            if 'number_of_folds' not in self.error_estimation_options:
                self.error_estimation_options['number_of_folds'] = np.min([10,
                                                                           N])
            if self.error_estimation_options['number_of_folds'] == N:
                self.error_estimation_type = 'leave_out'

        if not args and linear_solver is None:
            if P > N:
                try:
                    C = np.linalg.inv(np.matmul(np.transpose(A), A))
                except Exception:
                    C = np.linalg.pinv(np.matmul(np.transpose(A), A))
                linear_solver = 'solve'
            else:
                q_A, r_A = np.linalg.qr(A)
                linear_solver = 'qr'
        else:
            if linear_solver == 'qr':
                q_A, r_A = args
            elif linear_solver == 'solve':
                C = args[0]

        # Compute the absolute cross-validation error (cross-validation error
        # estimate of the mean-squared error), also called mean predicted
        # residual sum of squares (PRESS) or empirical mean-squared predicted
        # residual
        if self.error_estimation_type == 'leave_out':
            if N-1 < P:
                err = np.full(err.shape, np.inf)
                delta = np.squeeze(np.full(delta.shape, np.inf))
                if isinstance(err, np.ndarray) and err.size == 1:
                    err = err[0]
                return err, delta

            # Compute the predicted residuals using the Bartlett matrix
            # inversion formula (special case of the Sherman-Morrison-Woodbury
            # formula)
            if linear_solver == 'solve':
                T = np.sum(np.transpose(A) * np.matmul(C, np.transpose(A)), 0)
            elif linear_solver == 'qr':
                T = np.sum(q_A**2, 1)

            with np.errstate(divide='ignore', invalid='ignore'):
                delta = (y - np.matmul(A, sol)) / (1-np.reshape(T, y.shape))
            delta = np.squeeze(delta)
            # Compute the absolute cross-validation error
            err = np.mean(delta**2, 0)
        elif self.error_estimation_type == 'k_fold':
            from sklearn.model_selection import KFold

            n_folds = np.min(
                (N, self.error_estimation_options['number_of_folds']))

            if N * (1 - 1 / n_folds) < P:
                # print('Not enough samples for performing OLS on the ' +
                #       'training set.')
                err = np.full(err.shape, np.inf)
                delta = np.squeeze(np.full(delta.shape, np.inf))
                if isinstance(err, np.ndarray) and err.size == 1:
                    err = err[0]
                return err, delta

            cvp = KFold(n_splits=n_folds, shuffle=True).split(A)

            errors = []
            delta = np.zeros(y.shape)

            if linear_solver == 'solve':
                H = np.matmul(A, np.matmul(C, np.transpose(A)))
            elif linear_solver == 'qr':
                H = np.matmul(q_A, np.transpose(q_A))

            for _, test in cvp:
                # Compute the predicted residual for the current fold using the
                # Sherman-Morrison-Woodbury formula
                if n == 1:
                    y_loc = y[test]
                else:
                    y_loc = y[test, :]
                delta_loc = np.linalg.solve(np.eye(len(test)) -
                                            H[np.ix_(test, test)],
                                            y_loc-np.matmul(A[test, :], sol))
                # Compute the absolute cross-validation error for the current
                # fold
                errors.append(np.mean(delta_loc**2))
                if n == 1:
                    delta[test] = delta_loc
                else:
                    delta[test, :] = delta_loc

            # Average over the folds
            err = np.mean(errors)
            delta = np.squeeze(delta)
        else:
            raise NotImplementedError('Cross-validation {} not implemented'.
                                      format(self.error_estimation_type))

        # Compute the relative cross-validation error
        err = err / second_moment

        # Compute the corrected relative cross-validation error to reduce the
        # sensitivity of the error estimate to overfitting (the non-corrected
        # cross-validation error estimate underpredicts the error in L^2-norm
        # (generalization error))
        if linear_solver == 'qr' and \
                1 / np.linalg.cond(r_A) < np.finfo(float).eps:
            pass
        elif self.error_estimation_options['correction']:
            if N != P:
                if linear_solver == 'qr':
                    try:
                        inv_rA = np.linalg.inv(r_A)
                    except Exception:
                        inv_rA = np.linalg.pinv(r_A)
                    C = np.matmul(inv_rA, np.transpose(inv_rA))

                if 'gram_matrix' in self.error_estimation_options:
                    # Direct Eigenvalue Estimator (DEE) in the case of non
                    # orthonormal bases [Chapelle, Vapnik & Bengio, 2002],
                    # [Blatman & Sudret, 2011].
                    # The Gram matrix must be provided. Independence between
                    # variables is assumed
                    corr = (N / (N-P)) * (1 + np.trace(np.matmul(
                        C, self.error_estimation_options['gram_matrix'])))
                else:
                    # Direct Eigenvalue Estimator
                    # [Chapelle, Vapnik & Bengio, 2002],
                    # [Blatman & Sudret, 2011] -> accurate even when N is
                    # not >> P (corr ~ 1+2*P/N when N->Inf, as trace(C) ~ P/N
                    # when N->Inf)
                    corr = (N / (N-P)) * (1 + np.trace(C))

                if corr > 0:
                    err *= corr

        if isinstance(err, np.ndarray) and err.size == 1:
            err = err[0]
        return np.sqrt(err), delta
