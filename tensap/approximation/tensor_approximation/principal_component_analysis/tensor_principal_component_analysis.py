'''
Module tensor_principal_component_analysis.

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


class TensorPrincipalComponentAnalysis:
    '''
    Class TensorPrincipalComponentAnalysis: principal component analysis of an
    algebraic tensor.

    Attributes
    ----------
    display : bool
        Boolean specifying the verbosity of the methods.
    pca_sampling_factor : int
        A factor to determine the number of samples N for the estimation of
        the principal components (1 by default):
            - if prescribed precision, N = pca_sampling_factor*N_alpha,
            - if prescribed rank, N = pca_sampling_factor*t.
    pca_adaptive_sampling : bool
        Adaptive sampling to determine the principal components with prescribed
        precision.
    tol : int or float or list or numpy.ndarray
        An array containing the prescribed relative precisions (the length
        depends on the format).
        If len(tol)==1, use the same value for all alpha.
        Set tol = inf to prescribe the rank.
    max_rank : int or list or numpy.ndarray
        An array containing the maximum alpha-ranks (the length depends on the
        format).
        If len(max_rank)==1, use the same value for all alpha.
        Set max_rank = np.inf to prescribe the precision.

    '''

    def __init__(self):
        '''
        Constructor for the class TensorPrincipalComponentAnalysis.

        Returns
        -------
        None.

        '''
        self.display = True
        self.pca_sampling_factor = 1
        self.pca_adaptive_sampling = False
        self.tol = 1e-8
        self.max_rank = np.inf

    def alpha_principal_components(self, fun, shape, alpha, tol, B_alpha,
                                   I_alpha):
        '''
        Evaluate the alpha-principal components of a tensor f.

        For alpha in {0,...,d-1}, it evaluates the alpha-principal components
        of a tensor f, meaning the principal components of the matricisations
        f_alpha(i_alpha,i_notalpha), where i_alpha and i_notalpha are groups of
        indices.

        It evaluates f_alpha on the product of a set of indices in dimension
        alpha (of size Nalpha) and a set of random indices (N samples) in the
        complementary dimensions.
        Then, it computes approximations of the alpha-principal components
        in a given basis phi_1(i_alpha) ... phi_Nalpha(i_alpha).

        If t is an integer, t is the rank (number of principal components).
        If t<1, the rank (number of principal components) is determined such
        that the relative error after truncation is t.

        Parameters
        ----------
        fun : fun or tensap.Function
            Function of d variables i_1, ..., i_d which returns the entries of
            the tensor.
        shape : list or numpy.ndarray
            The shape of the tensor.
        alpha : int
            An array containing a tuple in {0,...,d-1}.
        tol : int or float
            The number of principal components or a positive number smaller
            than 1 (tolerance).
        B_alpha : numpy.ndarray
            Array of shape (N_\alpha, N_\alpha) whose i-th column is the
            evaluation of phi_i at the set of indices i_alpha in Ialpha.
        I_alpha : numpy.ndarray
            Array of shape (N_alpha, #alpha) containing N_alpha tuples i_alpha.

        Returns
        -------
        pc : numpy.ndarray
            The principal components of the tensor.
        output : dict
            A dictionnary of outputs, containing the singular values
            corresponding to the principal components, as well as the set of
            indices at which the tensor has been evaluated.

        '''
        alpha = np.atleast_1d(alpha)
        B_alpha = np.atleast_2d(B_alpha)
        I_alpha = np.array(I_alpha)

        X = tensap.random_multi_indices(shape)
        d = len(shape)
        not_alpha = np.setdiff1d(range(d), alpha)

        if tol < 1:
            N = self.pca_sampling_factor * B_alpha.shape[1]
        else:
            N = self.pca_sampling_factor * tol
        N = int(np.ceil(N))

        X_not_alpha = tensap.RandomVector(X.random_variables[not_alpha])
        I_not_alpha = X_not_alpha.random(N)

        alpha_not_alpha = np.concatenate((alpha, not_alpha))
        ind = [np.nonzero(alpha_not_alpha == x)[0][0] for x in range(d)]

        output = {}
        if tol < 1 and self.pca_adaptive_sampling:
            A = tensap.FullTensor(np.zeros((I_alpha.shape[0], 0)), 2,
                                  [I_alpha.shape[0], 0])
            for k in range(N):
                product_grid = tensap.FullTensorGrid(
                    [I_alpha, I_not_alpha[k, :]]).array()
                A_k = np.linalg.solve(B_alpha, fun(product_grid[:, ind]))
                A.data = np.column_stack((A.data, A_k))
                pc, sin_val = A.principal_components(tol)
                if sin_val[-1, -1] < 1e-15 or pc.shape[1] < \
                        np.ceil((k+1) / self.pca_sampling_factor):
                    break
            output['number_of_evaluations'] = I_alpha.shape[0] * (k+1)
        else:
            product_grid = tensap.FullTensorGrid([I_alpha,
                                                  I_not_alpha]).array()
            A = fun(product_grid[:, ind])
            A = np.reshape(A, [B_alpha.shape[0], N], 'F')
            A = np.linalg.solve(B_alpha, A)
            A = tensap.FullTensor(A, 2, [B_alpha.shape[0], N])
            pc, sin_val = A.principal_components(tol)
            output['number_of_evaluations'] = I_alpha.shape[0] * N

        output['singular_values'] = sin_val
        output['samples'] = product_grid
        return pc, output

    def hopca(self, fun, shape):
        '''
        Return the set of alpha-principal components of an algebraic tensor,
        for all alpha in {0,1,...,d-1}.

        For prescribed precision, set TPCA.max_rank = np.inf and TPCA.tol to
        the desired precision (possibly an array of length d).

        For prescribed rank, set TPCA.tol = np.inf and TPCA.max_rank to the
        desired rank (possibly an array of length d).

        See also the documentation of the class
        TensorPrincipalComponentAnalysis.

        Parameters
        ----------
        fun : fun or tensap.Function
            Function of d variables i_1, ..., i_d which returns the entries of
            the tensor.
        shape : list or numpy.ndarray
            The shape of the tensor.

        Raises
        ------
        ValueError
            If the provided tolerance and max ranks are not correct.

        Returns
        -------
        f_pc : list
            List of the alpha-principal components of the tensor.
        output : list
            List containing the outputs of the method
            alpha_principal_components.

        '''
        solver = deepcopy(self)
        d = len(shape)

        if np.ndim(self.tol) == 0 or len(self.tol) == 1:
            solver.tol = np.full(d, self.tol)
        elif len(self.tol) != d:
            raise ValueError('tol should be a scalar or an array of length d.')

        if np.ndim(self.max_rank) == 0 or len(self.max_rank) == 1:
            solver.max_rank = np.full(d, self.max_rank)
        elif len(self.max_rank) != d:
            raise ValueError('max_rank should be a scalar or an array of ' +
                             'length d.')

        f_pc = []
        output = []
        for alpha in range(d):
            I_alpha = np.reshape(np.arange(shape[alpha]), (-1, 1))
            B_alpha = np.eye(shape[alpha])
            tol_alpha = np.min((solver.tol[alpha], solver.max_rank[alpha]))
            f_pc_alpha, output_alpha = \
                solver.alpha_principal_components(fun, shape, alpha, tol_alpha,
                                                  B_alpha, I_alpha)
            f_pc.append(f_pc_alpha)
            output.append(output_alpha)
        return f_pc, output

    def tucker_approximation(self, fun, shape):
        '''
        Approximation of a tensor of order d in Tucker format based on a
        Principal Component Analysis.

        For a prescribed precision, set TPCA.max_rank = np.inf and TPCA.tol to
        the desired precision (possibly an array of length d).

        For a prescribed rank, set TPCA.tol = np.inf and TPCA.max_rank to the
        desired rank (possibly an array of length d).

        See also the documentation of the class
        TensorPrincipalComponentAnalysis.

        Parameters
        ----------
        fun : fun or tensap.Function
            Function of d variables i_1, ..., i_d which returns the entries of
            the tensor.
        shape : list or numpy.ndarray
            The shape of the tensor.

        Raises
        ------
        ValueError
            If the provided tolerance and max ranks are not correct.

        Returns
        -------
        tensap.TreeBasedTensor
            A tensor in tree based format with a trivial tree.
        dict
            Dictionnary containing the outputs of the method.

        '''
        solver = deepcopy(self)
        d = len(shape)
        tree = tensap.DimensionTree.trivial(d)

        if np.ndim(self.tol) == 1 and len(self.tol) == d:
            tol = solver.tol
            solver.tol = np.zeros(d+1)
            solver.tol[tree.dim2ind-1] = tol
        elif np.ndim(self.tol) == 1 and len(self.tol) > 1:
            raise ValueError('tol should be a scalar or an array of length d.')

        if np.ndim(self.max_rank) == 1 and len(self.max_rank) == d:
            rank = solver.max_rank
            solver.max_rank = np.zeros(d+1)
            solver.max_rank[tree.dim2ind-1] = rank
        elif np.ndim(self.max_rank) == 1 and len(self.max_rank) > 1:
            raise ValueError('max_rank should be a scalar or an array of ' +
                             'length d.')

        return solver.tree_based_approximation(fun, shape, tree)

    def tt_approximation(self, fun, shape):
        '''
        Approximation of a tensor of order d in tensor train format based on a
        Principal Component Analysis.

        For a prescribed precision, set TPCA.max_rank = np.inf and TPCA.tol to
        the desired precision (possibly an array of length d-1).

        For a prescribed rank, set TPCA.tol = np.inf and TPCA.max_rank to the
        desired rank (possibly an array of length d-1).

        See also the documentation of the class
        TensorPrincipalComponentAnalysis.

        Parameters
        ----------
        fun : fun or tensap.Function
            Function of d variables i_1, ..., i_d which returns the entries of
            the tensor.
        shape : list or numpy.ndarray
            The shape of the tensor.

        Raises
        ------
        ValueError
            If the provided tolerance and max ranks are not correct.

        Returns
        -------
        tensap.TreeBasedTensor
            A tensor in tree based format with a linear tree.
        dict
            Dictionnary containing the outputs of the method.

        '''
        solver = deepcopy(self)
        d = len(shape)
        tree = tensap.DimensionTree.linear(d)
        is_active_node = np.full(tree.nb_nodes, True)
        is_active_node[tree.dim2ind[1:]-1] = False
        rep_tt = np.nonzero(is_active_node)[0]
        rep_tt = np.flip(rep_tt[1:])

        if np.ndim(self.tol) == 1 and len(self.tol) == d-1:
            tol = solver.tol
            solver.tol = np.zeros(tree.nb_nodes)
            solver.tol[rep_tt] = tol
        elif np.ndim(self.tol) == 1 and len(self.tol) > 1:
            raise ValueError('tol should be a scalar or an array of length ' +
                             'd-1.')

        if np.ndim(self.max_rank) == 1 and len(self.max_rank) == d-1:
            rank = solver.max_rank
            solver.max_rank = np.zeros(tree.nb_nodes)
            solver.max_rank[rep_tt] = rank
        elif np.ndim(self.max_rank) == 1 and len(self.max_rank) > 1:
            raise ValueError('max_rank should be a scalar or an array of ' +
                             'length d-1.')

        return solver.tree_based_approximation(fun, shape, tree,
                                               is_active_node)

    def tree_based_approximation(self, fun, shape, tree, is_active_node=None):
        '''
        Approximation of a tensor of order d in tree based tensor format based
        on a Principal Component Analysis.

        For a prescribed precision, set TPCA.max_rank = np.inf and TPCA.tol to
        the desired precision (possibly an array of length d-1).

        For a prescribed rank, set TPCA.tol = np.inf and TPCA.max_rank to the
        desired rank (possibly an array of length d-1).

        See also the documentation of the class
        TensorPrincipalComponentAnalysis.

        Parameters
        ----------
        fun : fun or tensap.Function
            Function of d variables i_1, ..., i_d which returns the entries of
            the tensor.
        shape : list or numpy.ndarray
            The shape of the tensor.
        tree : tensap.DimensionTree
            The required dimension tree.
        is_active_node : list or numpy.ndarray, optional
            An array of booleans indicating which nodes of the tree are active.
            The default is None, settings all the nodes active.

        Raises
        ------
        ValueError
            If the provided tolerance and max ranks are not correct.

        Returns
        -------
        tensap.TreeBasedTensor
            A tensor in tree based format.
        dict
            Dictionnary containing the outputs of the method.

        '''
        solver = deepcopy(self)
        d = len(shape)

        if is_active_node is None:
            is_active_node = np.full(tree.nb_nodes, True)

        if (np.ndim(self.tol) == 0 or len(self.tol) == 1) and self.tol < 1:
            solver.tol /= np.sqrt(np.count_nonzero(is_active_node)-1)

        if np.ndim(self.tol) == 0 or len(self.tol) == 1:
            solver.tol = np.full(tree.nb_nodes, solver.tol)
        elif len(self.tol) != tree.nb_nodes:
            raise ValueError('tol should be a scalar or an array of length ' +
                             'tree.nb_nodes.')

        if np.ndim(self.max_rank) == 0 or len(self.max_rank) == 1:
            solver.max_rank = np.full(tree.nb_nodes, self.max_rank)
        elif len(self.max_rank) != tree.nb_nodes:
            raise ValueError('max_rank should be a scalar or an array of ' +
                             'length tree.nb_nodes.')

        grids = [np.reshape(np.arange(x), (-1, 1)) for x in shape]
        alpha_basis = np.empty(tree.nb_nodes, dtype=object)
        alpha_grids = np.empty(tree.nb_nodes, dtype=object)
        outputs = np.empty(tree.nb_nodes, dtype=object)
        samples = np.empty(tree.nb_nodes, dtype=object)
        tensors = [[]]*tree.nb_nodes
        number_of_evaluations = 0
        for nu in range(d):
            alpha = tree.dim2ind[nu]
            B_alpha = np.eye(shape[nu])
            if is_active_node[alpha-1]:
                tol_alpha = np.min((solver.tol[alpha-1],
                                    solver.max_rank[alpha-1]))
                pc_alpha, outputs[alpha-1] = \
                    solver.alpha_principal_components(fun, shape, nu,
                                                      tol_alpha, B_alpha,
                                                      grids[nu])
                samples[alpha-1] = outputs[alpha-1]['samples']
                shape_alpha = [shape[nu], pc_alpha.shape[1]]
                tensors[alpha-1] = tensap.FullTensor(pc_alpha, 2, shape_alpha)

                B_alpha = np.matmul(B_alpha, pc_alpha)
                I_alpha = tensap.magic_indices(B_alpha)[0]
                alpha_grids[alpha-1] = grids[nu][I_alpha, :]
                alpha_basis[alpha-1] = B_alpha[I_alpha, :]

                number_of_evaluations += outputs[alpha-1][
                    'number_of_evaluations']
                if solver.display:
                    print('alpha = %i : rank = %i, nb_eval = %i' %
                          (alpha, shape_alpha[-1],
                           outputs[alpha-1]['number_of_evaluations']))
            else:
                alpha_grids[alpha-1] = grids[nu]
                alpha_basis[alpha-1] = B_alpha

        for level in np.arange(np.max(tree.level), 0, -1):
            for alpha in np.intersect1d(tree.nodes_with_level(level),
                                        tree.internal_nodes):
                S_alpha = tree.children(alpha)
                B_alpha = TensorPrincipalComponentAnalysis.\
                    _tensor_product_b_alpha(alpha_basis[S_alpha-1])
                alpha_grids[alpha-1] = \
                    tensap.FullTensorGrid(alpha_grids[S_alpha-1]).array()

                tol_alpha = np.min((solver.tol[alpha-1],
                                    solver.max_rank[alpha-1]))
                pc_alpha, outputs[alpha-1] = \
                    solver.alpha_principal_components(fun, shape,
                                                      tree.dims[alpha-1],
                                                      tol_alpha,
                                                      B_alpha,
                                                      alpha_grids[alpha-1])
                samples[alpha-1] = outputs[alpha-1]['samples']
                shape_alpha = np.concatenate(([x.shape[1] for
                                              x in alpha_basis[S_alpha-1]],
                                             [pc_alpha.shape[1]]))
                tensors[alpha-1] = tensap.FullTensor(pc_alpha, len(S_alpha)+1,
                                                     shape_alpha)

                B_alpha = np.matmul(B_alpha, pc_alpha)
                I_alpha = tensap.magic_indices(B_alpha)[0]
                alpha_grids[alpha-1] = alpha_grids[alpha-1][I_alpha, :]
                alpha_basis[alpha-1] = B_alpha[I_alpha, :]
                number_of_evaluations += outputs[alpha-1][
                    'number_of_evaluations']
                if solver.display:
                    print('alpha = %i : rank = %i, nb_eval = %i' %
                          (alpha, shape_alpha[-1],
                           outputs[alpha-1]['number_of_evaluations']))

        alpha = tree.root
        S_alpha = tree.children(alpha)
        B_alpha = TensorPrincipalComponentAnalysis.\
            _tensor_product_b_alpha(alpha_basis[S_alpha-1])
        I_alpha = tensap.FullTensorGrid(alpha_grids[S_alpha-1]).array()
        shape_alpha = [x.shape[1] for x in alpha_basis[S_alpha-1]]
        ind = [np.nonzero(tree.dims[alpha-1] == x)[0][0] for x in range(d)]
        tensors[alpha-1] = tensap.FullTensor(
            np.linalg.solve(B_alpha, fun(I_alpha[:, ind])), len(S_alpha),
            shape_alpha)
        alpha_grids[alpha-1] = I_alpha
        number_of_evaluations += I_alpha.shape[0]
        samples[alpha-1] = I_alpha
        if solver.display:
            print('Interpolation - nb_eval = %i' % I_alpha.shape[0])

        f = tensap.TreeBasedTensor(tensors, tree)

        output = {'number_of_evaluations': number_of_evaluations,
                  'samples': samples,
                  'alpha_basis': alpha_basis,
                  'alpha_grids': alpha_grids,
                  'outputs': outputs}

        return f, output

    @staticmethod
    def _tensor_product_b_alpha(Bs):
        '''
        Function used in the method tree_based_approximation.

        Parameters
        ----------
        Bs : list
            List containing s matricices B1 , ..., Bs where Bi is a
            n[i]-by-r[i] array.

        Returns
        -------
        numpy.ndarray
            An array of shape prod(n)-by-prod(r) whose entries are
            B[I , J] = B1[i_1, j_1] ... Bs[i_s, j_s]
            with I = (i_1, ..., is) and J = (j1, ..., js).

        '''
        Bs = [tensap.FullTensor(x, 2, x.shape) for x in Bs]
        B = Bs[0]
        for k in np.arange(1, len(Bs)):
            B = B.tensordot(Bs[k], 0)
        return B.matricize(np.arange(0, B.order-1, 2)).data
