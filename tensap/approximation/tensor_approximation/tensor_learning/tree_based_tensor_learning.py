'''
Module tree_based_tensor_learning.

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
from itertools import combinations
from random import shuffle
import numpy as np
import tensap


class TreeBasedTensorLearning(tensap.TensorLearning):
    '''
    Class TreeBasedTensorLearning.

    See also tensap.TensorLearning.

    Attributes
    ----------
    tree : tensap.DimensionTree
        The dimension tree of the tree-based tensor.
    is_active_node : list or numpy.ndarray
        Booleans indicating if each node is active.

    '''

    def __init__(self, tree, is_active_node, *args):
        '''
        Constructor for the class TreeBasedTensorLearning.

        See also tensap.TensorLearning.

        Parameters
        ----------
        tree : tensap.DimensionTree
            The dimension tree of the tree-based tensor.
        is_active_node : list or numpy.ndarray
            Booleans indicating if each node is active.
        *args : tuple
            Additional parameters.

        Returns
        -------
        None.

        '''
        super().__init__(*args)

        self.tree = tree
        self.is_active_node = list(map(bool, is_active_node))
        self.order = tree.dim2ind.size
        self._number_of_parameters = np.count_nonzero(is_active_node)

        self.initialization_type = 'canonical'
        self.rank_adaptation_options['rank_one_correction'] = True
        self.rank_adaptation_options['theta'] = 0.8
        self.linear_model_learning_parameters[
            'basis_adaptation_internal_nodes'] = False

# %% Standard solver methods
    def initialize(self):
        assert self.tree is not None, \
            'Must provide a DimensionTree object in property tree.'

        if np.isscalar(self.rank) or len(self.rank) == self.order:
            rank = np.zeros(self.tree.nb_nodes, dtype=int)
            rank[self.is_active_node] = self.rank
            self.rank = rank
        self.rank[self.tree.root-1] = self.output_dimension

        shape = [x.shape[1] for x in self.bases_eval]
        if self.initialization_type == 'random':
            f = tensap.TreeBasedTensor.randn(self.tree, self.rank, shape,
                                             self.is_active_node)
        elif self.initialization_type == 'ones':
            f = tensap.TreeBasedTensor.ones(self.tree, self.rank, shape,
                                            self.is_active_node)
        elif self.initialization_type == 'initial_guess':
            f = self.initial_guess
            if not np.all(f.ranks == self.rank):
                tr = tensap.Truncator(tolerance=np.finfo(float).eps,
                                      max_rank=self.rank)
                f = tr.truncate(f)
        elif self.initialization_type == 'mean' or \
                self.initialization_type == 'mean_randomized':
            if not np.all(self.rank == 1):
                raise NotImplementedError('Initialization only implemented ' +
                                          'if np.all(self.rank == 1).')
            if not isinstance(self.training_data, list) or \
                    (isinstance(self.training_data, list) and
                     len(self.training_data) == 1):
                raise NotImplementedError('Initialization type not ' +
                                          'implemented in unsupervised ' +
                                          'learning.')
            if isinstance(self.bases, tensap.FunctionalBases):
                means = self.bases.mean()
            else:
                means = [np.mean(x) for x in self.bases_eval]
            if self.initialization_type == 'mean_randomized':
                means = [x + 0.01*np.random.randn(*x.shape) for x in means]
            means = [tensap.FullTensor(x, 2, [x.shape[0], 1]) for x in means]

            f = tensap.TreeBasedTensor.ones(self.tree, self.rank, shape)
            f.tensors[self.tree.dim2ind-1] = means
            nb_child = len(self.tree.children(self.tree.root))
            if np.ndim(self.training_data[1]) == 2:
                shape = np.concatenate((np.full(nb_child, 1),
                                       [self.training_data[1].shape[1]]))
            else:
                shape = np.full(nb_child, 1)
            f.tensors[self.tree.root-1] = \
                tensap.FullTensor(np.mean(self.training_data[1]), shape=shape)
            f.update_attributes()

            f = f.inactivate_nodes(
                np.nonzero(np.logical_not(self.is_active_node))[0]+1)
        elif self.initialization_type == 'canonical':
            if self.output_dimension != 1:
                print('Canonical initialization not implemented for ' +
                      'outputDimension > 1, performing a random ' +
                      'initialization.')
                f = tensap.TreeBasedTensor.randn(self.tree, self.rank, shape,
                                                 self.is_active_node)
            else:
                f = self.canonical_initialization(np.max(self.rank))
                if not np.all(f.ranks == self.rank):
                    tr = tensap.Truncator(tolerance=np.finfo(float).eps,
                                          max_rank=self.rank)
                    f = tr.truncate(f)
        else:
            raise ValueError('Wrong initialization type.')

        if not np.all(f.ranks == self.rank):
            f = TreeBasedTensorLearning.\
                enriched_edges_to_ranks_random(f, self.rank)

        # Exploration strategy of the tree by increasing level
        tree = f.tree
        exploration_strategy = np.zeros(self._number_of_parameters, dtype=int)
        active_nodes = f.active_nodes
        rep = 0
        for level in range(np.max(tree.level)+1):
            nodes = np.intersect1d(tree.nodes_with_level(level), active_nodes)
            exploration_strategy[rep:rep+len(nodes)] = nodes
            rep += len(nodes)
        self._exploration_strategy = exploration_strategy
        return self, f

    def pre_processing(self, f):
        if isinstance(self.linear_model_learning, (list, np.ndarray)) and \
                len(self.linear_model_learning) != f.tensor.tree.nb_nodes:
            tmp = np.empty(f.tensor.tree.nb_nodes, dtype=object)
            tmp[f.tensor.is_active_node] = self.linear_model_learning
            self.linear_model_learning = tmp
        return self, f

    def randomize_exploration_strategy(self):
        strategy = np.zeros(self._number_of_parameters)
        for level in np.arange(np.max(self.tree.level), -1, -1):
            active_nodes = np.intersect1d(self.tree.nodes_with_level(level),
                                          np.nonzero(self.is_active_node)[0])
            _, ind = np.intersect1d(self._exploration_strategy,
                                    active_nodes,
                                    return_indices=True)
            strategy[ind] = self._exploration_strategy[
                np.random.permutation(ind)]
        return strategy

    def prepare_alternating_minimization_system(self, f, mu):
        tree = f.tensor.tree
        if self.linear_model_learning[mu-1].basis_adaptation:
            if np.isin(mu, tree.internal_nodes):
                if self.linear_model_learning_parameters[
                        'basis_adaptation_internal_nodes']:
                    tr = tensap.Truncator(tolerance=np.finfo(float).eps,
                                          max_rank=np.max(f.tensor.ranks))
                    f.tensor = tr.hsvd(f.tensor)
                elif np.all(f.tensor.is_active_node[tree.children(mu)-1]):
                    self.linear_model_learning[mu-1].basis_adaptation = False
            f.tensor = f.tensor.orth_at_node(mu)
            self.tree = tree
            self.is_active_node = f.tensor.is_active_node
            self.linear_model_learning[mu-1].basis_adaptation_path = \
                self.create_basis_adaptation_path(f.tensor.ranks, mu)
        else:
            f.tensor = f.tensor.orth_at_node(mu)

        grad = f.parameter_gradient_eval(mu)
        if mu == tree.root:
            A = np.reshape(grad.data, [grad.shape[0], -1], order='F')
            self.linear_model_learning[mu-1].initial_guess = np.reshape(
                    f.tensor.tensors[mu-1].data,
                    [-1, f.tensor.ranks[tree.root-1]], order='F')
        else:
            A = np.reshape(grad.data, [grad.shape[0], -1,
                                       f.tensor.ranks[tree.root-1]], order='F')
            self.linear_model_learning[mu-1].initial_guess = np.reshape(
                    f.tensor.tensors[mu-1].data, -1, order='F')

        if isinstance(self.loss_function, tensap.DensityL2LossFunction):
            if isinstance(self.training_data, list) and \
                    len(self.training_data) == 2:
                y = self.training_data[1]
                if isinstance(y, tensap.FunctionalTensor):
                    y = y.tensor
                y = y.orth()
                if tree.is_leaf[mu-1]:
                    a = deepcopy(y)
                    for nod in tree.internal_nodes:
                        a.tensors[nod-1] = tensap.FullTensor(
                            y.tensors[nod-1].data *
                            f.tensor.tensors[nod-1].data,
                            shape=y.tensors[nod-1].shape)
                    ind = np.setdiff1d(range(self.order),
                                       np.nonzero(tree.dim2ind == mu)[0])
                    C = [x.data for x in f.tensor.tensors[tree.dim2ind-1]]
                    b = a.tensor_vector_product([C[x] for x in ind], ind)
                    b = b.tensors[0].data
                else:
                    b = f.tensor.dot(y) / f.tensor.tensors[mu-1].data
            else:
                b = []
        elif isinstance(self.training_data, list) and \
                len(self.training_data) == 2:
            b = self.training_data[1]
            self.linear_model_learning[mu-1].shared_coefficients = \
                mu != tree.root

        return self, A, b, f

    def set_parameter(self, f, mu, coef):
        f.tensor.tensors[mu-1] = \
            tensap.FullTensor(coef, shape=f.tensor.tensors[mu-1].shape)
        f.tensor.tensors[mu-1].is_orth = False
        return f

    def stagnation_criterion(self, f, f0):
        return (f.tensor - f0.tensor).norm() / f0.tensor.norm()

    def final_display(self, f):
        print('Ranks = [%s]' % ', '.join(map(str, f.tensor.ranks)), end='')

    def canonical_initialization(self, rank):
        '''
        Rank-r canonical initialization.

        Parameters
        ----------
        rank : int
            The rank of the canonical initialization.

        Returns
        -------
        tensap.TreeBasedTensor
            The rank-r canonical initialization..

        '''
        solver = tensap.CanonicalTensorLearning(self.order, self.loss_function)
        if isinstance(self.linear_model_learning, list):
            solver.linear_model_learning = self.linear_model_learning[0]
        else:
            solver.linear_model_learning = self.linear_model_learning
        solver.alternating_minimization_parameters = \
            deepcopy(self.alternating_minimization_parameters)
        solver.tolerance['on_stagnation'] = np.finfo(float).eps
        solver.tolerance['on_error'] = np.finfo(float).eps
        solver.bases = self.bases
        solver.bases_eval = self.bases_eval
        solver.bases_eval_test = self.bases_eval_test
        solver.display = False
        solver.alternating_minimization_parameters['display'] = False
        solver.initialization_type = 'mean'
        solver.rank_adaptation = True
        solver.rank_adaptation_options['max_iterations'] = rank
        solver.bases_adaptation_path = self.bases_adaptation_path
        solver.test_error = self.test_error
        solver.training_data = self.training_data
        solver.test_data = self.test_data
        solver._warnings = self._warnings

        f = solver.solve()[0]
        return f.tensor.tree_based_tensor(self.tree, self.is_active_node)

    def canonical_correction(self, f, rank):
        '''
        Rank-r canonical correction.

        Parameters
        ----------
        f : tensap.FunctionalTensor or None
            The current approximation.
        rank : int
            The rank of the canonical correction.

        Raises
        ------
        NotImplementedError
            If the method is not implemented.

        Returns
        -------
        f : tensap.FunctionalTensor
            The corrected approximation.

        '''
        if isinstance(f, tensap.FunctionalTensor):
            fx = f.tensor.tensor_matrix_product_eval_diag(self.bases_eval).data
        elif f is None:
            fx = 0
        else:
            raise NotImplementedError('Not implemented.')

        solver = deepcopy(self)
        if isinstance(solver.training_data, list) and \
                len(solver.training_data) == 2:
            solver.training_data[1] -= fx
        elif isinstance(solver.loss_function, tensap.DensityL2LossFunction):
            solver.training_data = [solver.training_data, f]

        f_add = solver.canonical_initialization(rank)
        if isinstance(f_add, tensap.FunctionalTensor):
            f_add = f_add.tensor
        if f is not None:
            f = f.tensor + f_add
        else:
            f = f_add
        return f

    def rank_one_correction(self, f):
        '''
        Rank one correction.

        Parameters
        ----------
        f : tensap.FunctionalTensor or None
            The current approximation.

        Raises
        ------
        NotImplementedError
            If the method is not implemented.

        Returns
        -------
        f : tensap.FunctionalTensor
            The corrected approximation.

        '''
        if isinstance(f, tensap.FunctionalTensor):
            fx = f.tensor.tensor_matrix_product_eval_diag(self.bases_eval).data
        elif f is None:
            fx = 0
        else:
            raise NotImplementedError('Not implemented.')

        solver = deepcopy(self)
        solver.model_selection = False
        if isinstance(solver.training_data, list) and \
                len(solver.training_data) == 2:
            solver.training_data[1] -= fx
        elif isinstance(solver.loss_function, tensap.DensityL2LossFunction):
            solver.training_data = [solver.training_data, f]
        solver.rank_adaptation = False
        solver.tree_adaptation = False
        solver.rank = 1
        solver.display = False
        solver.alternating_minimization_parameters['display'] = False
        solver.initialization_type = 'ones'
        solver.alternating_minimization_parameters['max_iterations'] = 1

        f_add = solver.solve()[0]
        if isinstance(f_add, tensap.FunctionalTensor):
            f_add = f_add.tensor
        if f is not None:
            f = f.tensor + f_add
        else:
            f = f_add
        return f

    def create_basis_adaptation_path(self, ranks, alpha):
        '''
        Creation of the basis adaptation path.

        Parameters
        ----------
        rank : list or numpy.ndarray
            The alpha-ranks of the current approximation.
        alpha : int
            The current node.

        Returns
        -------
        path : numpy.ndarray
            The basis adaptation path.

        '''
        tree = self.tree
        ranks = np.array(ranks)
        ranks[tree.root-1] = 1
        if tree.is_leaf[alpha-1]:
            path_alpha = self.bases_adaptation_path[
                np.nonzero(tree.dim2ind == alpha)[0][0]].astype(bool)
            ranks = ranks[alpha-1]
            path = path_alpha[np.newaxis, :, np.newaxis, :]
            path = np.tile(path, [1, 1, ranks, 2])
            path = np.reshape(path, [path.shape[1]*ranks, -1], order='F')
        else:
            assert not self.linear_model_learning_parameters[
                'basis_adaptation_internal_nodes'], \
                'Basis adaptation for internal nodes is not implemented.'
            ch = tree.children(alpha)
            if np.all(self.is_active_node[ch-1]):
                ch_a = ch[self.is_active_node[ch-1]]
                path = np.full(np.prod(ranks[ch_a-1]) * ranks[alpha-1], True)
            else:
                path_alpha = []
                for nod in ch:
                    if self.is_active_node[nod-1]:
                        path_alpha.append(np.full((ranks[nod-1], 1), True))
                    else:
                        ind = np.nonzero(nod == tree.dim2ind)[0][0]
                        path_alpha.append(
                            self.bases_adaptation_path[ind].astype(bool))
                path = path_alpha[-1]
                for ind in np.arange(len(path_alpha)-2, -1, -1):
                    path = np.kron(path, path_alpha[ind])
                path = np.tile(path, [ranks[alpha-1], 1]).astype(bool)
        return path

# %% Rank adaptation solver methods
    def local_solver(self):
        s_local = deepcopy(self)
        s_local.display = False
        s_local.rank_adaptation = False
        s_local.store_iterates = False
        s_local.test_error = False
        s_local.model_selection = False
        return s_local

    def new_rank_selection(self, f):
        if self.rank_adaptation_options['rank_one_correction']:
            s_local = deepcopy(self)
            ranks_add = np.ones(f.tensor.tree.nb_nodes, dtype=int)
            ranks_add[f.tensor.tree.root-1] = 0
            ranks_add[f.tensor.non_active_nodes-1] = 0
            s_local.rank = TreeBasedTensorLearning.make_ranks_admissible(
                f.tensor, f.tensor.ranks + ranks_add)[0]
            s_local.initialization_type = 'initial_guess'
            tr = tensap.Truncator(tolerance=0, max_rank=s_local.rank)
            s_local.initial_guess = tr.truncate(self.rank_one_correction(f))
            s_local.alternating_minimization_parameters['max_iterations'] = 10
            s_local.model_selection = False
            s_local.rank_adaptation = False
            s_local.display = False
            s_local.alternating_minimization_parameters['display'] = False
            tensor_for_selection = s_local.solve()[0].tensor
        else:
            tensor_for_selection = f.tensor

        sin_val = tensor_for_selection.singular_values()

        # Remove from the rank adaptation candidates: the inactive nodes, the
        # root, the leaf nodes with a rank equal to the dimension of the basis
        # associated to it, and the nodes for  which the smallest singular
        # value is almost zero.
        sin_val = np.array([np.nan if x is None else x for x in sin_val])
        sin_val[f.tensor.tree.root-1] = np.nan
        dim2ind = np.intersect1d(f.tensor.tree.dim2ind, f.tensor.active_nodes)
        ind = [len(set(x.size)) == 1 for x in f.tensor.tensors[dim2ind-1]]
        sin_val[dim2ind[ind]-1] = np.nan
        sin_val[s_local.rank != tensor_for_selection.ranks] = np.nan

        sin_val_min = np.array([np.min(x) for x in sin_val])
        sin_val_min[[x / tensor_for_selection.norm() < np.finfo(float).eps if
                     not np.isnan(x) else False for x in sin_val_min]] = np.nan

        # Remove nodes that cannot be enriched because their rank is equal to
        # the product of the ranks of their children, and their children cannot
        # be enriched themselves.
        tree = tensor_for_selection.tree
        rank = f.tensor.ranks
        desc = np.setdiff1d(np.arange(1, tree.nb_nodes+1),
                            np.nonzero(tree.is_leaf)[0]+1)
        cannot_be_increased = np.full(tree.nb_nodes, False)
        cannot_be_increased[tree.root-1] = True
        cannot_be_increased[tree.is_leaf] = np.isnan(sin_val_min[tree.is_leaf])
        for level in np.arange(np.max(tree.level), 0, -1):
            nod_lvl = np.intersect1d(tree.nodes_with_level(level), desc)
            for nod in nod_lvl:
                ch = tree.children(nod)
                if np.all(cannot_be_increased[ch-1]) and \
                        rank[nod-1] == np.prod(rank[ch-1]):
                    cannot_be_increased[nod-1] = True
        cannot_be_increased_nodes = tree.nodes_indices[cannot_be_increased]
        for level in np.arange(1, np.max(tree.level)):
            nod_lvl = np.intersect1d(tree.nodes_with_level(level),
                                     cannot_be_increased_nodes)
            for nod in nod_lvl:
                pa = tree.parent(nod)
                ind = np.setdiff1d(tree.children(pa), nod)
                ind = np.concatenate(([pa], ind))
                if np.all(cannot_be_increased[ind-1]) and \
                        rank[nod-1] == np.prod(rank[ind-1]):
                    cannot_be_increased[nod-1] = True
        sin_val_min[cannot_be_increased] = np.nan

        if np.all(np.isnan(sin_val_min)):
            enriched_nodes = np.array([], dtype=int)
        else:
            theta = self.rank_adaptation_options['theta'] * \
                np.nanmax(sin_val_min)
            enriched_nodes = np.nonzero([not np.isnan(x) and x >= theta for
                                         x in sin_val_min])[0] + 1

        new_rank = np.array(f.tensor.ranks)
        new_rank[enriched_nodes-1] += 1

        if not f.tensor.is_admissible_rank(new_rank):
            # Add to the already enriched nodes nodes one by one in decreasing
            # order of singular value until the rank is admissible.
            enriched_nodes_theta = np.array(enriched_nodes)
            rank_theta = np.array(new_rank)
            sin_val_min[enriched_nodes_theta-1] = np.nan
            sin_val_min_sorted = TreeBasedTensorLearning.unique_tol(
                sin_val_min, 1e-2)
            sin_val_min_sorted = np.flip(sin_val_min_sorted)
            sin_val_min_sorted = sin_val_min_sorted[
                np.isfinite(sin_val_min_sorted)]

            for sv in sin_val_min_sorted:
                new_rank = np.array(rank_theta)
                ind = [x >= sv if not np.isnan(x) else False for
                       x in sin_val_min]
                new_rank[ind] += 1
                if f.tensor.is_admissible_rank(new_rank):
                    enriched_nodes = np.concatenate((enriched_nodes_theta,
                                                     np.nonzero(ind)[0]+1))
                    break
            if not f.tensor.is_admissible_rank(new_rank):
                new_rank = f.tensor.ranks
                enriched_nodes = np.array([])

        return new_rank, enriched_nodes, tensor_for_selection

    def initial_guess_new_rank(self, s_local, f, new_rank):
        s_local.initialization_type = 'initial_guess'
        if not np.all(f.ranks == new_rank):
            tr = tensap.Truncator(tolerance=0, max_rank=new_rank)
            s_local.initial_guess = tr.truncate(f)
        else:
            s_local.initial_guess = f
        return s_local

    def adaptation_display(self, f, enriched_nodes):
        print('\tEnriched nodes: [%s]\n\tRanks = [%s]' %
              (', '.join(map(str, enriched_nodes)),
               ', '.join(map(str, f.tensor.ranks))))

    def adapt_tree(self, f, cv_error, test_error, output, *args):
        if not self.tree_adaptation:
            return self, f, output

        output['adapted_tree'] = False

        if np.any(f.tensor.ranks[f.tensor.active_nodes-1] == 0):
            print('Some ranks equal to 0, disabling tree adaptation for ' +
                  'this step.')
            return self, f, output
        if self.tree_adaptation_options['tolerance'] is not None:
            adapt_tree_error = self.tree_adaptation_options['tolerance']
        elif self.loss_function.error_type == 'relative':
            if test_error is None or test_error == 0:
                adapt_tree_error = cv_error
            elif cv_error is None or test_error != 0:
                adapt_tree_error = test_error
        else:
            print('Must provide a tolerance for the tree adaptation in ' +
                  'the treeAdaptationOptions property. Disabling tree ' +
                  'adaptation.')
            self.tree_adaptation = False
            return self, f, output

        f_perm = f.tensor.optimize_dimension_tree(
            adapt_tree_error, self.tree_adaptation_options['max_iterations'])
        if f_perm.storage() < f.tensor.storage():
            f.tensor = f_perm
            self.tree = f.tensor.tree
            self.is_active_node = f.tensor.is_active_node
            output['adapted_tree'] = True
            if self.display:
                print('\tTree adaptation:\n\t\tRanks after permutation ' +
                      '= [%s]' % ', '.join(map(str, f.tensor.ranks)))
        return self, f, output

# %% Inner rank adaptation solver
    def _solve_dmrg_rank_adaptation(self):
        if 'max_rank' not in self.rank_adaptation_options:
            self.rank_adaptation_options['max_rank'] = 100
        if 'post_alternating_minimization' not in self.rank_adaptation_options:
            self.rank_adaptation_options['post_alternating_minimization'] = \
                False
        if self.rank_adaptation_options['type'] == 'dmrg_low_rank' and \
                'model_selection_type' not in self.rank_adaptation_options:
            self.rank_adaptation_options['model_selection_type'] = 'cv_error'

        if self.display:
            self.alternating_minimization_parameters['display'] = True

        output = {'flag': 0}

        self, f = self.initialize()
        f = tensap.FunctionalTensor(f, self.bases_eval)

        # Exploration strategy of the tree by decreasing level
        tree = f.tensor.tree
        exploration_strategy = np.zeros(self._number_of_parameters, dtype=int)
        active_nodes = f.tensor.active_nodes
        rep = 0
        for level in np.arange(np.max(tree.level), -1, -1):
            nodes = np.intersect1d(tree.nodes_with_level(level), active_nodes)
            exploration_strategy[rep:rep+len(nodes)] = nodes
            rep += len(nodes)
        self._exploration_strategy = np.setdiff1d(exploration_strategy,
                                                  tree.root,
                                                  assume_unique=True)

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

            tree = f.tensor.tree
            for alpha in alpha_list:
                if self.linear_model_learning[alpha-1].basis_adaptation:
                    if np.isin(alpha, tree.internal_nodes):
                        if self.linear_model_learning_parameters[
                                'basis_adaptation_internal_nodes']:
                            tr = tensap.Truncator(
                                tolerance=np.finfo(float).eps,
                                max_rank=np.max(f.tensor.ranks))
                            f.tensor = tr.hsvd(f.tensor)
                        elif np.all(f.tensor.is_active_node[
                                tree.children(alpha)-1]):
                            self.linear_model_learning[alpha-1].\
                                basis_adaptation = False
                    f.tensor = f.tensor.orth_at_node(tree.parent(alpha))
                    self.tree = tree
                    self.is_active_node = f.tensor.is_active_node

                    if self.rank_adaptation_options['type'] == 'dmrg':
                        self.linear_model_learning[alpha-1].\
                            basis_adaptation_path = \
                            self.create_basis_adaptation_path_dmrg(
                                f.tensor.ranks, alpha)
                    elif self.rank_adaptation_options['type'] == \
                            'dmrg_low_rank':
                        self.linear_model_learning[alpha-1].\
                            basis_adaptation_path = self.\
                            create_basis_adaptation_path_dmrg_low_rank(
                                f.tensor.ranks, alpha)
                    else:
                        raise ValueError('Wrong rank adaptation type.')
                else:
                    f.tensor = f.tensor.orth_at_node(tree.parent(alpha))

                grad = f.parameter_gradient_eval_dmrg(
                    alpha, dmrg_type=self.rank_adaptation_options['type'])

                if isinstance(self.loss_function,
                              tensap.DensityL2LossFunction):
                    if isinstance(self.training_data, list) and \
                            len(self.training_data) == 2:
                        y = self.training_data[1]
                        if isinstance(y, tensap.FunctionalTensor):
                            y = y.tensor
                        y = y.orth()
                        if tree.is_leaf[alpha-1]:
                            a = deepcopy(y)
                            for nod in tree.internal_nodes:
                                a.tensors[nod-1] = tensap.FullTensor(
                                    y.tensors[nod-1].data *
                                    f.tensor.tensors[nod-1].data,
                                    shape=y.tensors[nod-1].shape)
                            ind = np.setdiff1d(
                                range(self.order),
                                np.nonzero(tree.dim2ind == alpha)[0])
                            C = [x.data for
                                 x in f.tensor.tensors[tree.dim2ind-1]]
                            b = a.tensor_vector_product([C[x] for x in ind],
                                                        ind)
                            b = b.tensors[0].data
                        else:
                            b = f.tensor.dot(y) / \
                                f.tensor.tensors[alpha-1].data
                    else:
                        b = []
                elif isinstance(self.training_data, list) and \
                        len(self.training_data) == 2:
                    b = self.training_data[1]
                    self.linear_model_learning[alpha-1].shared_coefficients = \
                        alpha != tree.root

                gamma = tree.parent(alpha)
                ind = np.setdiff1d(
                    np.arange(1, f.tensor.tensors[gamma-1].order+1),
                    tree.child_number(alpha))

                if self.rank_adaptation_options['type'] == 'dmrg':
                    A = np.reshape(grad.data, (grad.shape[0], -1), order='F')

                    self.linear_model_learning[alpha-1].training_data = [None,
                                                                         b]
                    self.linear_model_learning[alpha-1].basis = None
                    self.linear_model_learning[alpha-1].basis_eval = A

                    C, output_tmp = \
                        self.linear_model_learning[alpha-1].solve()

                    if C is None or np.count_nonzero(C) == 0 or \
                            not np.all(np.isfinite(C)):
                        print('Empty, zero or NaN solution, returning to ' +
                              'the previous iteration.')
                        output['flag'] = -2
                        output['error'] = np.inf
                        break

                    sz_1 = np.prod(f.tensor.tensors[alpha-1].shape[:-1])
                    sz_2 = np.prod(f.tensor.tensors[gamma-1].shape[ind-1])
                    C = tensap.FullTensor(C, 2, [sz_1, sz_2])

                    tr = tensap.Truncator()
                    tr.tolerance = self.tolerance['on_error'] / \
                        np.sqrt(np.count_nonzero(f.tensor.is_active_node)-1)
                    tr.max_rank = self.rank_adaptation_options['max_rank']
                    C = tr.truncate(C)
                    rank = [np.shape(C.space[0])[1]]

                    sz_1 = np.concatenate(
                        (f.tensor.tensors[alpha-1].shape[:-1], rank))
                    sz_2 = np.concatenate(
                        (f.tensor.tensors[gamma-1].shape[ind-1], rank))

                    a_alpha = np.reshape(C.space[0], sz_1, order='F')
                    a_gamma = np.reshape(C.space[1]*np.tile(
                        np.reshape(C.core.data, [1, -1]),
                        (np.shape(C.space[1])[0], 1)), sz_2, order='F')
                    perm = np.concatenate((ind, [tree.child_number(alpha)]))
                    a_gamma = np.transpose(a_gamma, np.argsort(perm-1))
                    sz_2[perm-1] = np.array(sz_2)
                elif self.rank_adaptation_options['type'] == 'dmrg_low_rank':
                    A = [np.reshape(x.data, [x.shape[0], -1], order='F') for
                         x in grad]

                    s_local = TreeBasedTensorLearning(
                        tensap.DimensionTree.trivial(2), [True, True, False],
                        self.loss_function)
                    s_local.rank_adaptation = True
                    s_local.tolerance['on_error'] = \
                        self.tolerance['on_error'] / \
                        np.sqrt(np.count_nonzero(f.tensor.is_active_node)-1)
                    s_local.tolerance['on_stagnation'] = \
                        self.tolerance['on_stagnation']
                    s_local.rank_adaptation_options['max_iterations'] = \
                        self.rank_adaptation_options['max_rank']
                    s_local.alternating_minimization_parameters = \
                        deepcopy(self.alternating_minimization_parameters)
                    s_local.alternating_minimization_parameters['display'] = \
                        False
                    s_local.store_iterates = True
                    s_local.test_error = False
                    s_local.error_estimation = True
                    s_local.display = False
                    s_local.order = 2
                    s_local.linear_model_learning = \
                        self.linear_model_learning[alpha-1]
                    s_local._warnings['orthonormality_warning_display'] = False
                    s_local._warnings['empty_bases_warning_display'] = False
                    s_local.bases_adaptation_path = \
                        self.linear_model_learning[alpha-1].\
                        basis_adaptation_path
                    s_local.training_data = self.training_data
                    s_local.bases_eval = A
                    s_local.model_selection = True
                    s_local.model_selection_options['type'] = \
                        self.rank_adaptation_options['model_selection_type']
                    C, output_tmp = s_local.solve()

                    rank = [C.tensor.ranks[1]]
                    sz_1 = np.concatenate(
                        (f.tensor.tensors[alpha-1].shape[:-1], rank))
                    sz_2 = np.concatenate(
                        (f.tensor.tensors[gamma-1].shape[ind-1], rank))

                    a_alpha = C.tensor.tensors[1].reshape(sz_1)
                    a_gamma = C.tensor.tensors[0].transpose([1, 0]).reshape(
                        sz_2)

                    perm = np.concatenate((ind, [tree.child_number(alpha)]))
                    a_gamma = a_gamma.itranspose(perm-1)
                    sz_2[perm-1] = np.array(sz_2)

                    a_alpha = a_alpha.data
                    a_gamma = a_gamma.data

                    if (a_alpha is None or np.count_nonzero(a_alpha) == 0 or
                            not np.all(np.isfinite(a_alpha))) or \
                        (a_gamma is None or np.count_nonzero(a_gamma) == 0 or
                            not np.all(np.isfinite(a_gamma))):
                        print('Empty, zero or NaN solution, returning to ' +
                              'the previous iteration.')
                        output['flag'] = -2
                        output['error'] = np.inf
                        break
                else:
                    raise ValueError('Wrong rank adaptation type.')
                f.tensor.tensors[alpha-1] = tensap.FullTensor(a_alpha,
                                                              np.size(sz_1),
                                                              sz_1)
                f.tensor.tensors[gamma-1] = tensap.FullTensor(a_gamma,
                                                              np.size(sz_2),
                                                              sz_2)

            if self.rank_adaptation_options['post_alternating_minimization']:
                print('\t\tPost alternating minimization.')
                s_local = TreeBasedTensorLearning(self.tree,
                                                  self.is_active_node,
                                                  self.loss_function)
                s_local.bases_eval = self.bases_eval
                s_local.bases_adaptation_path = self.bases_adaptation_path
                s_local.training_data = self.training_data
                s_local.test_error = self.test_error
                s_local.test_data = self.test_data
                s_local.bases_eval_test = self.bases_eval_test
                s_local.model_selection = False
                s_local.rank_adaptation = False
                s_local.store_iterates = False
                s_local.rank = f.tensor.ranks
                s_local.initialization_type = 'initial_guess'
                s_local.initial_guess = f.tensor
                s_local.tolerance['on_error'] = self.tolerance['on_error'] / \
                    np.sqrt(np.count_nonzero(f.tensor.is_active_node)-1)
                s_local.tolerance['on_stagnation'] = \
                    self.tolerance['on_stagnation']
                s_local.alternating_minimization_parameters = \
                    deepcopy(self.alternating_minimization_parameters)
                s_local.alternating_minimization_parameters['display'] = False
                s_local.error_estimation = True
                s_local.display = False
                s_local.linear_model_learning = \
                    [x for x in self.linear_model_learning if x is not None]
                s_local._warnings['orthonormality_warning_display'] = False
                s_local._warnings['empty_bases_warning_display'] = False
                f, output_tmp = s_local.solve()

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
                print('\tAlt. min. iteration %s/%i: stagnation = %2.5e' %
                      (str(iteration).
                       zfill(len(str(self.alternating_minimization_parameters
                                     ['max_iterations']-1))),
                       self.alternating_minimization_parameters
                       ['max_iterations']-1,
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

            if self.tree_adaptation and iteration > 0:
                C_old = f.tensor.storage()
                self, f, output = self.adapt_tree(f, output['error'],
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
                        if self.display:
                            print('\t\tTest error after permutation ' +
                                  '= %2.5e' % self.loss_function.test_error(
                                      f_eval_test, self.test_data))

                    if self.alternating_minimization_parameters['display']:
                        print('')

        if isinstance(self.bases, tensap.FunctionalBases):
            f = tensap.FunctionalTensor(f.tensor, self.bases)
        output['iter'] = iteration

        if 'adapted_tree' in output:
            del output['adapted_tree']

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

    def create_basis_adaptation_path_dmrg(self, ranks, alpha):
        tree = self.tree
        ranks = np.array(ranks)
        ranks[tree.root-1] = 1
        gamma = tree.parent(alpha)

        p_alpha = self.create_basis_adaptation_path(ranks, alpha)
        if p_alpha.ndim == 1:
            p_alpha = np.reshape(p_alpha, [-1, 1])
        p_alpha = p_alpha[:int(p_alpha.shape[0]/ranks[alpha-1]), :]

        ch = tree.children(gamma)
        p_gamma = [None]*len(ch)
        for nod in np.setdiff1d(ch, alpha):
            ind = tree.child_number(nod) < tree.child_number(alpha)
            if self.is_active_node[nod-1]:
                p_gamma[tree.child_number(nod)-1+ind] = \
                    np.full((ranks[nod-1], 1), True)
            else:
                rep = np.nonzero(nod == tree.dim2ind)[0][0]
                p_gamma[tree.child_number(nod)-1+ind] = \
                    self.bases_adaptation_path[rep].astype(bool)
        p_gamma[0] = p_alpha

        r_gamma = ranks[gamma-1]
        p = p_gamma[-1]
        for i in np.arange(len(p_gamma)-2, -1, -1):
            p = np.kron(p, p_gamma[i])
        return np.tile(p, (r_gamma, 1)).astype(bool)

    def create_basis_adaptation_path_dmrg_low_rank(self, ranks, alpha):
        tree = self.tree
        ranks = np.array(ranks)
        ranks[tree.root-1] = 1
        gamma = tree.parent(alpha)

        p_alpha = self.create_basis_adaptation_path(ranks, alpha)
        if p_alpha.ndim == 1:
            p_alpha = np.reshape(p_alpha, [-1, 1])
        p_alpha = p_alpha[:int(p_alpha.shape[0]/ranks[alpha-1]), :]

        ch = np.setdiff1d(tree.children(gamma), alpha)
        p_gamma = [None]*len(ch)
        for nod in ch:
            ind = tree.child_number(nod) > tree.child_number(alpha)
            if self.is_active_node[nod-1]:
                p_gamma[tree.child_number(nod)-1-ind] = \
                    np.full((ranks[nod-1], 1), True)
            else:
                rep = np.nonzero(nod == tree.dim2ind)[0][0]
                p_gamma[tree.child_number(nod)-1-ind] = \
                    self.bases_adaptation_path[rep].astype(bool)

        r_gamma = ranks[gamma-1]
        p = p_gamma[-1]
        for i in np.arange(len(p_gamma)-2, -1, -1):
            p = np.kron(p, p_gamma[i])
        p = np.tile(p, (r_gamma, 1)).astype(bool)

        return [p_alpha, p]

# %% Static methods
    @staticmethod
    def tensor_train(order, *args):
        '''
        Call of the constructor of the class TreeBasedTensorLearning, with a
        tree and active nodes corresponding to the Tensor-Train format in
        dimension order.

        See also TreeBasedTensorLearning.

        Parameters
        ----------
        order : int
            The order of the tensor.
        *args : tuple
            Additional parameters (see the constructor of
            TreeBasedTensorLearning).

        Returns
        -------
        TreeBasedTensorLearning
            The solver with a tree and active nodes associated with the
            Tensor-Train format.

        '''
        tree = tensap.DimensionTree.linear(order)
        is_active_node = np.full(tree.nb_nodes, True)
        is_active_node[tree.dim2ind[1:]-1] = False
        return TreeBasedTensorLearning(tree, is_active_node, *args)

    @staticmethod
    def tensor_train_tucker(order, *args):
        '''
        Call of the constructor of the class TreeBasedTensorLearning, with a
        tree and active nodes corresponding to the Tensor-Train Tucker format
        in dimension order.

        See also TreeBasedTensorLearning.

        Parameters
        ----------
        order : int
            The order of the tensor.
        *args : tuple
            Additional parameters (see the constructor of
            TreeBasedTensorLearning).

        Returns
        -------
        TreeBasedTensorLearning
            The solver with a tree and active nodes associated with the
            Tensor-Train Tucker format.

        '''
        tree = tensap.DimensionTree.linear(order)
        is_active_node = np.full(tree.nb_nodes, True)
        return TreeBasedTensorLearning(tree, is_active_node, *args)

    @staticmethod
    def enriched_edges_to_ranks_random(f, new_rank):
        '''
        Enrichment of the ranks of specified edges of the tensor f using random
        additions for each child / parent couple of the enriched edges.

        Parameters
        ----------
        f : tensap.TreeBasedTensor
            The tree-based tensor to enrich.
        new_rank : list or numpy.ndarray
            The new tree-based rank.

        Returns
        -------
        f : tensap.TreeBasedTensor
            The enriched tree-based tensor.

        '''
        f.is_orth = False
        tree = f.tree
        enriched_dims = np.nonzero(new_rank > f.ranks)[0]

        for level in np.arange(1, np.max(tree.level)+1):
            nod_lvl = np.intersect1d(tree.nodes_with_level(level),
                                     enriched_dims)
            for alpha in nod_lvl:
                gamma = tree.parent(alpha)
                rank = new_rank[alpha-1] - f.ranks[alpha-1]

                A = np.reshape(f.tensors[alpha-1].data,
                               [-1, f.ranks[alpha-1]], order='F')
                A = np.hstack((A, np.tile(np.reshape(A[:, -1], [-1, 1]),
                                          [1, rank]) *
                               (1+np.random.randn(A.shape[0], rank))))
                A[:, -1-rank-1:] /= np.sqrt(np.sum(A[:, -1-rank-1:]**2, 0))
                shape = np.array(f.tensors[alpha-1].shape)
                shape[-1] += rank
                f.tensors[alpha-1].data = np.reshape(A, shape, order='F')

                ch = f.tree.child_number(alpha)-1
                ind = np.setdiff1d(range(f.tensors[gamma-1].order), ch)
                ind = np.concatenate((ind, [ch]))
                A = np.transpose(f.tensors[gamma-1].matricize(ch).data)
                A = np.hstack((A, np.tile(np.reshape(A[:, -1], [-1, 1]),
                                          [1, rank]) *
                               (1+np.random.randn(A.shape[0], rank))))
                A[:, -1-rank-1:] /= np.sqrt(np.sum(A[:, -1-rank-1:]**2, 0))
                shape = np.array(f.tensors[gamma-1].shape)
                shape[ch] += rank
                A = np.reshape(A, shape[ind], order='F')
                f.tensors[gamma-1].data = np.transpose(A, np.argsort(ind))

                f = f.update_attributes()
        return f

    @staticmethod
    def make_ranks_admissible(f, rank):
        '''
        Adjustment of the ranks to make the associated tree-based tensor f
        rank-admissible, by enriching new edges associated with nodes of the
        tree until all the rank admissibility conditions are met.

        Parameters
        ----------
        f : tensap.TreeBasedTensor
            The current approximation in tree-based tensor format.
        rank : numpy.ndarray
            The proposed tree-based rank.

        Returns
        -------
        rank : numpy.ndarray
            The admissible tree-based rank (if possible).
        d : numpy.ndarray
            The enriched nodes.

        '''
        # Do not increase the ranks of leaf nodes with rank equal to the
        # dimension of the approximation space.
        nodes = f.active_nodes
        ind = [f.tree.is_leaf[x] and rank[x] > f.tensors[x].shape[0] for
               x in nodes-1]
        rank[nodes[ind]-1] = [x.shape[0] for x in f.tensors[nodes[ind]-1]]
        rank[np.logical_not(f.is_active_node)] = 0

        delta = rank - f.ranks
        if f.is_admissible_rank(f.ranks + delta):
            rank = f.ranks + delta
            d = np.nonzero(delta)[0] + 1
            return rank, d

        ind = np.nonzero(delta)[0]
        for i in np.arange(1, np.count_nonzero(delta)+1):
            pos = list(combinations(np.arange(np.count_nonzero(delta)), i))
            shuffle(pos)
            for pos_loc in pos:
                delta_loc = np.array(delta)
                delta_loc[ind[list(pos_loc)]] = 0
                if f.is_admissible_rank(f.ranks + delta_loc):
                    rank = f.ranks + delta_loc
                    d = np.nonzero(delta_loc)[0] + 1
                    return rank, d
        return rank, []

    @staticmethod
    def unique_tol(inp, tol):
        '''
        Unique values within tolerance, with sorted output.

        Parameters
        ----------
        inp : list of numpy.ndarray
            The values to be checked.
        tol : float
            The tolerance defining the uniqueness.

        Returns
        -------
        bout : numpy.ndarray
            The unique values of inp, within the tolerance tol.

        '''
        inp = np.sort(inp)
        out = [inp[0]]

        for ind in np.arange(1, inp.size):
            if np.abs(out[-1] - inp[ind]) / out[-1] > tol:
                out = np.concatenate((out, [inp[ind]]))
        return out
