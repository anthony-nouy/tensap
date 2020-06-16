'''
Module tree_based_tensor.

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

from functools import reduce
import copy
import numpy as np
import tensap


class TreeBasedTensor:
    '''
    Class TreeBasedTensor: algebraic tensors in tree-based tensor format.

    References:
    - Nouy, A. (2017). Low-rank methods for high-dimensional approximation
    and model order reduction. Model reduction and approximation, P. Benner,
    A. Cohen, M. Ohlberger, and K. Willcox, eds., SIAM, Philadelphia,
    PA, 171-226.
    - Falco, A., Hackbusch, W., & Nouy, A. (2018). Tree-based tensor formats.
    SeMA Journal, 1-15
    - Grelier, E., Nouy, A., & Chevreuil, M. (2018). Learning with tree-based
    tensor formats. arXiv preprint arXiv:1811.04455
    - Nouy, A. (2019). Higher-order principal component analysis for the
    approximation of tensors in tree-based low-rank formats. Numerische
    Mathematik, 141(3), 743-789

    Attributes
    ----------
    tensors : numpy.ndarray
        Parameters of the representation.
    ranks : numpy.ndarray
        Tree-based rank.
    order : int
        Order of the tensor.
    shape : numpy.ndarray
        Shape of the tensor.
    tree : tensap.FullTensor
        Dimension tree.
    is_orth : bool
        True if the representation of the tensor is orthogonal.
    is_active_node : numpy.ndarray
         Logical array indicating if the nodes are active.
    orth_node : int
        Node with respect to which the representation is orthogonalized (0 is
        the root node)

    '''

    # Override numpy's operations with reversed operands
    __array_priority__ = 1

    def __init__(self, cores, tree=None):
        '''
        Constructor for the class TreeBasedTensor.

        Parameters
        ----------
        cores : TreeBasedTensor or list
            TreeBasedTensor to be copied or list of tensors (parameters
            associated with the nodes of the tree).
        tree : tensap.FullTensor, optional
            Dimension tree of the tree-based tensor. The default is None.

        Raises
        ------
        NotImplementedError
            If the constructor is not implemented for the provided arguments.

        Returns
        -------
        None.

        '''
        if (tree is None) and isinstance(cores, TreeBasedTensor):
            # Create a copy of the TreeBasedTensor
            self.tensors = np.array(cores.tensors)
            self.ranks = np.array(cores.ranks)
            self.shape = np.array(cores.shape)
            self.order = np.array(cores.order)
            self.tree = copy.deepcopy(cores.tree)
            self.is_orth = np.array(cores.is_orth)
            self.is_active_node = np.array(cores.is_active_node)
            self.orth_node = np.array(cores.orth_node)
        elif isinstance(cores, (list, np.ndarray)) and \
                isinstance(tree, tensap.DimensionTree):
            self.tree = copy.deepcopy(tree)
            self.tensors = np.array([tensap.FullTensor(x) for x in cores])
            self.order = self.tree.dim2ind.size
            self.is_orth = False
            self.orth_node = None
            self.update_attributes()
        else:
            raise NotImplementedError('Constructor not implemented for the '
                                      'provided arguments.')

    @property
    def ndim(self):
        '''
        Compute the order of the tensor. Equivalent to self.order.

        Returns
        -------
        int
            The order of the tensor.

        '''
        return self.order

    @property
    def representation_rank(self):
        '''
        Return the representation tree-based rank of the tensor.

        Corresponds to self.ranks.

        Returns
        -------
        numpy.ndarray
            The representation tree-based rank of the tensor.

        '''
        return self.ranks

    @property
    def rank(self):
        '''
        Return the tree-based rank of the tensor (computed by SVD).

        Returns
        -------
        numpy.ndarray
            The tree-based rank of the tensor.

        '''
        return np.array([np.count_nonzero(s) for s in self.singular_values()])

    @property
    def active_nodes(self):
        '''
        Return the list of active nodes.

        Returns
        -------
        numpy.ndarray
            The list of active nodes.

        '''
        return self.tree.nodes_indices[self.is_active_node]

    @property
    def non_active_nodes(self):
        '''
        Return the list of non active nodes.

        Returns
        -------
        numpy.ndarray
            The list of non active nodes.

        '''
        return self.tree.nodes_indices[np.logical_not(self.is_active_node)]

    @property
    def active_dims(self):
        '''
        Return the list of active dimensions.

        Returns
        -------
        numpy.ndarray
            The list of active dimensions.

        '''
        return np.nonzero(np.isin(self.tree.dim2ind, self.active_nodes))[0]

    @property
    def non_active_dims(self):
        '''
        Return the list of non active dimensions.

        Returns
        -------
        numpy.ndarray
            The list of active dimensions.

        '''
        return tensap.fast_setdiff(np.arange(self.order), self.active_dims)

    def is_active_dim(self, dim):
        '''
        Return an array containing true if the given dimensions are active,
        false otherwise.

        Parameters
        ----------
        dim : integer
            The tested dimension.

        Returns
        -------
        result : numpy.ndarray or boolean
            Array containing true if the given dimensions are active, false
            otherwise.

        '''
        result = np.isin(self.tree.dim2ind[dim], self.active_nodes)
        if result.size == 1:
            result = result[0]
        return result

    def __repr__(self):
        return ('<TreeBasedTensor:{n}' +
                '{t}order = {},{n}' +
                '{t}ranks = {},{n}' +
                '{t}shape = {},{n}' +
                '{t}is_orth = {},{n}' +
                '{t}orth_node = {},{n}' +
                '{t}is_active_node = {}>').format(self.order,
                                                  self.ranks,
                                                  self.shape,
                                                  self.is_orth,
                                                  self.orth_node,
                                                  self.is_active_node,
                                                  n='\n', t='\t')

    # def __eq__(self, tensor2):
    #     return np.all(self.data == tensor2.data)

    def __add__(self, arg):
        tree = self.tree
        tensors = np.array(self.tensors)

        for nod in np.arange(1, tree.nb_nodes+1):
            if tree.is_leaf[nod-1] and self.is_active_node[nod-1]:
                tensors[nod-1] = \
                    tensors[nod-1].cat(arg.tensors[nod-1], 1)
            elif not tree.is_leaf[nod-1]:
                children = tree.children(nod)
                non_active_leaves = [a and not b for a, b in
                                     zip(tree.is_leaf[children-1],
                                         self.is_active_node[children-1])]
                if nod == tree.root:
                    ind = np.nonzero(np.logical_not(non_active_leaves))[0]
                    tensors[nod-1] = tensors[nod-1].cat(arg.tensors[nod-1],
                                                        ind)
                elif np.any(non_active_leaves):
                    other_dims = np.arange(children.size)
                    other_dims = other_dims[np.logical_not(non_active_leaves)]
                    if nod != tree.root:
                        other_dims = np.concatenate(
                            (other_dims, [tensors[nod-1].order-1]))
                    tensors[nod-1] = \
                        tensors[nod-1].cat(
                            arg.tensors[nod-1], other_dims)
                else:
                    tensors[nod-1] = \
                        tensors[nod-1].cat(arg.tensors[nod-1])
        return TreeBasedTensor(tensors, tree)

    def __radd__(self, arg):
        return self + arg

    def __sub__(self, arg):
        return self + (-arg)

    def __rsub__(self, arg):
        return -self + arg

    def __neg__(self):
        tensor = copy.deepcopy(self)
        tensor.tensors[tensor.tree.root] = -tensor.tensors[tensor.tree.root]
        return tensor

    def __mul__(self, arg):
        assert np.isscalar(arg), 'The second argument must be a scalar.'
        tensor = copy.deepcopy(self)
        tensor.tensors[tensor.tree.root-1] *= arg
        return tensor

    def __rmul__(self, arg):
        assert np.isscalar(arg) or isinstance(arg, np.ndarray), \
            'The second argument must be a scalar or a numpy.ndarray.'
        arg = np.atleast_2d(arg)
        tensor = copy.deepcopy(self)
        root = tensor.tree.root-1
        if arg.size == 1:
            tensor = tensor * arg[0, 0]
        elif arg.shape[1] == tensor.ranks[root] and arg.shape[1] > 1:
            tensor.tensors[root] = tensor.tensors[root].tensor_matrix_product(
                arg, tensor.tensors[root].order-1)
            tensor.ranks[root] = arg.shape[1]
        else:
            raise ValueError('Wrong shapes.')
        return tensor

    def __truediv__(self, arg):
        assert np.isscalar(arg), 'The second argument must be a scalar.'
        tensor = copy.deepcopy(self)
        tensor.tensors[tensor.tree.root-1] /= arg
        return tensor

    def hadamard_product(self, arg):
        '''
        Compute the Hadamard product of two tensors.

        Equivalent to self * arg.

        Parameters
        ----------
        arg : TreeBasedTensor
            The second tensor of the Hadamard product.

        Returns
        -------
        tensap.FullTensor
            The tensor resulting from the Hadamard product.

        '''
        return self * arg

    def is_admissible_rank(self, ranks=None, nargout=1):
        '''
        Check if a given tuple is an admissible tree-based rank.

        If no tree-based rank is provided, the tree-based rank of self is
        checked.

        Parameters
        ----------
        ranks : typle of numpy.ndarray, optional
            The tree-based rank to be checked. The default is None, indicating
            the tree-based rank of self.
        nargout : int, optional
            Indicates the number of expected outputs. The default is 1,
            indicating to return only the boolean characterizing if the
            tree-based rank is admissible.

        Returns
        -------
        is_admiss : bool
            True if the tree-based rank is admissible.
        ch_admiss : numpy.ndarray
            Array detailing the admissibility of the children ranks.

        '''
        if ranks is None:
            ranks = self.ranks
        else:
            ranks = np.atleast_1d(ranks)
        ranks = np.array(ranks)

        tree = self.tree
        ch_admiss = np.empty(tree.nb_nodes, dtype=object)

        # Do not take into account the root rank if it is greater than one.
        ranks[tree.root-1] = 1

        if np.any(ranks[self.non_active_nodes-1] != 0):
            is_admiss = False
            print('Inactive nodes must have an alpha-rank equal to 0.')
        else:
            is_admiss = True
            ranks[tensap.fast_intersect(tree.dim2ind,
                                        self.non_active_nodes)-1] =\
                self.shape[self.non_active_dims]
            for nod in tensap.fast_intersect(tree.internal_nodes,
                                             self.active_nodes):
                children = tree.children(nod)
                active_ch = tensap.fast_intersect(children, self.active_nodes)

                ch_admiss[nod-1] = [ranks[nod-1] <=
                                    np.prod(ranks[children-1])]
                for child in active_ch:
                    ch_no_nod = tensap.fast_setdiff(children, child)
                    ch_admiss[nod-1].append(
                        ranks[child-1] <= ranks[nod-1] *
                        np.prod(ranks[ch_no_nod-1]))
                is_admiss = is_admiss and np.all(ch_admiss[nod-1])

            for nod in tensap.fast_intersect(np.nonzero(tree.is_leaf)[0]+1,
                                             self.active_nodes):
                ch_admiss[nod-1] = ranks[nod-1] <= self.tensors[nod-1].shape[0]
                is_admiss = is_admiss and ch_admiss[nod-1]
        if nargout == 1:
            return is_admiss
        return is_admiss, ch_admiss

    def storage(self):
        '''
        Return the storage complexity of the TreeBasedTensor.

        Returns
        -------
        int
            The storage complexity of the TreeBasedTensor.

        '''
        return np.sum([x.storage() for x in self.tensors])

    def sparse_storage(self):
        '''
        Return the sparse storage complexity of the TreeBasedTensor.

        Returns
        -------
        int
            The sparse storage complexity of the TreeBasedTensor.

        '''
        return np.sum([x.sparse_storage() for x in self.tensors])

    def sparse_leaves_storage(self):
        '''
        Return the storage complexity of the TreeBasedTensor taking into
        account the sparsity in the leaves.

        Returns
        -------
        int
            The storage complexity of the TreeBasedTensor taking into account
            the sparsity in the leaves.

        '''
        return np.sum([x.storage() for x in
                       self.tensors[self.tree.internal_nodes-1]]) + \
            np.sum([x.sparse_storage() for x in
                    self.tensors[self.tree.dim2ind-1]])

    def full(self):
        '''
        Convert a TreeBasedTensor to a tensap.FullTensor

        Returns
        -------
        tensor : tensap.FullTensor
            A representation of the TreeBasedTensor as a tensap.FullTensor.

        '''
        tree = self.tree
        tensors = np.array(self.tensors)

        for level in np.arange(np.max(tree.level), -1, -1):
            nod_level = tensap.fast_setdiff(tree.nodes_with_level(level),
                                            tree.dim2ind)
            for nod in nod_level:
                children = tree.children(nod)
                dims = np.array([], dtype=int)
                for child in children:
                    if self.is_active_node[child-1]:
                        tensors[nod-1] = tensors[child-1].\
                            tensordot(tensors[nod-1],
                                      [tensors[child-1].order-1],
                                      [dims.size])
                        tensors[child-1] = None
                        dims = np.concatenate((tree.dims[child-1], dims))
                    else:
                        dims = np.concatenate((dims, tree.dims[child-1]))
                tree.dims[nod-1] = dims

        if self.ranks[tree.root-1] > 1:
            dims = np.concatenate((dims, [self.order]))
        tensor = tensors[tree.root-1]
        if tensor.order > 1:
            tensor = tensor.itranspose(dims)
        return tensor

    def numpy(self):
        '''
        Convert the TreeBasedTensor to a numpy.ndarray.

        Returns
        -------
        numpy.ndarray
            The TreeBasedTensor as a numpy.ndarray.

        '''
        return self.full().data.numpy()

    def sub_tensor(self, *indices):
        '''
        Extract a subtensor of the tensor.

        The result is a tensor s of shape
        len(indices[0]), ..., len(indices[self.order-1]),
        such that
        s(k1,...,kd) = x(indices[0][k1], ..., indices[self.order-1][kd]).

        Example: x.subTensor([1, 2], ':', [2, 5, 6]) returns a tensor with
        shape [2, self.shape[1], 3].



        Parameters
        ----------
        *indices : list
            The indices to extract in each dimension. ':' indicates all the
            indices.

        Returns
        -------
        TreeaBasedTensor
            The subtensor.

        '''
        assert len(indices) == self.order, 'Wrong number of input arguments.'
        tree = self.tree
        tensors = np.array(self.tensors)
        for dim in range(self.order):
            nod = tree.dim2ind[dim]
            if not self.is_active_node[nod-1]:
                parent = tree.parent(nod)
                child_nb = tree.child_number(nod)
                tensors[parent-1] = \
                    tensors[parent-1].eval_at_indices(indices[dim],
                                                      child_nb-1)
            else:
                tensors[nod-1] = \
                    tensors[nod-1].eval_at_indices(indices[dim], 0)
        return TreeBasedTensor(tensors, tree)

    def nodes_permutation_cost(self, alpha):
        '''
        Cost of the permutation of a given node alpha with the other nodes of
        the dimension tree.

        Parameters
        ----------
        alpha : int
            The node from which all the permutation costs are computed.

        Returns
        -------
        cost : numpy.ndarray
            The cost of permuting alpha with the other nodes of the tree.

        '''
        tree = self.tree
        ranks = self.ranks
        ranks[self.non_active_nodes-1] = self.shape[np.logical_not(
            self.is_active_node[tree.dim2ind-1])]

        cost = np.zeros(tree.nb_nodes)
        for beta in np.arange(1, tree.nb_nodes+1):
            asc_alpha = tree.ascendants(alpha)
            asc_beta = tree.ascendants(beta)
            if not (beta in asc_alpha or beta in tree.descendants(alpha) or
                    tree.parent(beta) == tree.parent(alpha)):
                common_asc = tensap.fast_intersect(asc_alpha, asc_beta)
                gamma = common_asc[tree.level[common_asc-1] ==
                                   np.max(tree.level[common_asc-1])]
                unique_asc = np.unique(np.concatenate((asc_alpha, asc_beta)))
                children = tensap.fast_setdiff(
                    tree.children(tensap.fast_setdiff(unique_asc,
                                                      tree.ascendants(gamma))),
                    unique_asc)
                cost[beta-1] = ranks[gamma-1] * np.prod(ranks[children-1])
        return cost

    def optimize_dimension_tree(self, tolerance, max_iter):
        '''
        Optimization over the set of trees to obtain a representation of the
        tensor with lower complexity.

        Parameters
        ----------
        tolerance : float
            The relative tolerance for the tree changes.
        max_iter : int
            The maximum number of tree changes.

        Returns
        -------
        tensor_star : tensap.TreeBasedTensor
            The tree-based tensor with optimized tree.

        '''
        max_iter = int(max_iter)
        nb_nodes = self.tree.nb_nodes
        tensor_star = self

        nodes = tensap.fast_setdiff(np.arange(1, nb_nodes+1), self.tree.root)

        proba_m = 1 / np.arange(1, self.order+1)**2

        c_star = self.storage()
        sigma_star = np.array([]).reshape(0, 2)
        m_perm = 0
        m_star = 0
        ind = True

        for _ in range(max_iter):
            m_current = np.random.choice(np.arange(1, self.order+1),
                                         p=proba_m/np.sum(proba_m))
            if m_perm < m_current or ind:
                m_perm = m_current
                ind = False
                tensor_sigma = copy.deepcopy(self)
                for i in range(m_star):
                    tensor_sigma = tensor_sigma.permute_nodes(
                        sigma_star[i, :], tolerance/(m_current+m_star))

            sigma = np.zeros([m_current, 2])
            tensor_current = copy.deepcopy(tensor_sigma)
            i = 0
            while i < m_current:
                proba_alpha = tensor_current.ranks[
                    tensor_current.tree.parent(nodes)-1]**2
                alpha = np.random.choice(nodes,
                                         p=proba_alpha/np.sum(proba_alpha))

                cost = tensor_current.nodes_permutation_cost(alpha)
                if np.any(cost != 0):
                    candidates = np.nonzero(cost != 0)[0]
                    proba_beta = 1 / cost[candidates]**2
                    beta = np.random.choice(candidates+1,
                                            p=proba_beta/np.sum(proba_beta))
                    sigma[i, :] = [alpha, beta]
                    tensor_current = tensor_current.permute_nodes(
                        sigma[i, :], tolerance/(m_current+m_star))
                    i += 1

            c_current = tensor_current.storage()
            if c_current < c_star and tensor_current.is_admissible_rank():
                ind = True
                c_star = c_current
                tensor_star = tensor_current
                sigma_star = np.vstack((sigma_star, sigma))
                m_star = sigma_star.shape[0]
        return tensor_star

    def permute_nodes(self, nodes, tolerance=1e-15):
        '''
         Permutation of two nodes of the tree.

        Permutations of the two nodes in nodes given a tolerance tol (for
        SVD-based truncations).

        Parameters
        ----------
        nodes : list or numpy.ndarray
            The two nodes to permute.
        tolerance : float, optional
            Relative precision for SVD truncations. The default is 1e-15.

        Raises
        ------
        ValueError
            If the first node to permute is an ascendant or a descendant of the
            second node to permute.

        Returns
        -------
        tensor : tensap.TreeBasedTensor
            The tree-based tensor with permuted nodes.

        '''
        tree = copy.deepcopy(self.tree)
        nodes = np.atleast_1d(nodes).astype(int)

        assert len(nodes) == 2, \
            'Must permute two nodes at most.'

        alpha, beta = nodes

        asc_alpha = tree.ascendants(alpha)
        asc_beta = tree.ascendants(beta)
        if np.isin(beta, asc_alpha) or \
                np.isin(beta, tree.descendants(alpha)):
            raise ValueError('Cannot permute the nodes a and b if b is ' +
                             'an ascendant or descendant of a.')
        if tree.parent(alpha) != tree.parent(beta):
            common_asc = tensap.fast_intersect(asc_alpha, asc_beta)
            gamma = common_asc[tree.level[common_asc-1] ==
                               np.max(tree.level[common_asc-1])][0]
            common_asc = tensap.fast_setdiff(common_asc, gamma)
            sub_nod = np.unique(np.concatenate((asc_alpha, asc_beta)))
            sub_nod = tensap.fast_setdiff(sub_nod, common_asc)

            trunc = tensap.Truncator(
                np.max((1e-15, tolerance/np.sqrt(sub_nod.size-1))))
            tensors = self.orth_at_node(gamma).tensors

            max_level = np.max(tree.level[sub_nod-1])
            if tree.root == gamma and self.ranks[tree.root-1] == 1:
                edges = tree.children(gamma)
            else:
                edges = np.concatenate((tree.children(gamma), [gamma]))

            for lvl in np.arange(tree.level[gamma-1]+1, max_level+1):
                nod_lvl = tree.nodes_with_level(lvl)
                for nod in tensap.fast_intersect(nod_lvl, sub_nod):
                    tensors[gamma-1] = tensors[nod-1].tensordot(
                        tensors[gamma-1], tensors[nod-1].order-1,
                        np.nonzero(edges == nod)[0][0])
                    tensors[nod-1] = []
                    children = tree.children(nod)
                    edges = np.concatenate((
                        np.atleast_1d(children),
                        edges[~np.in1d(edges, nod)]))

            perm = np.arange(edges.size)
            ind_alpha = np.nonzero(edges == alpha)[0][0]
            ind_beta = np.nonzero(edges == beta)[0][0]
            perm[[ind_alpha, ind_beta]] = perm[[ind_beta, ind_alpha]]
            tensors[gamma-1] = tensors[gamma-1].transpose(perm)
            edges = edges[perm]

            tree.adjacency_matrix[tree.parent(alpha)-1, alpha-1] = 0
            tree.adjacency_matrix[tree.parent(alpha)-1, beta-1] = 1
            tree.adjacency_matrix[tree.parent(beta)-1, beta-1] = 0
            tree.adjacency_matrix[tree.parent(beta)-1, alpha-1] = 1
            tree = tree._precompute_attributes()
            tree = tree.update_dims_from_leaves()

            for lvl in np.arange(max_level, tree.level[gamma-1], -1):
                nod_lvl = tree.nodes_with_level(lvl)
                for nod in tensap.fast_intersect(nod_lvl, sub_nod):
                    ch_nod = tree.children(nod)
                    ind = [np.nonzero(x == edges)[0][0] for x in ch_nod]
                    not_ind = tensap.fast_setdiff(np.arange(edges.size), ind)

                    tmp = tensors[gamma-1].matricize(ind)
                    trunc.max_rank = np.min(tmp.shape)
                    tmp = trunc.truncate(tmp)

                    edges = np.concatenate((edges[not_ind], [nod]))
                    new_rank = tmp.core.shape[0]

                    shape_nod = np.concatenate((
                        [tensors[gamma-1].shape[x] for x in ind],
                        [new_rank]))
                    tensors[nod-1] = tensap.FullTensor(tmp.space[0])
                    tensors[nod-1] = tensors[nod-1].reshape(shape_nod)

                    shape_gamma = np.concatenate((
                        [tensors[gamma-1].shape[x] for x in not_ind],
                        [new_rank]))
                    tensors[gamma-1] = tensap.FullTensor(tmp.space[1])
                    tensors[gamma-1] = tensors[gamma-1].reshape(shape_gamma)

                    perm = np.arange(tensors[gamma-1].order)
                    perm = np.concatenate((
                        perm[:tree.child_number(nod)-1],
                        [tensors[gamma-1].order-1],
                        perm[tree.child_number(nod)-1:
                             tensors[gamma-1].order-1]))
                    tensors[gamma-1] = tensors[gamma-1].transpose(perm)
                    tensors[gamma-1] = \
                        tensors[gamma-1].tensor_matrix_product(
                            np.diag(tmp.core.data),
                            tree.child_number(nod)-1)
                    edges = edges[perm]

            ch_gamma = tree.children(gamma)
            perm = [np.nonzero(x == edges)[0][0] for x in ch_gamma]
            if gamma != tree.root or self.ranks[tree.root-1] > 1:
                perm = np.concatenate((perm, [tensors[gamma-1].order-1]))
            tensors[gamma-1] = tensors[gamma-1].transpose(perm)

            tensor = TreeBasedTensor(tensors, tree)

            # Ensure that the tensor x is rank admissible
            trunc.tolerance = np.finfo(float).eps
            trunc.max_rank = tensor.ranks
            tensor = trunc.truncate(tensor)
        else:
            tensor = copy.deepcopy(self)
        return tensor

    def optimize_leaves_permutations(self, tolerance, max_iter):
        '''
        Optimization over the ordering of the leaves of the tree to obtain a
        representation of the tensor with lower complexity.

        Parameters
        ----------
        tolerance : float
            The relative tolerance for the tree changes.
        max_iter : int
            The maximum number of tree changes.

        Returns
        -------
        tensor_star : tensap.TreeBasedTensor
            The tree-based tensor with optimized leaves ordering.

        '''
        max_iter = int(max_iter)
        tensor_star = self

        nodes = self.tree.dim2ind

        proba_m = 1 / np.arange(1, self.order)**2

        c_star = self.storage()
        sigma_star = np.array([]).reshape(0, 2)
        m_perm = 0
        m_star = 0
        ind = True

        for _ in range(max_iter):
            m_current = np.random.choice(np.arange(1, self.order),
                                         p=proba_m/np.sum(proba_m))
            if m_perm < m_current or ind:
                m_perm = m_current
                ind = False
                tensor_sigma = copy.deepcopy(self)
                for i in range(m_star):
                    tensor_sigma = tensor_sigma.permute_nodes(
                        sigma_star[i, :], tolerance/(m_current+m_star))

            sigma = np.zeros([m_current, 2])
            tensor_current = copy.deepcopy(tensor_sigma)
            for i in range(m_current):
                proba_alpha = tensor_current.ranks[
                    tensor_current.tree.parent(nodes)-1]**2
                alpha = np.random.choice(nodes,
                                         p=proba_alpha/np.sum(proba_alpha))

                cost = tensor_current.nodes_permutation_cost(alpha)
                cost[np.logical_not(self.tree.is_leaf)] = 0
                if np.any(cost != 0):
                    candidates = np.nonzero(cost != 0)[0]
                    proba_beta = 1 / cost[candidates]**2
                    beta = np.random.choice(candidates+1,
                                            p=proba_beta/np.sum(proba_beta))
                    sigma[i, :] = [alpha, beta]
                    tensor_current = tensor_current.permute_nodes(
                        sigma[i, :], tolerance/(m_current+m_star))

            c_current = tensor_current.storage()
            if c_current < c_star and tensor_current.is_admissible_rank():
                ind = True
                c_star = c_current
                tensor_star = tensor_current
                sigma_star = np.vstack((sigma_star, sigma))
                m_star = sigma_star.shape[0]
        return tensor_star

    def permute_leaves(self, perm, tolerance=1e-15):
        '''
        Permutation of leaf nodes given a permutation of the dimensions.

        Permutations of the leaf nodes given a permutation perm of the
        dimensions and a tolerance (for SVD-based truncations).

        Parameters
        ----------
        perm : list or numpy.ndarray
            Permutation of (1,...,self.order).
        tolerance : float, optional
            Relative precision for SVD truncations. The default is 1e-15.

        Returns
        -------
        tensor : tensap.TreeBasedTensor
            The tree-based tensor with permuted leaf nodes.

        '''
        tensor = self
        perm = np.atleast_1d(perm)
        if not np.array_equal(perm, np.sort(perm)):
            # Decomposition of the permutation into a sequence of elementary
            # permutations of two dimensions
            elem_perm = []
            init = np.arange(perm.size)
            while not np.array_equal(perm, init):
                ind = np.nonzero(init != perm)[0][0]
                ind = [np.nonzero(perm == init[ind])[0][0],
                       np.nonzero(perm == perm[ind])[0][0]]
                elem_perm.append(ind)
                perm[ind] = perm[np.flip(ind)]

            for perm_dims in elem_perm:
                nodes_to_permute = self.tree.dim2ind[perm_dims]
                tensor = tensor.permute_nodes(nodes_to_permute,
                                              tolerance/len(elem_perm))
        return tensor

    def inactivate_nodes(self, nodes):
        '''
        Inactivate a list of nodes.

        Parameters
        ----------
        nodes : list or numpy.ndarray
            The list of nodes to inactivate.

        Returns
        -------
        TreeBasedTensor
            The tensor with inactivated nodes.

        '''
        tree = self.tree
        tensors = np.array(self.tensors)
        for level in np.arange(np.max(tree.level), -1, -1):
            nodes_lvl = tree.nodes_with_level(level)
            for nod in nodes_lvl:
                if np.isin(nod, nodes) and self.is_active_node[nod-1]:
                    parent = tree.parent(nod)
                    child_nb = tree.child_number(nod)
                    tensors[parent-1] = tensors[nod-1].tensordot(
                        tensors[parent-1], tensors[nod-1].order-1, child_nb-1)
                    ind = np.concatenate(
                        ([child_nb-1], tensap.fast_setdiff(
                            np.arange(tensors[parent-1].order), child_nb-1)))
                    tensors[parent-1] = tensors[parent-1].itranspose(ind)
                    tensors[nod-1] = tensap.FullTensor([])
        return TreeBasedTensor(tensors, tree)

    def eval_at_indices(self, indices, dims=None):
        '''
        Evaluate the tensor at indices.

        If dims is None, return
        s(k) = x(indices(k, 1), indices(k, 2), ..., indices(k, d)),
        1 <= k <= self.shape[0].

        If dims is not None, return a partial evaluation: up to a permutation
        (placing the dimensions dims on the left), return
        s(k, i_1, ..., i_d') = x(indices(k, 1), indices(k, 2), ...,
        indices(k, M), i_1, ..., i_d'),
        1 <= k <= self.shape[0], with M = dims.size and d' = self.order - M.

        Parameters
        ----------
        indices : list of numpy.ndarray
            The indices of the tensor.
        dims : list of numpy.ndarray, optional
            The dimensions associated with the indices. The default is None,
            indicating that indices refers to all the dimensions.

        Returns
        -------
        evaluations : numpy.ndarray or TreeBasedTensor
            The evaluations of the tensor.

        '''
        indices = np.atleast_2d(indices)
        shape = indices.shape[0]

        if dims is not None:
            tmp = [':'] * self.order
            dims = np.atleast_1d(dims)
            assert indices.shape[1] == dims.size, \
                'Wrong shape of input arguments.'
            for ind, dim in enumerate(dims):
                tmp[dim] = indices[:, ind]
            indices = tmp
        else:
            if indices.shape[1] != self.order:
                indices = np.transpose(indices)
                assert indices.shape[1] == self.order, \
                    'Wrong shape of input arguments.'
            indices = [indices[:, i] for i in range(self.order)]

        sub_tensor = self.sub_tensor(*indices)
        if dims is not None and np.size(dims) > 1 and shape == 1:
            sub_tensor = sub_tensor.squeeze()
        elif dims is not None and np.size(dims) > 1:
            sub_tensor = sub_tensor.eval_diag(dims)
        elif dims is None:
            sub_tensor = sub_tensor.eval_diag()
        return sub_tensor

    def eval_diag(self, dims=None, nargout=1):
        '''
        Extract the diagonal of the tensor.

        Parameters
        ----------
        nargout : int, optional
            The number of outputs. The default is 1, returning the diagonal.
            If set to 2, return the node tensors with their diagonal evaluated
            as well.

        Returns
        -------
        numpy.array or (numpy.array and list)
            The diagonal and, if nargout == 2, the node tensors with their
            diagonal evaluated.

        '''
        if dims is not None:
            raise NotImplementedError('Method not implemented.')

        tree = self.tree
        tensors = np.array(self.tensors)

        for level in np.arange(np.max(tree.level)-1, -1, -1):
            for nod in tensap.fast_intersect(tree.nodes_with_level(level),
                                             tree.internal_nodes):
                children = tree.children(nod)
                are_ch_active = self.is_active_node[children-1]
                ind_active = np.nonzero(are_ch_active)[0]

                if np.all(np.logical_not(are_ch_active)):
                    tensors[nod-1] = tensors[nod-1].eval_diag(
                        range(len(children)))
                else:
                    tmp = tensors[children[ind_active[0]]-1]
                    for ind in np.arange(1, ind_active.size):
                        tmp = tmp.outer_product_eval_diag(
                            tensors[children[ind_active[ind]]-1], 0, 0)
                    ind_inactive = np.nonzero(np.logical_not(are_ch_active))[0]
                    if ind_inactive.size != 0:
                        tensors[nod-1] = tmp.tensordot_eval_diag(
                            tensors[nod-1], np.arange(1, tmp.order),
                            ind_active, 0, ind_inactive)
                    else:
                        tensors[nod-1] = tmp.tensordot(tensors[nod-1],
                                                       np.arange(1, tmp.order),
                                                       ind_active)
                if nargout == 1:
                    tensors[children[ind_active]-1] = None
        diag = tensors[tree.root-1]
        if isinstance(diag, tensap.FullTensor) and diag.order == 1:
            diag = diag.numpy()
        return diag if nargout == 1 else (diag, tensors.tolist())

    def cat(self, tensor2):
        '''
        Concatenate the tensors.

        Parameters
        ----------
        tensor2 : TreeBasedTensor
            The second tensor to be concatenated.

        Returns
        -------
        tensor : TreeBasedTensor
            The concatenated tensors.

        '''
        tensors = np.array(self.tensors)
        for nod in np.arange(1, self.tree.nb_nodes+1):
            if self.is_active_node[nod-1]:
                tensors[nod-1] = tensors[nod-1].cat(tensor2.tensors[nod-1])
        return TreeBasedTensor(tensors, self.tree)

    def kron(self, tensor2):
        '''
        Kronecker product of tensors.

        Similar to numpy.kron but for tree-based tensors.

        Parameters
        ----------
        tensor2 : TreeBasedTensor
            The second tensor of the Kronecker product.

        Returns
        -------
        tensor : TreeBasedTensor
            The tensor resulting from the Kronecker product.

        '''
        tensors = np.array(self.tensors)
        for nod in np.arange(1, self.tree.nb_nodes+1):
            if self.is_active_node[nod-1]:
                tensors[nod-1] = tensors[nod-1].kron(tensor2.tensors[nod-1])
        return TreeBasedTensor(tensors, self.tree)

    def tensor_matrix_product(self, matrices, dims=None):
        '''
        Contract a tensor with matrices.

        The second dimension of the matrix matrices[k] is contracted with the
        k-th dimension of self, with the indices k given in dims (if provided).

        Parameters
        ----------
        matrices : numpy.ndarray or list of numpy.ndarray
            The matrices to use in the product.
        dims : list or numpy.ndarray, optional
            Indices of the contractions. The default is None, indicating all
            the dimensions.

        Returns
        -------
        TreeBasedTensor
            The tensor after the contractions with the matrices.

        '''
        if dims is None:
            assert isinstance(matrices, list), 'matrices should be a list.'
            assert len(matrices) == self.order, \
                'len(matrices) must be self.order.'
            dims = np.arange(self.order)
        else:
            dims = np.atleast_1d(dims)
            if not isinstance(matrices, list):
                matrices = [matrices]
            assert len(matrices) == dims.size, \
                'len(matrices) must be equal to dims.size.'

        tree = self.tree
        tensors = np.array(self.tensors)
        for dim in dims:
            nod = tree.dim2ind[dim]
            if not self.is_active_node[nod-1]:
                parent = tree.parent(nod)
                child_nb = tree.child_number(nod)
                tensors[parent-1] = tensors[parent-1].\
                    tensor_matrix_product(
                        matrices[np.nonzero(dim == dims)[0][0]], child_nb-1)
            else:
                tensors[nod-1] = tensors[nod-1].\
                    tensor_matrix_product(
                        matrices[np.nonzero(dim == dims)[0][0]], 0)
        return TreeBasedTensor(tensors, tree)

    def tensor_diagonal_matrix_product(self, matrices, dims=None):
        '''
        Contract a TreeBasedTensor with matrices built from their diagonals.

        The second dimension of the matrix matrices[k] is contracted with the
        k-th dimension of self, with the indices k given in dims (if provided).

        Parameters
        ----------
        matrices : numpy.ndarray or list of numpy.ndarray
            The diagonals of the matrices to use in the product.
        dims : list or numpy.ndarray, optional
            Indices of the contractions. The default is None, indicating all
            the dimensions.

        Returns
        -------
        TreeBasedTensor
            The tensor after the contractions with the matrices.

        '''
        if dims is None:
            assert isinstance(matrices, list), 'matrices should be a list.'
            assert len(matrices) == self.order, \
                'len(matrices) must be self.order.'
            dims = range(self.order)
        else:
            dims = np.array(dims)
            if not isinstance(matrices, list):
                matrices = [matrices]
            assert len(matrices) == dims.size, \
                'len(matrices) must be equal to dims.size.'

        matrices = [np.diag(np.reshape(x, [-1])) for x in matrices]
        return self.tensor_matrix_product(matrices, dims)

    def tensor_vector_product(self, vectors, dims=None):
        '''
        Compute the contraction of the tensor with vectors.

        Compute the contraction of self with each vector contained in the list
        vectors along dimensions specified by dims. The operation is such that
        V[k] is contracted with the dims[k]-th dimension of self.

        Parameters
        ----------
        vectors : numpy.ndarray or list of numpy.ndarray
            The vectors to use in the product.
        dims : list or numpy.ndarray, optional
            Indices of the contractions. The default is None, indicating all
            the dimensions.

        Returns
        -------
        TreeBasedTensor
            The tensor after the contractions with the vectors.

        '''
        if dims is None:
            assert isinstance(vectors, list), 'vectors should be a list.'
            assert len(vectors) == self.order, \
                'len(vectors) must be self.order.'
            dims = np.arange(self.order)
        else:
            dims = np.array(dims)
            if not isinstance(vectors, list):
                vectors = [vectors]
            assert len(vectors) == dims.size, \
                'len(vectors) must be equal to dims.size.'

        vectors = [np.reshape(x, [1, -1]) for x in vectors]
        return self.tensor_matrix_product(vectors, dims).squeeze(dims.tolist())

    def tensor_matrix_product_eval_diag(self, matrices, dims=None):
        '''
        Evaluate the diagonal of a tensor obtained by contraction with
        matrices.

        Provides the diagonal of the tensor obtained by contracting the tensor
        with matrices H[k] along dimensions dims(k)+1, for k = 0, ...,
        dims.size-1.

        Parameters
        ----------
        matrices : list
            The matrices to use in the product.
        dims : list or numpy.ndarray, optional
            Indices of the contractions. The default is None, indicating all
            the dimensions.

        Returns
        -------
        out : tensap.FullTensor
            The result of the contractions of the tensor with the matrices.

        '''
        if dims is None:
            tree = self.tree
            ind = np.nonzero(np.isin(tree.dim2ind, self.active_nodes))[0]
            tensors = self.tensor_matrix_product([matrices[i] for i in ind],
                                                 ind).tensors

            for level in np.arange(np.max(tree.level), -1, -1):
                nod_level = tensap.fast_intersect(tree.nodes_with_level(level),
                                                  tree.internal_nodes)
                nod_level = tensap.fast_intersect(nod_level, self.active_nodes)
                for nod in nod_level:
                    children = tree.children(nod)
                    active_ch = tensap.fast_intersect(children,
                                                      self.active_nodes)
                    inactive_ch = children
                    inactive_ch = \
                        inactive_ch[np.logical_not(
                            self.is_active_node[inactive_ch-1])]

                    if active_ch.size > 0:
                        tmp = tensors[active_ch[0]-1]
                        for k in np.arange(1, active_ch.size):
                            tmp = tmp.outer_product_eval_diag(
                                tensors[active_ch[k]-1], 0, 0)
                        tensors[nod-1] = tmp.tensordot(tensors[nod-1],
                                                       np.arange(1, tmp.order),
                                                       tree.child_number
                                                       (active_ch)-1)
                    if inactive_ch.size > 0:
                        ind = np.nonzero(tree.dim2ind == inactive_ch[0])[0][0]
                        tmp = tensap.FullTensor(matrices[ind])
                        for k in np.arange(1, inactive_ch.size):
                            ind = np.nonzero(tree.dim2ind ==
                                             inactive_ch[k])[0][0]
                            tmp = tmp.outer_product_eval_diag(
                                tensap.FullTensor(matrices[ind]), 0, 0)

                        if active_ch.size > 0:
                            tensors[nod-1] = \
                                tmp.tensordot_eval_diag(
                                    tensors[nod-1], np.arange(1, tmp.order),
                                    np.arange(1, tmp.order), 0, 0)
                        else:
                            tensors[nod-1] = tmp.tensordot(
                                tensors[nod-1], np.arange(1, tmp.order),
                                tree.child_number(inactive_ch)-1)
            data = tensors[tree.root-1]
        else:
            data = self.tensor_matrix_product(matrices, dims)
            dims = np.atleast_1d(dims)
            if np.all([self.shape[d] == 1 for d in dims]):
                if dims.size > 1:
                    data = data.squeeze(dims[1:]).full()
            else:
                raise NotImplementedError('Method not implemented.')
        return data

    def squeeze(self, dims=None):
        '''
        Remove the singleton dimensions of the tensor.

        Parameters
        ----------
        dims : list or numpy.ndarray, optional
            Dimensions to squeeze. The default is None, indicating all the
            singleton dimensions.

        Returns
        -------
        TreeBasedTensor
            The squeezed tensor.

        '''
        if dims is None:
            dims = np.arange(self.order)
            dims = dims[self.shape == 1]
        dims = np.sort(dims)
        remaining_dims = tensap.fast_setdiff(np.arange(self.order), dims)

        if remaining_dims.size == 0:
            tensor = self.full().squeeze()
            if isinstance(tensor, tensap.FullTensor):
                tensor = tensor.data
        else:
            tensors = np.array(self.tensors)
            tree = self.tree
            ind = tree.dim2ind[dims]
            for level in np.arange(np.max(tree.level), -1, -1):
                nod_level = tensap.fast_setdiff(tree.nodes_with_level(level),
                                                tree.dim2ind)
                for nod in nod_level:
                    children = tree.children(nod)
                    children_r = np.isin(children, ind)
                    for num, child in enumerate(children):
                        if self.is_active_node[child-1] and children_r[num]:
                            if tensors[child-1].order == 1:
                                tensors[child-1] = tensors[child-1].reshape(
                                    [1, -1])
                            tensors[nod-1] = tensors[nod-1].\
                                tensor_matrix_product(tensors[child-1], num)
                            tensors[child-1] = []
                    if np.any(children_r):
                        tensors[nod-1] = tensors[nod-1].squeeze(
                            np.nonzero(children_r)[0].tolist())
                    if np.all(children_r):
                        ind = np.union1d(ind, nod)

            keep_ind = tensap.fast_setdiff(np.arange(tree.nb_nodes), ind-1)
            tensors = tensors[keep_ind]
            adj = tree.adjacency_matrix[np.ix_(keep_ind, keep_ind)]
            dim2ind = [np.nonzero(x == keep_ind + 1)[0][0] + 1 for
                       x in tree.dim2ind[remaining_dims]]
            tree = tensap.DimensionTree(dim2ind, adj)
            tensor = TreeBasedTensor(tensors, tree)
            tensor = tensor.remove_unique_children()
        return tensor

    def remove_unique_children(self):
        '''
        Remove the unique children of a tree-based tensor (nodes with no
        siblings in the tree).

        Returns
        -------
        TreeBasedTensor
            The tensor with no unique children.

        '''
        tree = self.tree
        tensors = self.tensors
        nb_children = np.count_nonzero(tree._children, 0)
        unique_children = tree._children[0, (nb_children == 1)]
        adj = tree.adjacency_matrix
        dim2ind = np.array(tree.dim2ind)

        for level in np.arange(np.max(tree.level), -1, -1):
            nod_level = tensap.fast_intersect(unique_children,
                                              tree.nodes_with_level(level))
            for nod in nod_level:
                parent = np.nonzero(adj[:, nod-1])[0][0] + 1
                if self.is_active_node[nod-1]:
                    tensors[parent-1] = \
                        tensors[nod-1].tensordot(tensors[parent-1],
                                                 tensors[nod-1].order-1, 0)
                    adj[parent-1, :] = adj[nod-1, :]
                    adj[nod-1, :] = 0
                tensors[nod-1] = tensap.FullTensor([])
                dim2ind[dim2ind == nod] = parent
        keep_ind = tensap.fast_setdiff(np.arange(tree.nb_nodes),
                                       unique_children-1)
        dim2ind = [np.nonzero(x == keep_ind + 1)[0][0] + 1 for
                   x in dim2ind]
        tensors = tensors[keep_ind]
        adj = adj[np.ix_(keep_ind, keep_ind)]

        tree = tensap.DimensionTree(dim2ind, adj)
        return TreeBasedTensor(tensors, tree)

    def dot(self, tensor2):
        '''
        Return the inner product of two tensors.

        Parameters
        ----------
        tensor2 : TreeBasedTensor
            The second tensor of the inner products.

        Returns
        -------
        numpy.float
            The inner product of the two tensors.

        '''
        matrices = [np.eye(x) for x in self.shape]
        return self.dot_with_rank_one_metric(tensor2, matrices)

    def norm(self):
        '''
        Compute the canonical norm of the tensor.

        Returns
        -------
        numpy.float
            The norm of the tensor.

        '''
        if not self.is_orth:
            tensor = self.orth()
        else:
            tensor = copy.deepcopy(self)
        return tensor.tensors[tensor.orth_node-1].norm()

    def gramians(self, alpha=None):
        '''
        Compute the Gram matrices of the bases of minimal subspaces associated
        with nodes of the tree.

        Parameters
        ----------
        alpha : list or numpy.ndarray, optional
            The nodes associated to the Gram matrices to be computed. The
            default is None, for all the nodes of the tree.

        Returns
        -------
        gram : list
            The Gram matrices.
        TreeBasedTensor
            The orthogonalized TreeBasedTensor.

        '''
        tree = self.tree
        if not self.is_orth or self.orth_node != tree.root:
            tensor = self.orth()
        else:
            tensor = copy.deepcopy(self)

        gram = np.empty(tree.nb_nodes, dtype=object)
        gram[tree.root-1] = np.atleast_2d(1.)
        list_nodes = tensor.active_nodes

        if alpha is None:
            max_level = np.max(tree.level)
        else:
            alpha = np.atleast_1d(alpha)
            max_level = np.max(tree.level[alpha-1])
            ascendants = np.array([asc for nod in np.nditer(alpha)
                                   for asc in tree.ascendants(nod)],
                                  dtype=alpha.dtype)
            list_nodes = tensap.fast_intersect(list_nodes,
                                               np.concatenate((ascendants,
                                                               alpha)))

        for level in range(max_level+1):
            nod_level = tensap.fast_intersect(tree.nodes_with_level(level),
                                              list_nodes)
            nod_level = tensap.fast_setdiff(nod_level, tree.dim2ind)

            for nod in nod_level:
                tmp = tensor.tensors[nod-1]
                if nod != tree.root:
                    tmp = tmp.tensor_matrix_product(gram[nod-1],
                                                    tensor.tensors[nod-1].
                                                    order-1)
                children = tensap.fast_intersect(list_nodes,
                                                 tree.children(nod))
                for child in children:
                    c_list = tensap.fast_setdiff(np.arange(
                        tensor.tensors[nod-1].order),
                        tree.child_number(child)-1)
                    gram[child-1] = tensor.tensors[nod-1].\
                        tensordot(tmp, c_list, c_list).numpy()
        if alpha is not None:
            if alpha.size == 1:
                gram = gram[alpha[0]-1]
            else:
                gram = gram[alpha-1]
        return gram, tensor

    def orth(self):
        '''
        Orthogonalize the representation of the tensor.

        All core tensors except the root core represents orthonormal bases of
        principal subspaces.

        Returns
        -------
        tensor : TreeBasedTensor
            The TreeBasedTensor with an orthogonal representation.

        '''
        tensors = np.array(self.tensors)
        tree = self.tree
        max_level = np.max(tree.level)

        for level in np.arange(max_level, 0, -1):
            nod_level = tensap.fast_intersect(tree.nodes_with_level(level),
                                              self.active_nodes)
            for nod in nod_level:
                if not tensors[nod-1].is_orth or \
                    (tensors[nod-1].is_orth and
                     tensors[nod-1].orth_dim !=
                     tensors[nod-1].order):
                    tensors[nod-1], r_matrix = \
                        tensors[nod-1].orth(tensors[nod-1].order-1)
                    parent_nod = tree.parent(nod)
                    child_number = tree.child_number(nod)
                    tensors[parent_nod-1] = \
                        tensors[parent_nod-1].tensor_matrix_product(
                            r_matrix, child_number-1)
                    tensors[parent_nod-1].is_orth = False

        tensor = TreeBasedTensor(tensors, tree)
        tensor.is_orth = True
        tensor.orth_node = tree.root
        return tensor

    def orth_at_node(self, nod):
        '''
        Orthogonalize the representation with respect to a given node.

        All core tensors except the one of node nod represents orthonormal
        bases of principal subspaces the core tensor of node nod is such that
        the tensor x(i_alpha,i_alpha^c) = sum_k u_k(i_alpha) w_k(i_alpha^c),
        where w_k is a set of orthonormal vectors.

        Parameters
        ----------
        nod : int
            The node with respect to which the representation is orthogonal.

        Raises
        ------
        ValueError
            If nod is non active.

        Returns
        -------
        tensor : TreeBasedTensor
            The TreeBasedTensor with an orthogonal representation.

        '''
        if nod == self.tree.root:
            tensor = self.orth()
        elif self.is_active_node[nod-1]:
            gram, tensor = self.gramians(nod)
            u_svd, s_svd, _ = np.linalg.svd(gram)
            s_svd = np.sqrt(np.abs(s_svd))
            l_svd = np.matmul(u_svd, np.diag(s_svd))

            rep = np.linalg.matrix_rank(l_svd)
            if rep == 0:
                print('Zero alpha-rank, returning the original tensor.')
                return tensor
            if rep != gram.shape[0]:
                l_svd = l_svd[:, :rep]
                s_svd = np.linalg.pinv(l_svd)
            else:
                s_svd = np.linalg.inv(l_svd)
            tensor.tensors[nod-1] = tensor.tensors[nod-1].\
                tensor_matrix_product(np.transpose(l_svd),
                                      tensor.tensors[nod-1].order-1)
            tensor.tensors[nod-1].is_orth = False
            parent_nod = tensor.tree.parent(nod)
            tensor.tensors[parent_nod-1] = tensor.tensors[parent_nod-1].\
                tensor_matrix_product(s_svd, tensor.tree.child_number(nod)-1)
            tensor.tensors[parent_nod-1].is_orth = False
            tensor.orth_node = nod
            tensor.is_orth = False
            tensor = tensor.update_attributes()
        else:
            raise ValueError('Non active node.')
        return tensor

    def dot_with_rank_one_metric(self, tensor2, matrices):
        '''
        Compute the weighted inner product of two tree-based tensors.

        Compute the weighted canonical inner product of self and tensor2,
        where the inner product related to dimension k is weighted by
        matrix[k]. It is equivalent to
        self.dot(tensor2.tensor_matrix_product(matrix)),
        but can be much faster.

        Parameters
        ----------
        tensor2 : TreeBasedTensor
            The second tensor of the inner product.
        matrix : list or numpy.ndarray
            The weight matrices.

        Returns
        -------
        numpy.float
            The weighted inner product.

        '''
        tree = self.tree
        tensors1 = np.array(self.tensors)
        tensors2 = tensor2.tensors
        out = np.empty(tree.nb_nodes, dtype=object)
        out[tree.dim2ind-1] = matrices

        for nod in tree.dim2ind:
            if self.is_active_node[nod-1]:
                tmp = tensors2[nod-1].tensor_matrix_product(out[nod-1], 0)
                out[nod-1] = tensors1[nod-1].tensordot(tmp, 0, 0)
                out[nod-1] = out[nod-1].reshape([tensors1[nod-1].shape[-1],
                                                 tensors2[nod-1].shape[-1]])
                out[nod-1] = out[nod-1].numpy()

        for level in np.arange(np.max(tree.level)-1, -1, -1):
            nod_level = tensap.fast_intersect(tree.nodes_with_level(level),
                                              tree.internal_nodes)
            for nod in nod_level:
                children = self.tree.children(nod)
                if nod != tree.root or self.ranks[tree.root-1] > 1:
                    order = np.arange(tensors1[nod-1].order-1)
                else:
                    order = np.arange(tensors1[nod-1].order)
                tmp = tensors2[nod-1].tensor_matrix_product(
                    out[children-1].tolist(), order)
                out[nod-1] = tensors1[nod-1].tensordot(tmp, order, order)
                if level != 0:
                    out[nod-1] = out[nod-1].reshape(
                        [tensors1[nod-1].shape[-1], tensors2[nod-1].shape[-1]])
                out[nod-1] = out[nod-1].numpy()
        out = out[nod_level-1]
        if out.size == 1:
            out = out[0]
        return out

    def tensordot_matrix_product_except_dim(self, tensor2, matrices, dim):
        '''
        Particular type of contraction.

        Compute a special contraction of two tensors self, tensor2, a list of
        matrices matrices and a particular dimension dim. Note that dim must
        be a scalar, while matrices must be a list array with x.self.order
        elements.

        Parameters
        ----------
        tensor2 : TreeBasedTensor
            The second tensor of the contraction.
        matrices : list
            The list of matrices of the contraction.
        dim : int
            The excluded dimension.

        Returns
        -------
        numpy.ndarray
            The result of the contraction.

        '''
        ind = self.tree.dim2ind(dim-1)
        out = self.reduce_dot_with_rank_one_metric_at_node(tensor2, matrices,
                                                           ind)
        return out[0]

    # def reduce_dot_with_rank_one_metric_at_node(self, tensor2, matrices,
    #                                             alpha):
    #     raise NotImplementedError('Not implemented.')
    #     if np.any(self.is_active_node[self.tree.is_leaf]):
    #         raise NotImplementedError('Not implemented for active leaves.')
    #     tree = self.tree
    #     tensors1 = np.array(self.tensors)
    #     tensors2 = np.array(tensor2.tensors)
    #     nod_level = tree.level[alpha-1]
    #     is_root = nod_level == 0
    #     is_leaf = tree.is_leaf[alpha-1]

    #     tmp = np.empty(tree.nb_nodes, dtype=object)
    #     tmp[tree.dim2ind-1] = matrices
    #     if not is_leaf:
    #         out = np.empty(tensors1[alpha-1].order, dtype=object)
    #     else:
    #         out = np.empty(1, dtype=object)
    #     descendants = tree.descendants(alpha)
    #     is_descendant = [bool(i in descendants) for
    #                      i in np.arange(1, tree.nb_nodes + 1)]
    #     is_not_leaf = np.logical_not(tree.is_leaf)

    #     if not is_leaf:
    #         # Contract under the node alpha
    #         for level in np.arange(np.max(tree.level)-1, nod_level, -1):
    #             nod_level = [tree.nodes_indices[nod]
    #                          for nod in range(tree.nb_nodes)
    #                          if tree.level[nod] == level and is_not_leaf[nod]
    #                          and is_descendant[nod]]
    #             for nod in nod_level:
    #                 children = tree.children(nod)
    #                 if tree.parent(nod) != 0:
    #                     ind = range(tensors1[nod-1].order-1)
    #                 else:
    #                     ind = range(tensors1[nod-1].order)
    #                 tmp[nod-1] = tensors1[nod-1].tensor_matrix_product(
    #                     tensors2[nod-1].tensor_matrix_product(out[children-1],
    #                                                           ind), ind, ind)
    #                 # tmp[nod-1] = tmp[nod-1].reshape([tensors[nod-1].shape[]])
    #                 tmp[nod-1] = tmp[nod-1].numpy()

    #     if not is_root:
    #         pass

    #     # Fill the output
    #     if not is_leaf:
    #         children = tree.children(alpha)
    #         out[:children.size] = tmp[children-1]
    #     if not is_root:
    #         out[-1] = tmp[tree.parent(alpha)-1]

    #     return out

    def singular_values(self):
        '''
        Compute the tree-based singular values of a tensor, which are the
        singular values associated with alpha-matricizations of the tensor,
        for all alpha in the dimension tree.

        Returns
        -------
        numpy.ndarray
            The tree-based singular values of the tensor.

        '''
        tensor = self.orth()
        gramians, _ = tensor.gramians()
        return np.array([np.sqrt(np.linalg.svd(gram, compute_uv=False))
                         if gram is not None else None for gram in gramians])

    def plot(self, nodes_labels=None, title=None):
        '''
        Plot the tree with the nodes indices and the active nodes.

        This method requires the package igraph.

        Parameters
        ----------
        nodes_labels : list or numpy.ndarray, optional
            The labels of the nodes. The default is None, displaying the
            nodes numbers.
        title : str, optional
            The title of the graph. The default is None.

        Returns
        -------
        None.

        '''
        if nodes_labels is None:
            nodes_labels = self.tree.nodes_indices
        self.tree.plot_with_labels_at_nodes(nodes_labels,
                                            colored_nodes=self.active_nodes,
                                            title=title)

    def eval_diag_below(self, alpha=None, except_nodes=[]):
        '''
        Evaluate the diagonal of the tensor of the function v^beta of the
        representation
        f = sum_{k=1}^{beta} v^beta_k w^beta_k
        (optionally for all the nodes except the ascendants of a node alpha).

        Parameters
        ----------
        alpha : int, optional
            A node of the tree, exluding its ascendants in the computation of
            the functions v^beta. The default is None, indicating that no
            node is excluded.
        except_nodes : int or list or numpy.array, optional
            Nodes for which the computation is not performed. The default is
            None.

        Returns
        -------
        diag_below : numpy.ndarray
            An array containing the diagonals of the tensor of the functions
            v^beta for the included nodes, and None for the excluded nodes.

        '''
        tree = self.tree
        diag_below = np.array(self.tensors)

        if alpha is not None and alpha != tree.root:
            exclude_list = np.union1d(np.concatenate(([alpha],
                                                      tree.ascendants(alpha))),
                                      except_nodes)
        elif np.size(except_nodes) != 0:
            exclude_list = np.atleast_1d(except_nodes)
        else:
            exclude_list = np.array([], dtype=int)
        exclude_list = exclude_list.astype(int)
        diag_below[exclude_list[1:]-1] = None

        for level in np.arange(np.max(tree.level)-1, -1, -1):
            nod_level = reduce(tensap.fast_intersect,
                               (tree.nodes_with_level(level),
                                tree.internal_nodes,
                                self.active_nodes))
            nod_level = tensap.fast_setdiff(nod_level, exclude_list)
            for nod in nod_level:
                children = tensap.fast_setdiff(tree.children(nod),
                                               exclude_list)
                active_ch = children[self.is_active_node[children-1]]
                inactive_ch = tensap.fast_setdiff(children, active_ch)
                if active_ch.size != 0:
                    tmp = diag_below[active_ch[0]-1]
                    for child in np.arange(1, active_ch.size):
                        tmp = tmp.outer_product_eval_diag(
                            diag_below[active_ch[child]-1], 0, 0)
                    if inactive_ch.size != 0 and nod != tree.root:
                        diag_below[nod-1] = \
                            diag_below[nod-1].tensordot_eval_diag(
                                tmp, tree.child_number(active_ch)-1,
                                np.arange(1, tmp.order),
                                tree.child_number(inactive_ch)-1, 0)
                    else:
                        diag_below[nod-1] = tmp.tensordot(
                            diag_below[nod-1], np.arange(1, tmp.order),
                            tree.child_number(active_ch)-1)
                else:
                    ch_nb = tree.child_number(inactive_ch)-1
                    if np.size(inactive_ch) > 1:
                        diag_below[nod-1] = diag_below[nod-1].eval_diag(ch_nb)
                    perm = np.concatenate(
                        ([ch_nb[0]],
                         np.setdiff1d(np.arange(diag_below[nod-1].order),
                                      ch_nb[0])))
                    diag_below[nod-1] = diag_below[nod-1].transpose(perm)

        return diag_below

    def eval_diag_above(self, diag_below=None, alpha=None):
        '''
        Evaluate the diagonal of the tensor of the function w^\alpha of the
        representation
        f = sum_{k=1}^{beta} v^beta_k w^beta_k
        (optionally for the node alpha and its ascendants).

        Parameters
        ----------
        diag_below : numpy.ndarray, optional
            The result of the method eval_diag_below. The default is None,
            calling the method eval_diag_below.
        alpha : int, optional
            A node of the tree, including it and its ascendants in the
            computation of the functions w^beta. The default is None,
            indicating that all the nodes are included.

        Returns
        -------
        diag_above : numpy.ndarray or tensap.FullTensor
            The diagonals of the tensor of the functions w^beta for the
            included nodes, and None for the excluded nodes.

        '''
        tree = self.tree
        diag_above = np.empty(tree.nb_nodes, dtype=object)
        tensors = np.array(self.tensors)

        if diag_below is None:
            diag_below = self.eval_diag_below()
        if alpha is None:
            include_list = np.arange(1, tree.nb_nodes+1)
        else:
            include_list = np.concatenate(([alpha], tree.ascendants(alpha)))
            if alpha == tree.root:
                diag_above[tree.root-1] = tensap.FullTensor.ones(
                    diag_below[tree.root-1].shape)

        for level in np.arange(1, np.max(tree.level)+1):
            nod_level = reduce(tensap.fast_intersect,
                               (tree.nodes_with_level(level),
                                self.active_nodes,
                                include_list))
            for nod in nod_level:
                parent = tree.parent(nod)
                children = tensap.fast_setdiff(tree.children(parent), nod)
                active_ch = children[self.is_active_node[children-1]]
                inactive_ch = tensap.fast_setdiff(children, active_ch)
                if active_ch.size != 0:
                    tmp = diag_below[active_ch[0]-1]
                    for child in np.arange(1, active_ch.size):
                        tmp = tmp.outer_product_eval_diag(
                            diag_below[active_ch[child]-1], 0, 0)
                    if inactive_ch.size != 0:
                        diag_above[nod-1] = tmp.tensordot_eval_diag(
                            tensors[parent-1], np.arange(1, tmp.order),
                            tree.child_number(active_ch)-1, 0,
                            tree.child_number(inactive_ch)-1)
                    else:
                        diag_above[nod-1] = tmp.tensordot(
                            tensors[parent-1], np.arange(1, tmp.order),
                            tree.child_number(active_ch)-1)
                elif tree.child_number(inactive_ch).size != 1:
                    diag_above[nod-1] = tensors[parent-1].eval_diag(
                        tree.child_number(inactive_ch)-1)
                    ind = tensap.fast_setdiff(
                        np.arange(diag_above[nod-1].order),
                        tree.child_number(inactive_ch[0])-1)
                    ind = np.concatenate(
                        ([tree.child_number(inactive_ch[0])-1], ind))
                    diag_above[nod-1] = diag_above[nod-1].transpose(ind)
                else:
                    ind = tensap.fast_setdiff(
                        np.arange(tensors[parent-1].order),
                        tree.child_number(inactive_ch)-1)
                    ind = np.concatenate(
                        (np.atleast_1d(tree.child_number(inactive_ch)-1),
                         ind))
                    diag_above[nod-1] = tensors[parent-1].transpose(ind)
                if parent != tree.root:
                    diag_above[nod-1] = \
                        diag_above[parent-1].tensordot_eval_diag(
                            diag_above[nod-1], 1, 2, 0, 0)
                    if diag_above[nod-1].order == 3:
                        diag_above[nod-1] = \
                            diag_above[nod-1].transpose([0, 2, 1])
        if alpha is not None:
            diag_above = diag_above[alpha-1]
        return diag_above

    def parameter_gradient_eval_diag(self, alpha, matrices=None):
        '''
        Compute the diagonal of the gradient of the tensor with respect to a
        given parameter.

        Parameters
        ----------
        alpha : int
            Index of node of the dimension tree.
        matrices : list or numpy.array, optional
            Matrices with which to compute outer_product_eval_diag if alpha is
            associated with some dimensions. Useful for evaluating the gradient
            of a tensap.FunctionalTensor. The default is None, indicating
            identity matrices.

        Returns
        -------
        out : tensap.FullTensor
            The diagonal of the gradient of the tensor with respect to
            self.tensors[alpha-1].

        '''
        tree = self.tree
        tensors = np.array(self.tensors)

        diag_below = self.eval_diag_below(alpha)
        diag_above = self.eval_diag_above(diag_below, alpha)

        children = tree.children(alpha)
        active_ch = children[self.is_active_node[children-1]]
        inactive_ch = tensap.fast_setdiff(children, active_ch)

        if tree.is_leaf[alpha-1]:
            out = tensap.FullTensor.ones(diag_above.shape[0])
            if matrices is not None:
                ind = np.nonzero(tree.dim2ind == alpha)[0][0]
                out = out.outer_product_eval_diag(
                    tensap.FullTensor(matrices[ind]), 0, 0)
            else:
                out = out.outer_product_eval_diag(
                    tensap.FullTensor(np.eye(diag_below[alpha-1].shape[0])),
                    [], [], True)
            # out = out.squeeze()

            # ind = np.array([[np.nonzero(tree.dim2ind == alpha)[0][0]], [1]])
        else:
            if active_ch.size != 0:
                out = diag_below[active_ch[0]-1]
                for child in np.arange(1, active_ch.size):
                    out = out.outer_product_eval_diag(
                        diag_below[active_ch[child]-1], 0, 0)
            if inactive_ch.size != 0:
                tmp = tensap.FullTensor.ones(diag_above.shape[0])
                for child in inactive_ch:
                    if matrices is not None:
                        ind = np.nonzero(tree.dim2ind == child)[0][0]
                        tmp = tmp.outer_product_eval_diag(
                            tensap.FullTensor(matrices[ind]), 0, 0)
                    else:
                        ind = tree.child_number(child)-1
                        tmp = tmp.outer_product_eval_diag(
                            tensap.FullTensor(np.eye(
                                tensors[alpha-1].shape[ind])), [], [], True)
                # tmp = tmp.squeeze()
                if active_ch.size != 0:
                    out = out.outer_product_eval_diag(tmp, 0, 0)
                else:
                    out = tmp
            # ind = np.array(
            #     [np.nonzero(np.isin(tree.dim2ind, inactive_ch))[0],
            #      active_ch.size+1+np.arange(inactive_ch.size)])
        if alpha != tree.root:
            out = out.outer_product_eval_diag(diag_above, 0, 0)
        return out  # , ind

    def parameter_gradient_eval_diag_dmrg(self, alpha, matrices=None):
        '''
        Return the diagonal of the gradient of the tensor with respect to a
        given parameter, obtained by contraction of two node tensors along
        their common edge; used in a DMRG algorithm.

        Parameters
        ----------
        alpha : int
            Index of node of the dimension tree.
        matrices : list or numpy.array, optional
            Matrices with which to compute outer_product_eval_diag if alpha is
            associated with some dimensions. Useful for evaluating the gradient
            of a tensap.FunctionalTensor. The default is None, indicating
            identity matrices.

        Returns
        -------
        out : tensap.FullTensor
            The diagonal of the gradient of the tensor with respect to the
            parameter.

        '''
        tree = self.tree

        gamma = tree.parent(alpha)
        u_gamma = self.eval_diag_below(gamma, alpha)
        w_gamma = self.eval_diag_above(u_gamma, gamma)

        children = tensap.fast_setdiff(tree.children(gamma), alpha)
        active_ch = children[self.is_active_node[children-1]]
        inactive_ch = tensap.fast_setdiff(children, active_ch)

        if active_ch.size != 0:
            g_gamma = u_gamma[active_ch[0]-1]
            for child in np.arange(1, active_ch.size):
                g_gamma = g_gamma.outer_product_eval_diag(
                    u_gamma[active_ch[child]-1], 0, 0)
        if inactive_ch.size != 0:
            tmp = tensap.FullTensor.ones(w_gamma.shape[0])
            for child in inactive_ch:
                if matrices is not None:
                    ind = np.nonzero(tree.dim2ind == child)[0][0]
                    tmp = tmp.outer_product_eval_diag(
                        tensap.FullTensor(matrices[ind]), 0, 0)
                else:
                    ind = tree.child_number(child)-1
                    tmp = tmp.outer_product_eval_diag(
                        tensap.FullTensor(np.eye(
                            self.tensors[gamma-1].shape[ind])), [], [], True)
            # tmp = tmp.squeeze()
            if active_ch.size != 0:
                g_gamma = g_gamma.outer_product_eval_diag(tmp, 0, 0)
            else:
                g_gamma = tmp

        if gamma != tree.root:
            g_gamma = g_gamma.outer_product_eval_diag(w_gamma, 0, 0)

        u_alpha = self.eval_diag_below(alpha)

        children = tree.children(alpha)
        active_ch = children[self.is_active_node[children-1]]
        inactive_ch = tensap.fast_setdiff(children, active_ch)
        if tree.is_leaf[alpha-1]:
            g_alpha = tensap.FullTensor.ones(w_gamma.shape[0])
            if matrices is not None:
                ind = np.nonzero(tree.dim2ind == alpha)[0][0]
                g_alpha = g_alpha.outer_product_eval_diag(
                    tensap.FullTensor(matrices[ind]), 0, 0)
            else:
                g_alpha = g_alpha.outer_product_eval_diag(
                    tensap.FullTensor(np.eye(u_alpha[alpha-1].shape[0])),
                    [], [], True)
            # g_alpha = g_alpha.squeeze()
        else:
            if active_ch.size != 0:
                g_alpha = u_alpha[active_ch[0]-1]
                for child in np.arange(1, active_ch.size):
                    g_alpha = g_alpha.outer_product_eval_diag(
                        u_alpha[active_ch[child]-1], 0, 0)
            if inactive_ch.size != 0:
                tmp = tensap.FullTensor.ones(w_gamma.shape[0])
                for child in inactive_ch:
                    if matrices is not None:
                        ind = np.nonzero(tree.dim2ind == child)[0][0]
                        tmp = tmp.outer_product_eval_diag(
                            tensap.FullTensor(matrices[ind]), 0, 0)
                    else:
                        ind = tree.child_number(child)-1
                        tmp = tmp.outer_product_eval_diag(
                            tensap.FullTensor(np.eye(
                                self.tensors[alpha-1].shape[ind])),
                            [], [], True)
                # tmp = tmp.squeeze()
                if active_ch.size != 0:
                    g_alpha = g_alpha.outer_product_eval_diag(tmp, 0, 0)
                else:
                    g_alpha = tmp

        return g_alpha.outer_product_eval_diag(g_gamma, 0, 0), g_alpha, g_gamma

    def update_attributes(self):
        '''
        Update the attributes of the TreeBasedTensor.

        Returns
        -------
        TreeBasedTensor
            A TreeBasedTensor with updated attributes.

        '''
        self.ranks = np.zeros(self.tree.nb_nodes, dtype='int')
        self.is_active_node = np.array([x.storage() != 0 for
                                        x in self.tensors])

        if not np.all([self.is_active_node[i-1] for i in
                       self.tree.internal_nodes]):
            raise NotImplementedError('Method not implemented for this '
                                      'format.')

        for nod in self.tree.internal_nodes:
            children = self.tree.children(nod)
            if self.tree.parent(nod) != 0 or \
                    self.tensors[nod-1].order == len(children)+1:
                self.ranks[nod-1] = self.tensors[nod-1].shape[-1]
            else:
                self.ranks[nod-1] = 1

        self.shape = np.zeros(self.order, dtype='int')
        for nod in self.tree.dim2ind:
            dim = np.nonzero(self.tree.dim2ind == nod)[0]
            if self.tensors[nod-1].storage() == 0:
                parent_nod = self.tree.parent(nod)
                ind = self.tree.child_number(nod)
                self.shape[dim] = self.tensors[parent_nod-1].shape[ind-1]
            else:
                self.shape[dim] = self.tensors[nod-1].shape[0]
                self.ranks[nod-1] = self.tensors[nod-1].shape[-1]
        return self

    @staticmethod
    def create(generator, tree, ranks=None, shape=None, is_active_node=None):
        '''
        Create a tree-based tensor from a generator.

        Parameters
        ----------
        generator : function
            Function generating a tensap.FullTensor, given a shape.
        tree : tensap.FullTensor
            The tensap.FullTensor of the TreeBasedTensor.
        ranks : numpy.ndarray, list or 'random', optional
            Ranks of the TreeBasedTensorFormat. The default is None, generating
            random ranks between 1 and 5.
        shape : numpy.ndarray, list or 'random', optional
            The size of the spaces of the leaves of the tree. The default is
            None, assigning ranks to shape.
        is_active_node : numpy.ndarray, list or 'random', optional
            Booleans indicating if the node is active. The default is None, for
            which all the nodes are active.

        Raises
        ------
        ValueError
            If some internal nodes are inactive.

        Returns
        -------
        TreeBasedTensor
            A TreeBasedTensor with the input characteristics.

        '''
        if ranks is None or (isinstance(ranks, str) and ranks == 'random'):
            ranks = np.random.randint(1, 6, tree.nb_nodes)
            ranks[tree.root-1] = 1
        else:
            ranks = np.atleast_1d(ranks)

        if shape is None:
            shape = ranks[tree.dim2ind-1]
        elif isinstance(shape, str) and shape == 'random':
            shape = np.random.randint(1, 11, tree.dim2ind.size)
        else:
            shape = np.atleast_1d(shape)

        if is_active_node is None:
            is_active_node = np.full(tree.nb_nodes, True)
        elif isinstance(is_active_node, str) and is_active_node == 'random':
            is_active_node = TreeBasedTensor._random_is_active_node(tree)
        else:
            is_active_node = np.atleast_1d(is_active_node)

        tensors = np.empty(tree.nb_nodes, dtype='object')
        for nod in range(tree.nb_nodes):
            if is_active_node[nod] and not tree.is_leaf[nod]:
                children = tree._children[:, nod]
                children = children[children != 0]

                shape_i = np.array([], dtype=int)
                for child in children:
                    if is_active_node[child-1]:
                        shape_i = np.concatenate((shape_i, [ranks[child-1]]))
                    elif tree.is_leaf[child-1]:
                        shape_i = np.concatenate((shape_i,
                                                 shape[tree.dim2ind == child]))
                    elif not tree.is_leaf[child-1] and \
                            not is_active_node[child-1]:
                        raise ValueError('Inactive nodes should be leaves.')
                if nod+1 != tree.root or (nod+1 == tree.root and
                                          ranks[nod] > 1):
                    shape_i = np.concatenate((shape_i, [ranks[nod]]))
                tensors[nod] = tensap.FullTensor(generator(shape_i.tolist()))
            elif is_active_node[nod] and tree.is_leaf[nod]:
                shape_i = np.concatenate((shape[tree.dim2ind == nod+1],
                                         [ranks[nod]]))
                tensors[nod] = tensap.FullTensor(generator(shape_i.tolist()))
            elif not is_active_node[nod] and not tree.is_leaf[nod]:
                raise ValueError('Inactive nodes should be leaves.')
            else:
                tensors[nod] = []
        return TreeBasedTensor(tensors, tree)

    @staticmethod
    def randn(tree, ranks=None, shape=None, is_active_node=None):
        '''
        Create a tensor of shape shape and tree-based rank ranks with node
        tensors generated with the method random.randn of numpy.

        Parameters
        ----------
        tree : tensap.FullTensor
            The tensap.FullTensor of the TreeBasedTensor.
        ranks : numpy.ndarray, list or 'random', optional
            Ranks of the TreeBasedTensorFormat. The default is None, generating
            random ranks between 1 and 5.
        shape : numpy.ndarray, list or 'random', optional
            The size of the spaces of the leaves of the tree. The default is
            None, assigning ranks to shape.
        is_active_node : numpy.ndarray, list or 'random', optional
            Booleans indicating if the node is active. The default is None, for
            which all the nodes are active.

        Returns
        -------
        TreeBasedTensor
            A TreeBasedTensor with the input characteristics.

        '''
        return TreeBasedTensor.create(lambda x: np.random.randn(*x), tree,
                                      ranks, shape, is_active_node)

    @staticmethod
    def rand(tree, ranks=None, shape=None, is_active_node=None):
        '''
        Create a tensor of shape shape and tree-based rank ranks with node
        tensors generated with the method random.rand of numpy.

        Parameters
        ----------
        tree : tensap.FullTensor
            The tensap.FullTensor of the TreeBasedTensor.
        ranks : numpy.ndarray, list or 'random', optional
            Ranks of the TreeBasedTensorFormat. The default is None, generating
            random ranks between 1 and 5.
        shape : numpy.ndarray, list or 'random', optional
            The size of the spaces of the leaves of the tree. The default is
            None, assigning ranks to shape.
        is_active_node : numpy.ndarray, list or 'random', optional
            Booleans indicating if the node is active. The default is None, for
            which all the nodes are active.

        Returns
        -------
        TreeBasedTensor
            A TreeBasedTensor with the input characteristics.

        '''
        return TreeBasedTensor.create(lambda x: np.random.rand(*x), tree,
                                      ranks, shape, is_active_node)

    @staticmethod
    def ones(tree, ranks=None, shape=None, is_active_node=None):
        '''
        Create a tensor of shape shape and tree-based rank ranks with node
        tensors generated with the method ones of numpy.

        Parameters
        ----------
        tree : tensap.FullTensor
            The tensap.FullTensor of the TreeBasedTensor.
        ranks : numpy.ndarray, list or 'random', optional
            Ranks of the TreeBasedTensorFormat. The default is None, generating
            random ranks between 1 and 5.
        shape : numpy.ndarray, list or 'random', optional
            The size of the spaces of the leaves of the tree. The default is
            None, assigning ranks to shape.
        is_active_node : numpy.ndarray, list or 'random', optional
            Booleans indicating if the node is active. The default is None, for
            which all the nodes are active.

        Returns
        -------
        TreeBasedTensor
            A TreeBasedTensor with the input characteristics.

        '''
        return TreeBasedTensor.create(np.ones, tree, ranks, shape,
                                      is_active_node)

    @staticmethod
    def zeros(tree, ranks=None, shape=None, is_active_node=None):
        '''
        Create a tensor of shape shape and tree-based rank ranks with node
        tensors generated with the method zeros of numpy.

        Parameters
        ----------
        tree : tensap.FullTensor
            The tensap.FullTensor of the TreeBasedTensor.
        ranks : numpy.ndarray, list or 'random', optional
            Ranks of the TreeBasedTensorFormat. The default is None, generating
            random ranks between 1 and 5.
        shape : numpy.ndarray, list or 'random', optional
            The size of the spaces of the leaves of the tree. The default is
            None, assigning ranks to shape.
        is_active_node : numpy.ndarray, list or 'random', optional
            Booleans indicating if the node is active. The default is None, for
            which all the nodes are active.

        Returns
        -------
        TreeBasedTensor
            A TreeBasedTensor with the input characteristics.

        '''
        return TreeBasedTensor.create(np.zeros, tree, ranks, shape,
                                      is_active_node)

    @staticmethod
    def tensor_train(cores, dims=None):
        '''
        Create a tree-based tensor with a tensor-train structure.

        Parameters
        ----------
        cores : list or numpy.ndarray
            List of tensap.FullTensor, each associated with one dimension.
        dims : list or numpy.ndarray, optional
            Dimension associated with each core. The default is
            None, indicating that the i-th core is associated with the
            dimension (i-1).

        Returns
        -------
        tensap.TreeBasedTensor
            The tree-based tensor with a tensor-train structure.

        '''
        order = len(cores)
        tree = tensap.DimensionTree.linear(order)
        tensors = np.empty(tree.nb_nodes, dtype='object')
        if dims is not None:
            cores = cores[np.argsort(dims)]

        dim2ind = tree.dim2ind
        nod = dim2ind[0]
        for dim in range(order):
            tensors[nod-1] = cores[dim]
            nod = tree.parent(nod)
        tensors[dim2ind[1:]-1] = tensap.FullTensor([])

        return TreeBasedTensor(tensors, tree)

    @staticmethod
    def _random_is_active_node(tree):
        '''
        Return a random list of active nodes.

        Parameters
        ----------
        tree : tensap.FullTensor
            The considered tensap.FullTensor.

        Returns
        -------
        numpy.ndarray
            List of active nodes.

        '''
        choice = np.random.randint(3)
        if choice == 0:
            is_active_node = np.full(tree.nb_nodes, True)
        elif choice == 1:
            is_active_node = np.logical_not(tree.is_leaf)
        else:
            dim = tree.dim2ind.size
            perm = np.random.permutation(dim)
            perm = perm[:np.random.randint(dim)+1]
            is_active_node = np.full(tree.nb_nodes, True)
            is_active_node[tree.dim2ind[perm]-1] = False
        return is_active_node
