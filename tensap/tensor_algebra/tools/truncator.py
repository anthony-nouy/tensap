'''
Module truncator.

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

import warnings
import numpy as np
import tensap


class Truncator:
    '''
    Class Truncator.

    Attributes
    ----------
    tolerance : float
        The relative tolerance for the truncation.
    max_rank : int
        The maximum rank for the truncation.
    thresholding_type : str
        The thresholding type ('hard' or 'soft').
    thresholding_parameter : float
        The thresholding parameter.

    '''

    def __init__(self, tolerance=1e-8, max_rank=np.inf,
                 thresholding_type='hard', thresholding_parameter=None):
        '''
        Constructor for the class Truncator.

        Parameters
        ----------
        tolerance : float, optional
            The relative tolerance for the truncation. The default is 1e-8.
        max_rank : int, optional
            The maximum rank for the truncation. The default is numpy.inf.
        thresholding_type : str, optional
            The thresholding type ('hard' or 'soft'). The default is 'hard'.
        thresholding_parameter : float, optional
            The thresholding parameter. The default is None.

        Returns
        -------
        None.

        '''
        self.tolerance = tolerance
        self.max_rank = max_rank
        self.thresholding_type = thresholding_type
        self.thresholding_parameter = thresholding_parameter
        self._hsvd_type = 2  # 1 for root to leaves, 2 for leaves to root

    def truncate(self, tensor):
        '''
        Compute the truncation of the tensor with relative precision
        self.tolerance and maximal rank self.max_rank.

        Parameters
        ----------
        tensor : numpy.ndarray or tensap.FullTensor or tensorflow.Tensor or
        tensap.TreeBasedTensor
            The tensor to truncate.

        Raises
        ------
        NotImplementedError
            If the decomposition is not implemented for the tensor format.
        ValueError
            If the tensor is of order 1.

        Returns
        -------
        out : tensap.CanonicalTensor or tensap.TreeBasedTensor
            The truncated tensor.

        '''

        if not hasattr(tensor, 'order'):
            tensor = tensap.FullTensor(tensor)

        if tensor.order == 2:
            out = self.svd(tensor)
        elif tensor.order > 2:
            if isinstance(tensor, tensap.FullTensor):
                out = self.hosvd(tensor)
            elif isinstance(tensor, tensap.TreeBasedTensor):
                out = self.hsvd(tensor)
            else:
                raise NotImplementedError(
                    'Not implemented with this tensor format.')
        else:
            raise ValueError('Wrong tensor order.')
        return out

    def trunc_svd(self, matrix, tolerance=None, power=2):
        '''
        Compute the truncated svd of the matrix x with relative precision
        self.tolerance (or tolerance if provided) in Schatten p-norm (with p
        given by the input power) and maximal rank self.max_rank.

        Parameters
        ----------
        matrix : numpy.ndarray
            The matrix to truncate.
        tolerance : float, optional
            The relative tolerance for the truncation. The default is
            self.tolerance.
        power : int or 'inf' or float('inf') or numpy.inf, optional
            The integer p of the Schatten-p norm (1 <= p <= inf, p = 2 for
            Frobenius). The default is 2.

        Returns
        -------
        tensap.CanonicalTensor
            The truncated matrix.

        '''
        if tolerance is None:
            tolerance = self.tolerance

        left, sin_val, right = np.linalg.svd(matrix, full_matrices=False)
        if power in ('inf', float('inf'), np.inf):
            error = sin_val / np.max(sin_val)
        else:
            error = np.power(np.flip(np.cumsum(np.flip(np.power(sin_val,
                                                                power)))) /
                             np.sum(np.power(sin_val, power)), 1/power)
        error = np.concatenate((np.atleast_1d(error[1:]), [0]))
        ind = np.nonzero(error < tolerance)[0]
        if ind.size == 0:
            ind = np.min(matrix.shape) + 1
        else:
            ind = np.min(ind) + 1

        ind = np.int(np.min([ind, self.max_rank]))
        if self.thresholding_parameter is not None and \
                self.thresholding_parameter != 0:
            if self.thresholding_type == 'soft':
                sin_val -= self.thresholding_parameter
                ind = np.int(np.min([ind, np.nonzero(sin_val >= 0)[0][-1]+1]))
            elif self.thresholding_type == 'hard':
                ind = np.int(np.min([ind, np.nonzero(
                    sin_val >= self.thresholding_parameter)[0][-1]+1]))
        left = np.atleast_2d(left[:, :ind])
        sin_val = np.atleast_1d(sin_val[:ind])
        right = np.transpose(np.atleast_2d(right[:ind, :]))

        return tensap.CanonicalTensor([left, right], sin_val)

    def svd(self, tensor):
        '''
        Compute the truncated svd of an order-2 tensor.

        Parameters
        ----------
        tensor : numpy.ndarray or tensap.FullTensor or tensorflow.Tensor or
        tensap.TreeBasedTensor
            The tensor to truncate.

        Raises
        ------
        NotImplementedError
            If the method is not implemented.

        Returns
        -------
        out : tensap.CanonicalTensor or tensap.TreeBasedTensor
            The truncated tensor.

        '''
        assert tensor.ndim == 2, 'Wrong order.'

        if isinstance(tensor, np.ndarray):
            out = self.trunc_svd(tensor)
        elif hasattr(tensor, 'numpy'):
            out = self.trunc_svd(tensor.numpy())
        elif isinstance(tensor, tensap.TreeBasedTensor):
            out = self.hsvd(tensor)
        elif isinstance(tensor, tensap.CanonicalTensor):
            tensor = tensor.orth()
            out = self.trunc_svd(tensor.core.data)
            out.space[0] = np.matmul(tensor.space[0], out.space[0])
            out.space[1] = np.matmul(tensor.space[1], out.space[1])
            out.shape = tensor.shape
        else:
            raise NotImplementedError('Method not implemented.')
        return out

    def hosvd(self, tensor):
        '''
        Compute the truncated hosvd of tensor.

        Parameters
        ----------
        tensor : np.ndarray or tensap.FullTensor or tensap.TreeBasedTensor
            The tensor to truncate.

        Raises
        ------
        ValueError
            If the input tensor is of the wrong type.

        Returns
        -------
        out : tensap.CanonicalTensor or tensap.TreeBasedTensor
            The truncated tensor.

        '''
        if isinstance(tensor, np.ndarray):
            tensor = tensap.FullTensor(tensor)

        order = tensor.order
        if order == 2:
            out = self.svd(tensor)
        else:
            max_rank = np.atleast_1d(self.max_rank)
            if max_rank.size == 1:
                max_rank = np.repeat(max_rank, order)
            local_tol = self.tolerance / np.sqrt(order)
            if isinstance(tensor, tensap.FullTensor):
                vec = np.empty(order, dtype=object)
                for dim in range(order):
                    self.max_rank = max_rank[dim]
                    vec[dim] = self.trunc_svd(tensor.matricize(dim).numpy(),
                                              local_tol)
                    vec[dim] = vec[dim].space[0]
                core = tensor.tensor_matrix_product(np.transpose(vec[0]), 0)
                for dim in np.arange(1, order):
                    core = core.tensor_matrix_product(
                        np.transpose(vec[dim]), dim)
                tensors = [core] + [tensap.FullTensor(x) for x in vec]
                tree = tensap.DimensionTree.trivial(order)
                out = tensap.TreeBasedTensor(tensors, tree)
            else:
                raise ValueError('Wrong type.')
        return out

    def hsvd(self, tensor, tree=None, is_active_node=None):
        '''
        Compute the truncated svd in tree-based tensor format of tensor.

        Parameters
        ----------
        tensor : tensap.FullTensor or tensap.TreeBasedTensor
            The tensor to truncate.
        tree : tensap.DimensionTree, optional
            The tree of the output tree-based tensor. The default is None,
            indicating if tensor is a tensap.TreeBasedTensor to take
            tensor.tree.
        is_active_node : numpy.ndarray, optional
            Logical array indicating if the nodes are active.. The default is
            None, indicating if tensor is a tensap.TreeBasedTensor to take
            tensor.is_active_node.

        Raises
        ------
        ValueError
            If the wrong value of the atttribude _hsvd_type is provided.
        NotImplementedError
            If the method is not implemented for the format.

        Returns
        -------
        out : tensap.TreeBasedTensor
            The truncated tensor in tree-based tensor format.

        '''
        if isinstance(tensor, tensap.TreeBasedTensor):
            if tree is not None or is_active_node is not None:
                warnings.warn('The provided tree and/or is_active_node '
                              'are not taken into account when x is a '
                              'tensap.TreeBasedTensor.')
            is_active_node = tensor.is_active_node
            tree = tensor.tree
        elif is_active_node is None:
            is_active_node = np.full(tree.nb_nodes, True)

        max_rank = np.atleast_1d(self.max_rank)
        if max_rank.size == 1:
            max_rank = np.repeat(max_rank, tree.nb_nodes)
            max_rank[tree.root-1] = 1

        local_tol = self.tolerance / np.sqrt(
            np.count_nonzero(is_active_node)-1)

        if isinstance(tensor, tensap.FullTensor):
            root_rank_greater_than_one = tensor.order == len(tree.dim2ind)+1

            tensors = np.empty(tree.nb_nodes, dtype=object)
            shape = np.array(tensor.shape)
            nodes_x = tree.dim2ind
            ranks = np.ones(tree.nb_nodes, dtype=int)

            for level in np.arange(np.max(tree.level), 0, -1):
                for nod in tree.nodes_with_level(level):
                    if is_active_node[nod-1]:
                        if tree.is_leaf[nod-1]:
                            rep = np.nonzero(nod == nodes_x)[0][0]
                        else:
                            children = tree.children(nod)
                            rep = [np.nonzero(np.isin(nodes_x, x))[0][0]
                                   for x in children]
                        rep_c = tensap.fast_setdiff(np.arange(nodes_x.size),
                                                    rep)

                        if root_rank_greater_than_one:
                            rep_c = np.concatenate((rep_c, [tensor.order-1]))

                        self.max_rank = max_rank[nod-1]
                        tmp = self.trunc_svd(tensor.matricize(rep).numpy(),
                                             local_tol)
                        tensors[nod-1] = tmp.space[0]
                        ranks[nod-1] = tensors[nod-1].shape[1]
                        shape_loc = np.hstack((shape[rep], ranks[nod-1]))
                        tensors[nod-1] = tensap.FullTensor(tensors[nod-1],
                                                           shape=shape_loc)
                        tmp = np.matmul(tmp.space[1],
                                        np.diag(tmp.core.data))
                        shape = np.hstack((shape[rep_c], ranks[nod-1]))
                        tensor = tensap.FullTensor(tmp, shape=shape)

                        if root_rank_greater_than_one:
                            perm = np.concatenate((np.arange(tensor.order-2),
                                                   [tensor.order-1],
                                                   [tensor.order-2]))
                            tensor = tensor.transpose(perm)
                            shape = shape[perm]
                            rep_c = rep_c[:-1]

                        nodes_x = np.hstack((nodes_x[rep_c], nod))
                    else:
                        tensors[nod-1] = []

            root_ch = tree.children(tree.root)
            rep = [np.nonzero(np.isin(nodes_x, x))[0][0] for x in root_ch]
            if root_rank_greater_than_one:
                rep = np.concatenate((rep, [tensor.order-1]))
            tensors[tree.root-1] = tensor.transpose(rep)
            out = tensap.TreeBasedTensor(tensors, tree)
        elif isinstance(tensor, tensap.TreeBasedTensor):
            if self._hsvd_type == 1:
                out = tensor.orth()
                gram = out.gramians()[0]
                mat = np.empty(gram.shape, dtype=object)
                shape = np.zeros(gram.shape)
                for nod in range(gram.size):
                    # Truncation of the Gramian in trace norm for a control
                    # of Frobenius norm of the tensor
                    if gram[nod] is not None:
                        self.max_rank = max_rank[nod]
                        tmp = self.trunc_svd(gram[nod], local_tol ** 2)
                        shape[nod] = tmp.core.shape[0]
                        mat[nod] = np.transpose(tmp.space[0])

                # Interior nodes without the root
                for level in np.arange(1, np.max(tree.level)):
                    nod_level = tensap.fast_setdiff(
                        tree.nodes_with_level(level),
                        np.nonzero(tree.is_leaf)[0]+1)
                    for nod in tree.nodes_indices[nod_level-1]:
                        order = out.tensors[nod-1].order
                        out.tensors[nod-1] = \
                            out.tensors[nod-1].tensor_matrix_product(
                                mat[nod-1], order-1)
                        parent = tree.parent(nod)
                        ch_nb = tree.child_number(nod)
                        out.tensors[parent-1] = \
                            out.tensors[parent-1].tensor_matrix_product(
                                mat[nod-1], ch_nb-1)

                # Leaves
                for nod in tree.dim2ind:
                    if out.is_active_node[nod-1]:
                        order = out.tensors[nod-1].order
                        out.tensors[nod-1] = \
                            out.tensors[nod-1].tensor_matrix_product(
                                mat[nod-1], order-1)
                        parent = tree.parent(nod)
                        ch_nb = tree.child_number(nod)
                        out.tensors[parent-1] = \
                            out.tensors[parent-1].tensor_matrix_product(
                                mat[nod-1], ch_nb-1)
                # Update the shape
                out = out.update_attributes()
                out.is_orth = False

            elif self._hsvd_type == 2:
                out = tensor.orth()
                gram = out.gramians()[0]
                for level in np.arange(np.max(tree.level), 0, -1):
                    for nod in tensap.fast_intersect(
                            tree.nodes_with_level(level), out.active_nodes):
                        # Truncation of the Gramian in trace norm for a control
                        # of Frobenius norm of the tensor
                        self.max_rank = max_rank[nod-1]
                        tmp = self.trunc_svd(gram[nod-1], local_tol ** 2)
                        tmp = np.transpose(tmp.space[0])
                        order = out.tensors[nod-1].order
                        out.tensors[nod-1] = \
                            out.tensors[nod-1].tensor_matrix_product(tmp,
                                                                     order-1)
                        parent = tree.parent(nod)
                        ch_nb = tree.child_number(nod)
                        out.tensors[parent-1] = out.tensors[parent-1].\
                            tensor_matrix_product(tmp, ch_nb-1)
                out = out.update_attributes()
                out.is_orth = True
                out.orth_node = tree.root
            else:
                raise ValueError('Wrong value of _hsvd_type.')
        else:
            raise NotImplementedError('Method not implemented.')
        return out

    def ttsvd(self, tensor):
        '''
        Compute the truncated svd in tensor-train format of tensor.

        Parameters
        ----------
        tensor : tensap.FullTensor or tensap.TreeBasedTensor
            The tensor to truncate.

        Returns
        -------
        tensap.TreeBasedTensor
            The truncated tensor in tree-based tensor format with a linear
            tree.

        '''
        tree = tensap.DimensionTree.linear(tensor.order)
        is_active_node = np.full(tree.nb_nodes, True)
        is_active_node[tree.dim2ind[1:]-1] = False

        return self.hsvd(tensor, tree, is_active_node)
