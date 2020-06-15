'''
Module diagonal_tensor.

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


class DiagonalTensor:
    '''
    Class DiagonalTensor.

    Attributes
    ----------
    data : numpy.ndarray
        The diagonal entries of the tensor.
    order : int
        The order of the tensor.
    shape : numpy.ndarray
        The shape of the tensor.
    is_orth : bool
        Boolean indicating if the representation of the tensor is orthogonal
        (i.e. one mu-matricization is orthogonal).

    '''

    def __init__(self, data, order=None):
        '''
        Constructor for the class DiagonalTensor.

        Parameters
        ----------
        data : numpy.ndarray or tensap.DiagonalTensor
            The diagonal entries of the tensor.
        order : int, optional
            The order of the tensor. The default is None.

        Returns
        -------
        None.

        '''
        self.is_orth = True

        if order is None:
            assert isinstance(data, DiagonalTensor), \
                'The input must be a DiagonalTensor.'
            self.data = np.array(data.data)
            self.order = data.order
            self.shape = data.shape
        else:
            assert isinstance(data, (list, np.ndarray)), \
                'The input must be a list or numpy.ndarray.'
            self.data = np.reshape(data, -1)
            self.order = order
            self.shape = np.full(order, self.data.size)

    def __repr__(self):
        return ('<{} DiagonalTensor:{n}' +
                '{t}order = {},{n}' +
                '{t}shape = {},{n}' +
                '{t}is_orth = {}>').format('x'.join(map(str, self.shape)),
                                           self.order,
                                           self.shape,
                                           self.is_orth,
                                           t='\t', n='\n')

    def tree_based_tensor(self, tree=None, is_active_node=None):
        '''
        Convert the tensap.DiagonalTensor into a tensap.TreeBasedTensor.

        Parameters
        ----------
        tree : tensap.DimensionTree, optional
            The tree associated with the tree-based tensor representation. The
            default is a linear tree.
        is_active_node : list or numpy.ndarray, optional
            List or array of booleans indicating if each node of the tree is
            active. The default is True for all nodes except the leaves.

        Raises
        ------
        ValueError
            If the internal nodes are not all active.

        Returns
        -------
        tensap.TreeBasedTensor
            A tree-based tensor representation of the diagonal tensor.

        '''
        if tree is None:
            tree = tensap.DimensionTree.linear(self.order)

        if is_active_node is None:
            is_active_node = np.full(tree.nb_nodes, True)
            is_active_node[tree.is_leaf] = False

        tensors = np.empty(tree.nb_nodes, dtype=object)
        tensors[np.logical_not(is_active_node)] = tensap.FullTensor([])
        r = self.shape[0]
        for nod in np.arange(1, tree.nb_nodes+1):
            ch = tree.children(nod)
            if tree.parent(nod) == 0:
                tensors[nod-1] = tensap.FullTensor.diag(self.data, ch.size)
            elif tree.is_leaf[nod-1] and is_active_node[nod-1]:
                tensors[nod-1] = tensap.FullTensor(np.eye(r), 2, [r, r])
            elif is_active_node[nod-1]:
                tensors[nod-1] = tensap.FullTensor.diag(np.ones(r), ch.size+1)
            elif not tree.is_leaf[nod-1] and not is_active_node[nod-1]:
                raise ValueError('The internal nodes should be active.')
        return tensap.TreeBasedTensor(tensors, tree)

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

    def storage(self):
        '''
        Return the storage complexity of the DiagonalTensor.

        Returns
        -------
        int
            The storage complexity of the DiagonalTensor.

        '''
        return self.data.size

    def sparse_storage(self):
        '''
        Return the sparse storage complexity of the DiagonalTensor.

        Returns
        -------
        int
            The sparse storage complexity of the DiagonalTensor.

        '''
        return np.count_nonzero(self.data)

    def __add__(self, y):
        return DiagonalTensor(self.data + y.data, self.order)

    def __neg__(self):
        return DiagonalTensor(-self.data, self.order)

    def __sub__(self, y):
        return DiagonalTensor(self.data - y.data, self.order)

    def __mul__(self, y):
        return DiagonalTensor(self.data * y.data, self.order)

    def reshape(self, shape):
        '''
        Reshape the tensor. The method has no effet.

        Parameters
        ----------
        shape : list or numpy.ndarray
            The new shape of the tensor.

        Returns
        -------
        tensor : DiagonalTensor
            The reshaped tensor.

        '''
        out = deepcopy(self)
        out.shape = np.ravel(shape)
        return out

    def sub_tensor(self, *args):
        '''
        Extract a subtensor of the tensor.

        See also tensap.FullTensor.sub_tensor.

        Parameters
        ----------
        *indices : list
            The indices to extract in each dimension. ':' indicates all the
            indices.

        Returns
        -------
        FullTensor
            The subtensor.

        '''

        out = self.full()
        return out.sub_tensor(*args)

    def update_attributes(self):
        '''
        Update the attribute shape of self if data or order have been modified.

        Returns
        -------
        None.

        '''
        self.shape = np.full(self.order, self.data.size)

    def tensor_vector_product(self, vectors):
        '''
        Compute the contraction of the tensor with vectors.

        Compute the contraction of self with each vector contained in the list
        vectors along all the dimensions. The operation is such that V[k] is
        contracted with the k-th dimension of self.

        Parameters
        ----------
        vectors : numpy.ndarray or list of numpy.ndarray
            The vectors to use in the product.

        Returns
        -------
        DiagonalTensor
            The tensor after the contractions with the vectors.

        '''
        if isinstance(vectors, list):
            vectors = np.hstack([np.reshape(x, [-1, 1]) for x in vectors])
        data = self.data * np.prod(vectors, 1)
        order = self.order - vectors.shape[1]
        if order == 0:
            return np.sum(data)
        return DiagonalTensor(data, order)

    def tensor_matrix_product(self, matrices, dims=None):
        '''
        Contract a tensor with matrices.

        The second dimension of the matrix matrices[k] is contracted with the
        k-th dimension of self, with the indices k given in dims (if provided).

        See also tensap.FullTensor.tensor_matrix_product.

        Parameters
        ----------
        matrices : numpy.ndarray or list of numpy.ndarray
            The matrices to use in the product.
        dims : list or numpy.ndarray, optional
            Indices of the contractions. The default is None, indicating all
            the dimensions.

        Returns
        -------
        FullTensor
            The tensor after the contractions with the matrices.

        '''
        out = self.full()
        return out.tensor_matrix_product(matrices, dims)

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
        tensap.DiagonalTensor or numpy.ndarray
            The diagonal of the contractions of the tensor with the matrices.

        '''
        return self.tensor_matrix_product(matrices, dims).eval_diag(dims)

    def tensor_diagonal_matrix_product(self, matrices):
        '''
        Contract a FullTensor with matrices built from their diagonals.

        The second dimension of the matrix matrices[k] is contracted with the
        k-th dimension of self.

        Parameters
        ----------
        matrices : numpy.ndarray or list of numpy.ndarray
            The diagonals of the matrices to use in the product.

        Returns
        -------
        DiagonalTensor
            The tensor after the contractions with the matrices.

        '''
        if isinstance(matrices, list):
            matrices = np.hstack([np.reshape(x, [-1, 1]) for x in matrices])
        return DiagonalTensor(np.prod(matrices, 1) * self.data, self.order)

    def tensordot(self, y, dims1, dims2=None):
        '''
        Contract two tensors along specified dimensions.

        See also tensap.FullTensor.tensordot.

        Parameters
        ----------
        tensor2 : Tensor
            The second tensor of the contraction.
        dims1 : list or int
            The dimensions of contractions for the first tensor.
        dims2 : list or int, optional
            The dimensions of contractions for the second tensor. The default
            is None which indicates, if dims1 = 0, to perform the outer
            product of the two tensors, similarly to tensorflow.tensordot.

        Returns
        -------
        out : FullTensor
            The resulting tensor.

        '''
        x = self.full()
        y = y.full()
        return x.tensordot(y, dims1, dims2)

    def dot(self, y):
        '''
        Return the inner product of two tensors.

        Parameters
        ----------
        tensor2 : DiagonalTensor
            The second tensor of the inner products.

        Returns
        -------
        numpy.float
            The inner product of the two tensors.

        '''
        return np.sum(self.data*y.data)

    def norm(self):
        '''
        Compute the canonical norm of the DiagonalTensor.

        Returns
        -------
        numpy.float
            The norm of the tensor.

        '''
        return np.linalg.norm(self.data)

    def full(self):
        '''
        Convert the DiagonalTensor to a tensap.FullTensor.

        Returns
        -------
        tensap.FullTensor
            The DiagonalTensor as a tensap.FullTensor.

        '''
        return tensap.FullTensor.diag(self.data, self.order)

    def numpy(self):
        '''
        Convert the DiagonalTensor to a numpy.ndarray.

        Returns
        -------
        numpy.ndarray
            The DiagonalTensor as a numpy.ndarray.

        '''
        return self.full().data

    def sparse(self):
        '''
        Convert the DiagonalTensor to a tensap.SparseTensor.

        Returns
        -------
        tensap.SparseTensor
            The DiagonalTensor as a tensap.SparseTensor.

        '''
        ind = np.nonzero(self.data)[0]
        data = self.data[self.data != 0]
        indices = tensap.MultiIndices(np.tile(np.reshape(ind, [-1, 1]),
                                              (1, self.order)))
        return tensap.SparseTensor(data, indices, self.shape)

    def cat(self, y):
        '''
        Concatenate the tensors.

        Concatenates self and y in a tensor z such that:
        z(i_1 ,..., i_d) = x(i_1, ..., i_d) if i_k <= sz[k]-1 for k in dims,
        z(i_1, ..., i_d) = y(i_1-sz[0], ..., i_d-sz[d-1]) if i_k >= sz[k]
        for k in dims,
        z(i_1, ..., i_d) = 0 otherwise, with sz = self.shape and
        dims = range(self.order).

        Parameters
        ----------
        y : DiagonalTensor
            The second tensor to be concatenaed.
        Returns
        -------
        DiagonalTensor
            The concatenated tensors.

        '''
        data = np.concatenate((self.data, y.data))
        return DiagonalTensor(data, order=self.order)

    def kron(self, y):
        '''
        Kronecker product of tensors.

        Similar to numpy.kron but for arbitrary tensors.

        Parameters
        ----------
        tensor2 : DiagonalTensor
            The second tensor of the Kronecker product.

        Returns
        -------
        DiagonalTensor
            The tensor resulting from the Kronecker product.

        '''
        data = np.reshape(np.outer(y.data, self.data), -1, order='F')
        return DiagonalTensor(data, order=self.order)

    def dot_with_rank_one_metric(self, y, matrix):
        '''
        Compute the weighted inner product of two tensors.

        Compute the weighted canonical inner product of self and y,
        where the inner product related to dimension k is weighted by
        matrix[k]. It is equivalent to
        self.dot(y.tensor_matrix_product(matrix)), but can be much faster.

        Parameters
        ----------
        y : DiagonalTensor
            The second tensor of the inner product.
        matrix : list or numpy.ndarray or FullTensor
            The weight matrix.

        Returns
        -------
        numpy.float
            The weighted inner product.

        '''
        if isinstance(matrix, list):
            matrix = np.hstack([np.reshape(x, [-1, 1]) for x in matrix])
        matrix = np.reshape(np.expand_dims(matrix, 1),
                            (self.shape[0], y.shape[0], self.order), order='F')
        matrix = np.prod(matrix, 2)
        return np.matmul(np.transpose(self.data), np.matmul(matrix, y.data))

    def tensordot_matrix_product_except_dim(self, y, matrices, dim):
        '''
        Particular type of contraction.

        Compute a special contraction of two tensors self, y, a list of
        matrices matrices and a particular dimension dim. Note that dim must
        be a scalar, while matrices must be a list array with self.order
        elements.

        Parameters
        ----------
        y : DiagonalTensor
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
        ind = np.setdiff1d(range(self.order), dim)
        matrices = np.hstack([matrices[x] for x in ind])
        matrices = np.reshape(np.expand_dims(matrices, 1),
                              (self.shape[0], y.shape[0], ind.size),
                              order='F')
        matrices = np.prod(matrices, 2)
        return matrices * np.outer(self.data, y.data)

    def orth(self):
        '''
        Placeholder method returning a copy of self.

        Returns
        -------
        DiagonalTensor
            A copy of self.

        '''
        return deepcopy(self)

    def eval_diag(self, dims=None):
        '''
        Extract the diagonal of the tensor.

        Parameters
        ----------
        dims : list of numpy.ndarray, optional
            The dimensions associated with the indices of the diagonal. The
            default is None,indicating that the indices refer to all the
            dimensions.

        Returns
        -------
        data : DiagonalTensor or numpy.ndarray
            The evaluations of the diagonal of the tensor.

        '''
        if dims is None or np.size(dims) == self.order:
            out = np.array(self.data)
        else:
            dims = np.sort(np.atleast_1d(dims))
            rep = np.concatenate((np.arange(dims[0]+1),
                                  np.arange(dims[-1]+1, self.order)))
            out = deepcopy(self)
            out.shape = out.shape[rep]
            out.order = np.size(rep)
        return out

    def eval_at_indices(self, ind, dims=None):
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
        evaluations : numpy.ndarray or DiagonalTensor
            The evaluations of the tensor.

        '''

        ind = np.atleast_2d(ind)
        r = np.full(ind.shape[0], True)
        for k in range(ind.shape[1]):
            r = np.logical_and(r, ind[:, 0] == ind[:, k])

        if dims is None or np.size(dims) == self.order:
            out = np.zeros(ind.shape[0])
            out[r] = self.data[ind[r, 0]]
            return out
        raise NotImplementedError('Method not implemented.')

    def transpose(self, dims):
        '''
        Transpose (permute) the dimensions of the tensor.

        Parameters
        ----------
        dims : list or numpy.ndarray
            The new ordering of the dimensions.

        Returns
        -------
        tensor : DiagonalTensor
            The transposed (permuted) tensor.

        '''
        out = deepcopy(self)
        out.shape = out.shape[dims]
        return out

    def itranspose(self, dims):
        '''
        Return the inverse transpose (permutation) of the dimensions of the
        tensor.

        Parameters
        ----------
        dims : list or numpy.ndarray
            The original transpose (permutation) indices.

        Returns
        -------
        DiagonalTensor
            The transposed (permuted) tensor.

        '''
        return self.transpose(np.argsort(dims))

    @staticmethod
    def create(generator, rank, order):
        '''
        Create a DiagonalTensor of rank rank and order order using a given
        generator.

        Parameters
        ----------
        generator : function
            Function generating an array, given a rank.
        rank : int
            The rank of the tensor.
        order : int
            The order of the tensor.

        Returns
        -------
        DiagonalTensor
            The created tensor.

        '''
        return DiagonalTensor(generator(rank), order)

    @staticmethod
    def rand(rank, order):
        '''
        Create a DiagonalTensor of rank rank and order order with i.i.d.
        entries drawn according to the uniform distribution on [0, 1].

        Parameters
        ----------
        rank : int
            The rank of the tensor.
        order : int
            The order of the tensor.

        Returns
        -------
        DiagonalTensor
            The created tensor.

        '''
        return DiagonalTensor.create(np.random.rand, rank, order)

    @staticmethod
    def randn(rank, order):
        '''
        Create a DiagonalTensor of rank rank and order order with i.i.d.
        entries drawn according to the standard gaussian distribution.

        Parameters
        ----------
        rank : int
            The rank of the tensor.
        order : int
            The order of the tensor.

        Returns
        -------
        DiagonalTensor
            The created tensor.

        '''
        return DiagonalTensor.create(np.random.randn, rank, order)

    @staticmethod
    def zeros(rank, order):
        '''
        Create a DiagonalTensor of rank rank and order order with with entries
        equal to 0.

        Parameters
        ----------
        rank : int
            The rank of the tensor.
        order : int
            The order of the tensor.

        Returns
        -------
        DiagonalTensor
            The created tensor.

        '''
        return DiagonalTensor.create(np.zeros, rank, order)

    @staticmethod
    def ones(rank, order):
        '''
        Create a DiagonalTensor of rank rank and order order with with
        entries equal to 1.

        Parameters
        ----------
        rank : int
            The rank of the tensor.
        order : int
            The order of the tensor.

        Returns
        -------
        DiagonalTensor
            The created tensor.

        '''
        return DiagonalTensor.create(np.ones, rank, order)
