'''
Module sparse_tensor.

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
from scipy.sparse import lil_matrix, coo_matrix
import tensap


class SparseTensor:
    '''
    Class SparseTensor.

    Attributes
    ----------
    data : numpy.ndarray
        The values of the tensor at the entries in indices.
    indices : tensap.MultiIndices
        The set of multi-indices corresponding to the non-zero coefficients of
        the tensor.
    order : int
        The order of the tensor.
    shape : numpy.ndarray
        The shape of the tensor.

    '''

    def __init__(self, data=None, indices=None, shape=None):
        '''
        Constructor for the class SparseTensor.

        Parameters
        ----------
        data : list or numpy.ndarray, optional
            The values of the tensor at the entries in indices, or the tensor
            itself. The default is None.
        indices : tensap.MultiIndices, optional
            The set of multi-indices corresponding to the non-zero coefficients
            of the tensor. The default is None, indicating to infer it from
            the tensor provided in data.
        shape : list or numpy.ndarray, optional
            The shape of the tensor. The default is None, indicating to infer
            it from the tensor provided in data.

        Raises
        ------
        ValueError
            If the provided arguments are wrong.

        Returns
        -------
        None.

        '''
        if data is None and indices is None and shape is None:
            self.data = []
            self.order = 0
            self.indices = []
            self.shape = []
        elif data is not None and indices is None and shape is None:
            if isinstance(data, (list, np.ndarray)):
                self.order = np.ndim(data)
                self.shape = np.shape(data)

                rep = np.nonzero(data)
                self.data = data[rep]
                self.indices = tensap.MultiIndices(
                    np.hstack([np.reshape(x, [-1, 1]) for x in rep]))
            elif np.all([hasattr(data, x) for x in ['data', 'order',
                                                    'shape', 'indices']]):
                self.data = data.data
                self.order = data.order
                self.shape = data.shape
                self.indices = data.indices
            else:
                raise ValueError('Wrong input arguments.')
        elif data is not None and indices is not None and shape is not None:
            assert isinstance(indices, tensap.MultiIndices), \
                'Argument indices must be a MultiIndices.'

            self.indices = indices
            self.order = indices.ndim()
            self.shape = shape
            self.data = data
            assert np.size(data) == indices.cardinal(), \
                'data and indices must have the same number of elements.'
        else:
            raise ValueError('Wrong input arguments.')
        self.data = np.squeeze(self.data)
        self.shape = np.squeeze(self.shape)

    @property
    def size(self):
        '''
        Compute the size of the tensor. Equivalent to self.storage().

        Returns
        -------
        numpy.ndarray
            The size of the tensor.

        '''
        return np.prod(self.shape)

    def storage(self):
        '''
        Return the storage complexity of the SparseTensor.

        Returns
        -------
        int
            The storage complexity of the SparseTensor.

        '''
        return self.size

    def sparse_storage(self):
        '''
        Return the sparse storage complexity of the SparseTensor.

        Returns
        -------
        int
            The sparse storage complexity of the SparseTensor.

        '''
        return self.count_non_zero()

    def count_non_zero(self):
        '''
        Return the number of non-zero coefficients of the SparseTensor.
        Equivalent to self.sparse_storage().

        Returns
        -------
        int
            The number of non-zero coefficients of the SparseTensor.

        '''
        return self.indices.cardinal()

    @property
    def ndim(self):
        '''
        Compute the order of the tensor. Equivalent to self.order.

        Returns
        -------
        int
            The order of the tensor.

        '''
        return np.size(self.shape)

    def full(self):
        '''
        Convert the SparseTensor to a tensap.FullTensor.

        Returns
        -------
        y : tensap.FullTensor
            The SparseTensor as a tensap.FullTensor.

        '''
        y = tensap.FullTensor(np.zeros(self.shape))
        ind = tuple(self.indices.to_list())
        y.data[ind] = self.data
        return y

    def numpy(self):
        '''
        Convert the SparseTensor to a scipy.sparse.lil.lil_matrix, which can
        be converted to a numpy.matrix using the command todense().

        Returns
        -------
        y : scipy.sparse.lil.lil_matrix
            The SparseTensor as a scipy.sparse.lil.lil_matrix.

        '''
        assert self.ndim <= 2, \
            'nd sparse arrays are not allowed for d > 2.'

        y = lil_matrix(tuple(self.shape), dtype=float)
        ind = tuple(self.indices.to_list())
        y[ind] = self.data
        return y

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
        ind : list of numpy.ndarray
            The indices of the tensor.
        dims : list of numpy.ndarray, optional
            The dimensions associated with the indices. The default is None,
            indicating that indices refers to all the dimensions.

        Returns
        -------
        evaluations : scipy.sparse.lil.lil_matrix
            The evaluations of the tensor.

        '''
        assert dims is None or np.all(dims == np.arange(self.order)), \
            'Method not implemented.'

        if isinstance(ind, tensap.MultiIndices):
            ind = ind.array

        J = self.indices.array
        loc_J, loc_I = np.nonzero(np.all(ind == J[:, np.newaxis], axis=2))
        evaluations = lil_matrix((ind.shape[0], 1), dtype=float)
        evaluations[loc_I, 0] = self.data[loc_J]
        return evaluations

    def squeeze(self, dims):
        '''
        Remove the singleton dimensions of the tensor.

        Parameters
        ----------
        dims : list or numpy.ndarray, optional
            Dimensions to squeeze. The default is None, indicating all the
            singleton dimensions.

        Returns
        -------
        SparseTensor
            The squeezed tensor.

        '''
        raise NotImplementedError('Method not implemented.')

    def __add__(self, y):
        ind = self.indices.add_indices(y.indices)
        _, rep_x, _ = ind.intersect_indices(self.indices)
        _, rep_y, _ = ind.intersect_indices(y.indices)
        data = np.zeros(ind.cardinal())
        data[rep_x] += self.data
        data[rep_y] += y.data
        return SparseTensor(data, ind, self.shape)

    def __neg__(self):
        raise NotImplementedError('Method not implemented.')

    def __sub__(self, y):
        raise NotImplementedError('Method not implemented.')

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
        out : SparseTensor
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

        vectors = [np.ravel(x) for x in vectors]

        out = deepcopy(self)
        not_dims = np.setdiff1d(range(out.order), dims)
        if not_dims.size == 0:
            out.shape = []
        else:
            out.shape = out.shape[not_dims]

        for i in range(dims.size):
            a = out.data * vectors[i][out.indices.array[:, dims[i]]]

            out.indices.array = out.indices.array[:, np.setdiff1d(
                range(out.indices.array.shape[1]), dims[i])]
            out.indices.array = out.indices.array[a != 0, :]

            if out.indices.array.size != 0:
                out.indices.array, ind = np.unique(out.indices.array, axis=0,
                                                   return_inverse=True)

                a = a[a != 0]
                out.data = np.bincount(ind, weights=a)

            dims -= dims > dims[i]

        out.order -= len(vectors)

        if np.size(out.shape) == 0:
            out = out.data

        return out

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
        out : SparseTensor
            The tensor after the contractions with the matrices.

        '''
        if dims is None:
            assert isinstance(matrices, (list, np.ndarray)), \
                'matrices should be a list or a numpy.ndarray.'
            assert len(matrices) == self.order, \
                'len(matrices) must be self.order.'
            dims = range(self.order)
        else:
            dims = np.atleast_1d(dims)
            if not isinstance(matrices, list):
                matrices = [matrices]
            assert len(matrices) == dims.size, \
                'len(matrices) must be equal to dims.size.'

        k = 0
        out = deepcopy(self)
        for mu in dims:
            perm_dims = np.concatenate(
                ([mu], np.setdiff1d(np.arange(out.order), mu)))
            out = out.transpose(perm_dims)
            if out.order == 1:
                out.shape[1] = 1
            ind = tensap.MultiIndices(
                out.indices.array[out.data != 0, :]).sub2ind(out.shape)
            x1, x2 = np.unravel_index(ind,
                                      [out.shape[0], np.prod(out.shape[1:])],
                                      order='F')
            x2u, x2uind = np.unique(x2, return_inverse=True)
            s = coo_matrix((out.data[out.data != 0],
                            (x1, x2uind)),
                           shape=(out.shape[0], np.max(x2uind)+1))
            a = np.transpose(s.transpose().dot(np.transpose(matrices[k])))
            y1, y2 = np.nonzero(a)
            out.shape[0] = matrices[k].shape[0]
            ind = np.ravel_multi_index((y1, np.reshape(x2u[y2], y1.shape)),
                                       (out.shape[0], np.prod(out.shape[1:])),
                                       order='F')
            out.indices = tensap.MultiIndices.ind2sub(out.shape, ind)
            out.data = np.ravel(a[a != 0])
            out = out.itranspose(perm_dims)
            k += 1
        return out

    def tensor_matrix_product_eval_diag(self, matrices):
        '''
        Evaluate the diagonal of a tensor obtained by contraction with
        matrices.

        Parameters
        ----------
        matrices : list
            The matrices to use in the product.

        Returns
        -------
        SparseTensor
            The diagonal of the contractions of the tensor with the matrices.

        '''
        y = matrices[0][:, self.indices.array[:, 0]]
        for k in np.arange(1, self.order):
            y *= matrices[k][:, self.indices.array[:, k]]
        return np.matmul(y, self.data)

    def transpose(self, dims):
        '''
        Transpose (permute) the dimensions of the tensor.

        Parameters
        ----------
        dims : list or numpy.ndarray
            The new ordering of the dimensions.

        Returns
        -------
        out : SparseTensor
            The transposed (permuted) tensor.

        '''
        out = deepcopy(self)
        out.indices.array = out.indices.array[:, dims]
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
        SparseTensor
            The transposed (permuted) tensor.

        '''
        out = deepcopy(self)
        return out.transpose(np.argsort(dims))

    def reshape(self, shape):
        '''
        Reshape the tensor.

        Parameters
        ----------
        shape : list or numpy.ndarray
            The new shape of the tensor.

        Returns
        -------
        tensor : SparseTensor
            The reshaped tensor.

        '''
        shape = np.array(shape)
        ind = self.indices.sub2ind(self.shape)
        out = deepcopy(self)
        out.indices = tensap.MultiIndices.ind2sub(shape, ind)
        out.shape = shape
        out.order = shape.size
        return out

    def tensor_diagonal_matrix_product(self, matrices, dims=None):
        '''
        Contract a SparseTensor with matrices built from their diagonals.

        The second dimension of the matrix matrices[k] is contracted with the
        k-th dimension of self, with the indices k given in dims (if provided).

        FIXME: not optimal, does not exploit sparsity.

        Parameters
        ----------
        matrices : numpy.ndarray or list of numpy.ndarray
            The diagonals of the matrices to use in the product.
        dims : list or numpy.ndarray, optional
            Indices of the contractions. The default is None, indicating all
            the dimensions.

        Returns
        -------
        SparseTensor
            The tensor after the contractions with the matrices.

        '''
        if not isinstance(matrices, list):
            matrices = [matrices[:, i] for i in range(np.shape(matrices)[1])]

        if dims is None:
            assert len(matrices) == self.order, \
                'len(matrices) must be self.order.'
            dims = range(self.order)
        else:
            dims = np.array(dims)
            assert len(matrices) == dims.size, \
                'len(matrices) must be equal to dims.size.'

        matrices = [tensap.FullTensor(np.diag(np.reshape(x, [-1])))
                    for x in matrices]
        return self.tensor_matrix_product(matrices, dims)

    def dot(self, y):
        '''
        Return the inner product of two tensors.

        Parameters
        ----------
        y : tensap.Tensor
            The second tensor of the inner products. Must be convertible to
            a SparseTensor.

        Returns
        -------
        numpy.float
            The inner product of the two tensors.

        '''
        if not isinstance(y, SparseTensor):
            try:
                y = y.sparse()
            except Exception:
                raise ValueError('Cannot convert input to SparseTensor.')

        _, ind_x, ind_y = self.indices.intersect_indices(y.indices)
        return np.sum(self.data[ind_x] * y.data[ind_y])

    def __mul__(self, y):
        if not isinstance(y, SparseTensor):
            try:
                y = y.sparse()
            except Exception:
                raise ValueError('Cannot convert input to SparseTensor.')

        _, ind_x, ind_y = self.indices.intersect_indices(y.indices)
        out = deepcopy(self)
        out.data = self.data[ind_x] * y.data[ind_y]
        out.indices.array = self.indices.array[ind_x, :]
        return out

    def norm(self):
        '''
        Compute the canonical norm of the SparseTensor.

        Returns
        -------
        numpy.float
            The norm of the tensor.

        '''
        return np.sqrt(self.dot(self))

    def orth(self, dim):
        '''
        Orthogonalize the tensor.

        Parameters
        ----------
        dim : int
            The dimension of the orthogonal dim-matricization of self.

        Returns
        -------
        SparseTensor
            A tensor whose dim-matricization is an orthogonal matrix
            corresponding to the Q factor of a QR factorization of the
            dim-matricization of self.
        r_matrix : numpy.ndarray
            The R factor.

        '''
        raise NotImplementedError('Method not implemented.')

    def cat(self, y, dims):
        '''
        Concatenate the tensors.

        Concatenates self and y in a tensor z such that:
        z(i_1 ,..., i_d) = x(i_1, ..., i_d) if i_k <= sz[k]-1 for k in dims,
        z(i_1, ..., i_d) = y(i_1-sz[0], ..., i_d-sz[d-1]) if i_k >= sz[k]
        for k in dims,
        z(i_1, ..., i_d) = 0 otherwise, with sz = self.shape and
        dims = range(self.order) if not provided.

        Parameters
        ----------
        y : Tensor
            The second tensor to be concatenaed.
        dims : list or numpy.ndarray, optional
            The dimensions of the concatenation. The default is None,
            indicating all the dimensions.
        Returns
        -------
        SparseTensor
            The concatenated tensors.

        '''
        raise NotImplementedError('Method not implemented.')

    def kron(self, y):
        '''
        Kronecker product of tensors.

        Similar to numpy.kron but for sparse tensors.

        Parameters
        ----------
        y : Tensor
            The second tensor of the Kronecker product.

        Returns
        -------
        SparseTensor
            The tensor resulting from the Kronecker product.

        '''
        raise NotImplementedError('Method not implemented.')

    def dot_with_rank_one_metric(self, y, M):
        '''
        Compute the weighted inner product of two tensors.

        Compute the weighted canonical inner product of self and y,
        where the inner product related to dimension k is weighted by
        M[k]. It is equivalent to
        self.dot(y.tensor_matrix_product(M)),
        but can be much faster.

        Parameters
        ----------
        y : Tensor
            The second tensor of the inner product.
        M : list or numpy.ndarray or FullTensor
            The weight matrix.

        Returns
        -------
        numpy.float
            The weighted inner product.

        '''
        s = y.tensor_matrix_product(M)
        return self.dot(s)

    def tensordot_matrix_product_except_dim(self, y, M, dim):
        '''
        Particular type of contraction.

        Compute a special contraction of two tensors self, y, a list of
        matrices M and a particular dimension dim. Note that dim must
        be a scalar, while M must be a list array with x.self.order
        elements.

        Parameters
        ----------
        y : Tensor
            The second tensor of the contraction.
        M : list
            The list of matrices of the contraction.
        dim : int
            The excluded dimension.

        Returns
        -------
        numpy.ndarray
            The result of the contraction.

        '''
        # dims = np.setdiff1d(np.arange(self.order), dim)
        # s = y.tensor_matrix_product(M[dims], dims)
        # return self.tensordot(s, dims, dims)
        raise NotImplementedError('Method not implemented.')

    def eval_diag(self, dims=None):
        '''
        Extract the diagonal of the tensor.

        The tensor must be such that self.shape[mu] = n for all mu (in dims if
        provided).

        Parameters
        ----------
        dims : list of numpy.ndarray, optional
            The dimensions associated with the indices of the diagonal. The
            default is None,indicating that indices refers to all the
            dimensions.

        Returns
        -------
        data : numpy.ndarray
            The evaluations of the diagonal of the tensor.

        '''
        if dims is None:
            dims = np.arange(self.order)
        else:
            dims = np.atleast_1d(dims)

        if dims.size == 1:
            data = self
        else:
            assert np.all([self.shape[x] == self.shape[dims[0]] for
                           x in dims]),\
             'The shapes of the tensor in dimensions dims should be equal.'
            ind = np.repeat(np.reshape(np.arange(self.shape[0]), [-1, 1]),
                            dims.size, 1)
            data = self.eval_at_indices(ind, dims)
        return data

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
        SparseTensor
            The subtensor.

        '''
        raise NotImplementedError('Method not implemented.')
