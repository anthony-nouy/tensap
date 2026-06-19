# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Module tucker_like_tensor.
"""

import numpy as np
import tensap


class TuckerLikeTensor:
    """
    Class TuckerLikeTensor.
    Algebraic tensors in Tucker-like tensor format.

    Attributes
    ----------
    core : tensap.FullTensor, tensap.TreeBasedTensor, or tensap.SparseTensor
        The core of the Tucker tensor.
    space : list of numpy.ndarray
        The bases of the subspaces (factor matrices) for each dimension.
    order : int
        The order of the tensor.
    shape : numpy.ndarray
        The shape of the tensor in the full physical space.
    ranks : numpy.ndarray
        The Tucker ranks (the shape of the core tensor).
    is_orth : bool
        Boolean indicating if the representation of the tensor is orthogonal
        (i.e., if the factor matrices in `space` are orthogonal).
    """

    def __init__(self, space, core):
        """
        Constructor for the class TuckerLikeTensor.

        Parameters
        ----------
        space : list of numpy.ndarray
            The tensor space (factor matrices).
        core : tensap.FullTensor, tensap.TreeBasedTensor, etc.
            The core tensor.
        """
        assert isinstance(space, (list, tuple)), "The input space must be a list of arrays."

        # In tensap, if core is given as a numpy array, we cast it to FullTensor
        if isinstance(core, np.ndarray):
            core = tensap.FullTensor(core)

        self.space = list(space)
        self.core = core
        self._update_properties()

    def _update_properties(self):
        """Updates shape, order, and orthogonality flags."""
        self.order = len(self.space)
        self.shape = np.array([x.shape[0] for x in self.space])
        self.ranks = np.array([x.shape[1] for x in self.space])

        # A Tucker tensor is orthogonal if its core is orthogonal and its space is orthonormal
        self.is_orth = self.core.is_orth and self.space.is_orth

    def __repr__(self):
        return (
                "<{} TuckerLikeTensor:\n"
                + "\torder = {},\n"
                + "\tshape = {},\n"
                + "\tranks = {},\n"
                + "\tis_orth = {}>"
        ).format(
            "x".join(map(str, self.shape)),
            self.order,
            self.shape,
            self.ranks,
            self.is_orth,
        )

    def __neg__(self):
        """Return the negative of the tensor."""
        return TuckerLikeTensor(list(self.space), -self.core)

    def full(self):
        """
        Convert the object to a dense tensap.FullTensor.

        Returns
        -------
        tensap.FullTensor
            The Tucker tensor reconstructed as a dense tensor.
        """
        # We rely on the core's ability to contract with matrices (FullTensor does this)
        return self.core.tensor_matrix_product(self.space)

    def numpy(self):
        """
        Convert the TuckerLikeTensor to a dense numpy.ndarray.
        """
        return self.full().numpy()

    def orth(self):
        """
        Orthogonalize the TuckerLikeTensor.

        Performs a QR decomposition on each factor matrix. The R matrices
        are absorbed into the core tensor.

        Returns
        -------
        TuckerLikeTensor
            The orthogonalized tensor.
        """
        qr_decomps = [np.linalg.qr(x) for x in self.space]
        self.space = [q for q, r in qr_decomps]
        M = [r for q, r in qr_decomps]

        # Absorb the R matrices into the core
        self.core = self.core.tensor_matrix_product(M)

        # Optionally orthogonalize the core itself (if the core format supports it)
        if hasattr(self.core, 'orth'):
            self.core = self.core.orth()[0]

        self.is_orth = True
        self.update_properties()
        return self

    def dot(self, tensor_2):
        """
        Return the inner product of two TuckerLikeTensors.

        Parameters
        ----------
        tensor_2 : TuckerLikeTensor

        Returns
        -------
        float
            The inner product.
        """
        assert isinstance(tensor_2, TuckerLikeTensor), "Argument must be a TuckerLikeTensor."
        # M_k = (U_k)^T * V_k
        matrices = [np.matmul(x.T, y) for x, y in zip(self.space, tensor_2.space)]

        # <X, Y> = <Core_X, Core_Y x_1 M_1 ... x_d M_d>
        core_2_projected = tensor_2.core.tensor_matrix_product(matrices)
        return self.core.dot(core_2_projected)

    def norm(self):
        """
        Compute the canonical norm of the TuckerLikeTensor.
        """
        if self.is_orth:
            return self.core.norm()
        else:
            return np.sqrt(np.abs(self.dot(self)))

    def storage(self):
        """
        Return the storage complexity.
        """
        space_storage = sum(x.size for x in self.space)
        return space_storage + self.core.storage()

    def sparse_storage(self):
        """
        Return the sparse storage complexity.
        """
        space_storage = sum(np.count_nonzero(x) for x in self.space)
        return space_storage + self.core.sparse_storage()

    def representation_rank(self):
        """
        Return the representation rank of the tensor.
        """
        return np.prod(self.ranks)

    def tensor_matrix_product(self, matrices, dims=None):
        """
        Contract the Tucker tensor with matrices along the physical dimensions.

        Parameters
        ----------
        matrices : list of numpy.ndarray
        dims : list of int, optional
        """
        if dims is None:
            dims = range(self.order)
        else:
            dims = np.atleast_1d(dims)
            matrices = [matrices] if not isinstance(matrices, list) else matrices

        space = list(self.space)
        for i, dim in enumerate(dims):
            space[dim] = np.matmul(matrices[i], space[dim])

        return TuckerLikeTensor(space, self.core)

    @staticmethod
    def create(generator, ranks, shape):
        """
        Create a TuckerLikeTensor using a given generator.

        Parameters
        ----------
        generator : function
        ranks : list or numpy.ndarray
        shape : list or numpy.ndarray
        """
        space = [generator((s, r)) for s, r in zip(shape, ranks)]
        core = tensap.FullTensor(generator(ranks))
        return TuckerLikeTensor(space, core)

    @staticmethod
    def zeros(ranks, shape):
        return TuckerLikeTensor.create(np.zeros, ranks, shape)

    @staticmethod
    def ones(ranks, shape):
        return TuckerLikeTensor.create(np.ones, ranks, shape)

    @staticmethod
    def rand(ranks, shape):
        return TuckerLikeTensor.create(lambda x: np.random.rand(*x), ranks, shape)

    @staticmethod
    def randn(ranks, shape):
        return TuckerLikeTensor.create(lambda x: np.random.randn(*x), ranks, shape)

    @staticmethod
    def eye(shape):
        """Constructs an identity operator conceptually adapted to Tucker format."""
        ranks = np.ones(len(shape), dtype=int)
        core = tensap.FullTensor(np.ones(ranks))
        space = [np.eye(s) for s in shape]
        return TuckerLikeTensor(space, core)