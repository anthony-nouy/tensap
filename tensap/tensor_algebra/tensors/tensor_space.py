import numpy as np


def _promote_spaces(spaces):
    """Validate and promote a list of spaces to 3D arrays."""
    if not isinstance(spaces, (list, tuple)):
        raise TypeError("spaces must be a list or tuple")

    processed = []
    for x in spaces:
        if x.ndim == 3:
            processed.append(x)
        elif x.ndim == 2:
            processed.append(x[:, None, :])
        else:
            raise ValueError(
                "Spaces must be 3D (Operator) or 2D (Vector) arrays."
            )
    return processed


def _update_properties_from_spaces(spaces):
    order = len(spaces)
    dims_out = np.array([x.shape[0] for x in spaces])
    dims_in = np.array([x.shape[1] for x in spaces])
    ranks = np.array([x.shape[2] for x in spaces])
    return order, dims_out, dims_in, ranks


class TSpace:
    """Base class for tensor product spaces.

    Stores a list of d contiguous 3D arrays, each of shape
    ``(N_out, N_in, Rank)``.

    Attributes
    ----------
    spaces : list of numpy.ndarray
    order : int
    dims_out : numpy.ndarray
    dims_in : numpy.ndarray
    ranks : numpy.ndarray
    is_orth : bool
    """

    def __init__(self, spaces, is_orth=False):
        self.spaces = _promote_spaces(spaces)
        self.is_orth = is_orth
        self._update_properties()

    def _update_properties(self):
        (
            self.order,
            self.dims_out,
            self.dims_in,
            self.ranks,
        ) = _update_properties_from_spaces(self.spaces)

    # ---- Common operations ----

    def storage(self):
        """Return the storage complexity."""
        return sum(space.size for space in self.spaces)

    def sparse_storage(self):
        """Return the number of non-zero entries."""
        return sum(np.count_nonzero(s) for s in self.spaces)

    def representation_rank(self):
        """Return the representation rank (dimensions of subspaces)."""
        return self.ranks.copy()

    def cat(self, other):
        """Concatenate the bases of two TSpaces (sum of ranks)."""
        if self.order != other.order:
            raise ValueError("The spaces must have the same order")
        if not np.array_equal(self.dims_out, other.dims_out):
            raise ValueError("The output dimensions must coincide.")
        if not np.array_equal(self.dims_in, other.dims_in):
            raise ValueError("The input dimensions must coincide.")

        new_spaces = [
            np.concatenate((s, o), axis=2)
            for s, o in zip(self.spaces, other.spaces)
        ]
        return self.__class__(new_spaces, is_orth=False)

    def dot(self, other, dims=None):
        """Compute Gram matrices via Frobenius inner product.

        Parameters
        ----------
        other : TSpace
        dims : list of int, optional

        Returns
        -------
        list of numpy.ndarray
            Gram matrices of shape ``(Rank_self, Rank_other)`` per dimension.
        """
        if dims is None:
            dims = range(self.order)
        else:
            dims = np.atleast_1d(dims)

        M = []
        for mu in dims:
            if self.dims_out[mu] != other.dims_out[mu]:
                raise ValueError(
                    f"Output dimensions mismatch at dim {mu}."
                )
            if self.dims_in[mu] != other.dims_in[mu]:
                raise ValueError(
                    f"Input dimensions mismatch at dim {mu}."
                )
            M.append(
                np.einsum(
                    "oix, oiy -> xy",
                    self.spaces[mu],
                    other.spaces[mu],
                )
            )
        return M

    def matrix_times_space(self, matrices, dims=None):
        """Left-multiply bases by matrices: C_{k'} = sum_k M_{k',k} A_k."""
        if dims is None:
            dims = range(self.order)
        else:
            dims = np.atleast_1d(dims)

        if isinstance(matrices, np.ndarray):
            matrices = [matrices]

        if len(matrices) != len(dims):
            raise ValueError(
                "The number of matrices must match the number of "
                "dimensions to transform."
            )

        new_spaces = list(self.spaces)
        for idx, mu in enumerate(dims):
            M = matrices[idx]
            if M.shape[1] != self.ranks[mu]:
                raise ValueError(
                    f"Matrix shape {M.shape} incompatible with "
                    f"rank {self.ranks[mu]} at dim {mu}."
                )
            new_spaces[mu] = np.einsum("oik, pk -> oip",
                                       self.spaces[mu], M)

        return self.__class__(new_spaces, is_orth=False)

    def space_times_matrix(self, matrices, dims=None):
        """Right-multiply bases by matrices: C_{k'} = sum_k A_k M_{k,k'}."""
        if dims is None:
            dims = range(self.order)
        else:
            dims = np.atleast_1d(dims)

        if isinstance(matrices, np.ndarray):
            matrices = [matrices]

        if len(matrices) != len(dims):
            raise ValueError(
                "The number of matrices must match the number of "
                "dimensions to transform."
            )

        new_spaces = list(self.spaces)
        for idx, mu in enumerate(dims):
            M = matrices[idx]
            if M.shape[0] != self.ranks[mu]:
                raise ValueError(
                    f"Matrix shape {M.shape} incompatible with "
                    f"rank {self.ranks[mu]} at dim {mu}."
                )
            new_spaces[mu] = np.einsum("oik, kp -> oip",
                                       self.spaces[mu], M)

        return self.__class__(new_spaces, is_orth=False)

    def eval_in_space(self, dim, coefs):
        """Evaluate a vector/operator from its coefficients on the basis.

        Returns
        -------
        numpy.ndarray of shape (N_out, N_in)
        """
        coefs = np.asarray(coefs)
        if coefs.ndim != 1:
            raise ValueError("The coefficients must be a 1D array.")
        if len(coefs) != self.ranks[dim]:
            raise ValueError(
                f"Expected {self.ranks[dim]} coefficients, "
                f"got {len(coefs)} for dimension {dim}."
            )
        return np.einsum("oik, k -> oi", self.spaces[dim], coefs)

    def orth(self, dims=None):
        """Orthonormalize bases via QR decomposition.

        Returns
        -------
        TSpace
            Space with orthonormalized bases.
        list of numpy.ndarray
            Upper triangular matrices R from QR for each dimension.
        """
        if dims is None:
            dims = range(self.order)
        else:
            dims = np.atleast_1d(dims)

        new_spaces = list(self.spaces)
        R_matrices = []
        for mu in dims:
            n_out, n_in, rank = self.spaces[mu].shape
            mat_2d = self.spaces[mu].reshape(n_out * n_in, rank)
            Q, R = np.linalg.qr(mat_2d)
            new_spaces[mu] = Q.reshape(n_out, n_in, rank)
            R_matrices.append(R)

        is_fully_orth = len(dims) == self.order
        return self.__class__(new_spaces, is_orth=is_fully_orth), R_matrices

    # ---- Dimension manipulation ----

    def permute(self, dims):
        """Reorder dimensions."""
        dims = np.atleast_1d(dims)
        if len(dims) != self.order:
            raise ValueError(
                "permute: dims must have the same length as order."
            )
        new_spaces = [self.spaces[i] for i in dims]
        return self.__class__(new_spaces, is_orth=self.is_orth)

    def keep_space(self, dims):
        """Keep only the specified dimensions."""
        dims = np.atleast_1d(dims)
        new_spaces = [self.spaces[i] for i in dims]
        return self.__class__(new_spaces, is_orth=False)

    def remove_space(self, dims):
        """Remove the specified dimensions."""
        dims = np.atleast_1d(dims)
        keep = [i for i in range(self.order) if i not in dims]
        return self.keep_space(keep)
