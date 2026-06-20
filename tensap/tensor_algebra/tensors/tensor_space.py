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
            np.concatenate((s1, s2), axis=2)
            for s1, s2 in zip(self.spaces, other.spaces)
        ]
        return self.__class__(new_spaces, is_orth=False)

    def dot(self, other, dims=None):
        """For each dimension in dims, compute Gram matrices via Frobenius
        inner product.

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
                    "ijk, ijl -> kl",
                    self.spaces[mu],
                    other.spaces[mu],
                )
            )
        return M

    def matrix_times_space(self, matrices, dims=None):
        """Left-multiply bases along the **physical** dimension.

        For each dimension ``mu``, contracts the output axis (axis 0) of
        ``space[mu]`` with the *second* axis of ``M``::

            new_space[p, i, k] = sum_o M[p, o] * space[o, i, k]

        ── Shape semantics ──────────────────────────────────
        space[mu] shape :  (N_out, N_in, R)
        M          shape :  (K, N_out)
        result     shape :  (K, N_in, R)

        ⇒ **Physical** dim changes (N_out → K), **rank** unchanged (R).

        ── Analogy ─────────────────────────────────────────
        If space[mu] were 2D ``(N_out, R)``, this would be ``M @ space``.
        Typically used to evaluate the tensor against matrices
        (e.g. tensor_matrix_product).

        See also
        --------
        space_times_matrix : right-multiply the *rank* dimension instead.

        Parameters
        ----------
        matrices : list of numpy.ndarray
            Each matrix ``M`` has shape ``(new_phys_dim, old_phys_dim)``.
        dims : list of int, optional
            Dimensions to transform. Defaults to all.

        Returns
        -------
        TSpace
        """
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
            if M.shape[1] != self.dims_out[mu]:
                raise ValueError(
                    f"Matrix shape {M.shape} incompatible with "
                    f"output dimension {self.dims_out[mu]} at dim {mu}."
                )
            new_spaces[mu] = np.einsum("po, oik -> pik",
                                       M, self.spaces[mu])

        return self.__class__(new_spaces, is_orth=False)

    def space_times_matrix(self, matrices, dims=None):
        """Right-multiply bases along the **rank** dimension.

        For each dimension ``mu``, contracts the rank axis (axis2) of
        ``space[mu]`` with the *first* axis of ``M``::

            new_space[o, i, p] = sum_k space[o, i, k] * M[k, p]

        ── Shape semantics ──────────────────────────────────
        space[mu] shape :  (N_out, N_in, R)
        M          shape :  (R, R')
        result     shape :  (N_out, N_in, R')

        ⇒ **Physical** dim unchanged (N_out), **rank** changes (R → R').

        ── Analogy ─────────────────────────────────────────
        If space[mu] were 2D ``(N_out, R)``, this would be ``space @ M``.
        Typically used to change the basis / orthogonalise subspaces
        (e.g. orth, normalize_basis).

        See also
        --------
        matrix_times_space : left-multiply the *physical* dimension instead.
        """
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
        """Orthonormalize bases via QR factorisation.

        For each specified dimension, computes ``X = Q @ R`` and returns
        the space of Q factors together with the transformation
        matrices ``M = R``. The caller should absorb ``M`` into the core::

            new_core @ M  ×  Q  =  old_core × (Q @ R)  =  old_tensor

        Parameters
        ----------
        dims : list of int, optional
            Dimensions to orthogonalize. Defaults to all.

        Returns
        -------
        TSpace
            Space with orthonormalized bases.
        list of numpy.ndarray
            Transformation matrices M (the R factors) for each dim.
        """
        if dims is None:
            dims = range(self.order)
        dims = np.atleast_1d(dims)

        new_spaces = list(self.spaces)
        M_matrices = []
        for mu in dims:
            X = self.spaces[mu]
            N_out, N_in, R = X.shape
            Q, R_fact = np.linalg.qr(X.reshape(N_out * N_in, R), mode="reduced")
            new_spaces[mu] = Q.reshape(N_out, N_in, R)
            M_matrices.append(R_fact)

        result = self.__class__(new_spaces, is_orth=False)
        if len(dims) == self.order:
            result.is_orth = True
        return result, M_matrices

    def truncate(self, dims=None, tol=1e-16):
        """Truncate basis ranks via orth + SVD(R) with tolerance.

        Calls :meth:`orth` to obtain ``(Q_space, R_list)``, then for
        each dimension performs an SVD of ``R`` with energy-based
        truncation to rank ``m``. Returns the truncated space
        ``Q @ U[:, :m]`` and the composite transformation
        ``M = diag(s[:m]) @ Vh[:m, :]`` (where ``R = U @ diag(s) @ Vh``).
        The caller should absorb ``M`` into the core.

        Parameters
        ----------
        dims : list of int, optional
            Dimensions to truncate. Defaults to all.
        tol : float, optional
            Tolerance for rank truncation. Defaults to 1e-16.

        Returns
        -------
        TSpace
            Space with truncated ranks and orthonormalised bases.
        list of numpy.ndarray
            Transformation matrices M for each dimension.
        """
        if dims is None:
            dims = range(self.order)
        dims = np.atleast_1d(dims)

        Q_space, R_list = self.orth(dims)

        tol_sq = tol ** 2
        new_spaces = list(self.spaces)
        M_matrices = []
        for i, mu in enumerate(dims):
            U_r, s, Vh = np.linalg.svd(R_list[i], full_matrices=False)
            s_sq = s ** 2
            total = np.sum(s_sq)
            if total == 0:
                m = 1
            else:
                err_sq = 1 - np.cumsum(s_sq) / total
                m = np.where(err_sq < tol_sq)[0]
                if len(m) == 0:
                    m = len(s)
                else:
                    m = m[0] + 1
            N_out, N_in, _ = self.spaces[mu].shape
            new_Q = Q_space.spaces[mu].reshape(N_out * N_in, -1) @ U_r[:, :m]
            new_spaces[mu] = new_Q.reshape(N_out, N_in, m)
            M_matrices.append(np.diag(s[:m]) @ Vh[:m, :])

        result = self.__class__(new_spaces, is_orth=False)
        if len(dims) == self.order:
            result.is_orth = True
        return result, M_matrices

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
