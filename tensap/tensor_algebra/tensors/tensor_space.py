import numpy as np

class TSpace:
    """
    Tensor Space of Vectors or Operators.
    Encapsulate a list of d contiguous 3D arrays representing operators.
    Each array has shape (N_out, N_in, Rank).

    Attributes
    - spaces
    - order
    - dims_out
    - dims_in
    - is_orth
    """

    def __init__(self, spaces, is_orth=False):
        if not isinstance(spaces, (list, tuple)):
            raise TypeError("spaces must be a list or tuple")

        processed_spaces = []
        for x in spaces:
            if x.ndim == 3:
                processed_spaces.append(x)
            elif x.ndim == 2:
                processed_spaces.append(x[:, None, :])  # Promotion en 3D
            else:
                raise ValueError("Spaces must be 3D (Operator)"
                                 " or 2D (Vector) arrays.")

        self.spaces = processed_spaces
        self.is_orth = is_orth
        self._update_properties()

        is_vector = np.all(self.dims_in == 1)
        is_operator = np.all(self.dims_in > 1)
        if not (is_vector or is_operator):
            raise ValueError("All input dimension must be 1 (vectors)"
                             " or all be > 1 (operator)")

    def _update_properties(self):
        self.order = len(self.spaces)
        self.dims_out = np.array([x.shape[0] for x in self.spaces])
        self.dims_in = np.array([x.shape[1] for x in self.spaces])
        self.ranks = np.array([x.shape[2] for x in self.spaces])

    def storage(self):
        """return the storage complexity"""
        return sum(space.size for space in self.spaces)

    def sparse_storage(self):
        raise NotImplementedError("TSpaceOperator.sparse_storage not implemented")

    def cat(self, other_tspace):
        """
        Concatenates the elements of two TSpaceOperators.
        Increases the rank of the space by combining the operators.
        The results representation ranks are the sum of the two TSpaceOperators

        Parameters
        ----------
        other_tspace : TSpace
            Another tensor space of operators of the same order and physical size.

        Returns
        -------
        TSpace
            A new TSpaceOperator with concatenated spaces.
        """
        if self.order != other_tspace.order:
            raise ValueError("The spaces must have the same order")

        if not np.array_equal(self.dims_out, other_tspace.dims_out):
            raise ValueError("The output dimensions must coincide.")
        if not np.array_equal(self.dims_in, other_tspace.dims_in):
            raise ValueError("The input dimensions must coincide.")

        new_spaces = []
        for mu in range(self.order):
            # self.spaces[mu]        : (N_out, N_in, R1_nu)
            # other_space.spaces[mu] : (N_out, N_in, R2_nu)
            # Expected result        : (N_out, N_in, R1_nu + R2_nu)
            concatenated_space = np.concatenate(
                (self.spaces[mu], other_tspace.spaces[mu]),
                axis=2)
            new_spaces.append(concatenated_space)

        return TSpace(new_spaces, is_orth=False)

    def representation_rank(self):
        """Return the representation rank (dimensions of subspaces)"""
        return self.ranks

    def mtimes(self, other_space, dims=None):
        """
        Multiplication of two Tensor Spaces
        - T-space of operators * T-space of operators
        - T-space of operators * T-space of vectors

        Parameters
        ----------
        other_space : TSpace
            The "left" space in the multiplication.
            - T-space of operators: (Dim out, Dim in, Rank)
            - T-space of vectors: (Dim out, 1, Rank)
        dims : list of int, optional
            The dimensions which has to be multiplied. All by default
        """
        if dims is None:
            dims = range(self.order)
        else:
            dims = np.atleast_1d(dims)

        for mu in dims:
            if self.dims_in[mu] != other_space.dims_out[mu]:
                raise ValueError(
                    f"Dimension mismatch at dim {mu}: "
                    f"self.dims_in[{mu}]={self.dims_in[mu]} != "
                    f"other.dims_out[{mu}]={other_space.dims_out[mu]}. "
                    "For mtimes, self must be an operator space and "
                    "dims_in(self) must equal dims_out(other).")

        new_spaces = []
        for mu in dims:
            # self.spaces[mu]        : (N1_out, N1_in, R1)
            # other_space.spaces[mu] : (N2_out, N2_in, R2)
            #       -> with N1_in = N2_out
            #       -> N2_in = 1 (vector) or not (operator)

            # 1. product grid of ranks (R1, R2)
            # o = out, k = inter, i = in, x = rank_self, y = rank_other
            res = np.einsum('okx, kiy -> oixy',
                            self.spaces[mu],
                            other_space.spaces[mu])
            # 2. lexicographic order: flatten the rank grid
            new_spaces.append(res.reshape(res.shape[0], res.shape[1], -1))

        return TSpace(new_spaces, is_orth=False)

    def dot(self, other_space, dims=None):
        """
        Computes the Gram matrices of the inner product between two TSpaces,
        using the Frobenius inner product over the physical dimensions.

        Parameters
        ----------
        other_space : TSpace
            The other tensor space to compute the inner product with.
        dims : list of int, optional
            The dimensions to compute the dot product for. All by default.

        Returns
        -------
        list of numpy.ndarray
            A list of Gram matrices of shape (Rank_self, Rank_other) for each
             dimension specified in dims.
        """
        if dims is None:
            dims = range(self.order)
        else:
            dims = np.atleast_1d(dims)

        M = []
        for mu in dims:
            if self.dims_out[mu] != other_space.dims_out[mu]:
                raise ValueError(
                    f"Output dimensions mismatch at dim {mu}.")
            if self.dims_in[mu] != other_space.dims_in[mu]:
                raise ValueError(
                    f"Input dimensions mismatch at dim {mu}.")

            # self.spaces[mu]        : (N_out, N_in, R1)
            # other_space.spaces[mu] : (N_out, N_in, R2)
            # Compute the Gram matrix of size: (R1, R2)

            # o = out, i = in, x = rank_self, y = rank_other
            metric = np.einsum('oix, oiy -> xy',
                               self.spaces[mu],
                               other_space.spaces[mu])

            M.append(metric)

        return M

    def matrix_times_space(self, matrices, dims=None):
        """
        Applies a linear transformation to the basis components of the spaces.

        Given a list of matrices M, computes the new space components C such
        that: C_{k'} = sum_k M_{k', k} A_{k}

        Parameters
        ----------
        matrices : list of numpy.ndarray or numpy.ndarray
            A single matrix or a list of matrices (one for each dimension in dims).
            Each matrix M must be of shape (R_new, R_old), where R_old is the
            current rank of the space at that dimension.
        dims : list of int, optional
            The dimensions to apply the transformation to. All by default.

        Returns
        -------
        TSpace
            A new TSpace object with the transformed components.
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
                "dimensions to transform.")

        new_spaces = list(self.spaces)

        for idx, mu in enumerate(dims):
            M = matrices[idx]

            if M.shape[1] != self.ranks[mu]:
                raise ValueError(
                    f"Matrix shape {M.shape} incompatible with "
                    f"rank {self.ranks[mu]} at dim {mu}.")

            # self.spaces[mu] : (N_out, N_in, R_old)
            # M               : (R_new, R_old)
            # Result          : (N_out, N_in, R_new)

            # o = out, i = in, k = R_old, p = R_new
            new_spaces[mu] = np.einsum('oik, pk -> oip',
                                       self.spaces[mu], M)

        return TSpace(new_spaces, is_orth=False)

    def space_times_matrix(self, matrices, dims=None):
        """
        Applies a linear transformation to the basis components of the spaces
        (Right-multiplication).

        Given a matrix M of shape (R_old, R_new), computes the new space
        components C such that: C_{k'} = sum_k A_{k} M_{k, k'}

        This is mathematically equivalent to matrix_times_space with the
        transposed matrix.
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
                "dimensions to transform.")

        new_spaces = list(self.spaces)

        for idx, mu in enumerate(dims):
            M = matrices[idx]

            if M.shape[0] != self.ranks[mu]:
                raise ValueError(
                    f"Matrix shape {M.shape} incompatible with "
                    f"rank {self.ranks[mu]} at dim {mu}.")

            # self.spaces[mu] : (N_out, N_in, R_old)
            # M               : (R_old, R_new)
            # Result          : (N_out, N_in, R_new)

            # o = out, i = in, k = R_old, p = R_new
            new_spaces[mu] = np.einsum('oik, kp -> oip',
                                       self.spaces[mu], M)

        return TSpace(new_spaces, is_orth=False)

    def eval_in_space(self, dim, coefs):
        """
        Evaluates a matrix/vector in a given subspace from its coefficients on
        the basis.
        Computes the linear combination of the basis components.

        Parameters
        ----------
        dim : int
            The dimension (subspace) index to evaluate (0-indexed).
        c : array_like
            A 1D array of coefficients of length equal to the rank of the subspace.

        Returns
        -------
        numpy.ndarray
            The evaluated matrix of shape (N_out, N_in).
            (If the space is a vector space, N_in is 1).
        """
        coefs = np.asarray(coefs)

        if coefs.ndim != 1:
            raise ValueError("The coefficients 'c' must be a 1D array.")
        if len(coefs) != self.ranks[dim]:
            raise ValueError(
                f"Expected {self.ranks[dim]} coefficients, "
                f"got {len(coefs)} for dimension {dim}.")

        # self.spaces[dim]: (N_out, N_in, Rank)
        # c               : (Rank,)
        # Expected result : (N_out, N_in)

        # o = out, i = in, k = rank
        return np.einsum('oik, k -> oi',
                         self.spaces[dim], coefs)

    def transpose(self, conjugate=False):
        """
        Applies transposition (and optionally complex conjugation) to all operators.

        For each dimension, the physical axes (N_out, N_in) are swapped.
        The rank axis remains at the end.

        Parameters
        ----------
        conjugate : bool, optional
            If True, applies the complex conjugate transpose (Hermitian).
            If False (default), applies the standard transpose.

        Returns
        -------
        TSpace
            A new TSpace object with transposed operators.
        """
        new_spaces = []

        for mu in range(self.order):
            # self.spaces[mu] : (N_out, N_in, Rank)
            # result          : (N_in, N_out, Rank)
            transposed_space = self.spaces[mu].transpose(1, 0, 2)

            if conjugate:
                transposed_space = np.conj(transposed_space)

            new_spaces.append(transposed_space)

        return TSpace(new_spaces, is_orth=self.is_orth)

    @property
    def T(self):
        """
        Property to get the standard transpose of the tensor space.
        Allows the user to write: transposed_space = my_space.T
        """
        return self.transpose(conjugate=False)

    @property
    def H(self):
        """
        Property to get the Hermitian (complex conjugate) transpose of the tensor space.
        Allows the user to write: adjoint_space = my_space.H
        """
        return self.transpose(conjugate=True)

    def orth(self, dims=None):
        """
        Orthonormalizes the bases associated with the specified dimensions using
        QR decomposition.

        Parameters
        ----------
        dims : list of int, optional
            The dimensions to orthogonalize. All by default.

        Returns
        -------
        TSpace
            A new TSpace object with orthogonalized bases.
        list of numpy.ndarray
            A list of upper triangular matrices M (the R from QR decomposition)
            for each orthogonalized dimension. These matrices contain the
            transformation applied to the bases.
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

            # QR decomposition ('reduced' mode by default)
            # Q est (N_out * N_in, Rank) - orthonormal basis
            # R est (Rank, Rank)         - right triangular factor
            Q, R = np.linalg.qr(mat_2d)

            # get back the original shape
            new_spaces[mu] = Q.reshape(n_out, n_in, rank)

            R_matrices.append(R)

        # the result space is is_orth if all the bases have been orth
        is_fully_orth = len(dims) == self.order
        return TSpace(new_spaces, is_orth=is_fully_orth), R_matrices
