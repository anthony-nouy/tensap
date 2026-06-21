"""Tests for TSpace, TSpaceVectors, and TSpaceOperators."""

import numpy as np
import tensap


def _rng(seed=0):
    return np.random.RandomState(seed)


class TestTSpace:
    """Tests for the base TSpace class."""

    def test_init(self):
        spaces = [np.random.rand(5, 1, 3), np.random.rand(7, 1, 4)]
        s = tensap.TSpace(spaces)
        assert s.order == 2
        assert np.all(s.dims_out == [5, 7])
        assert np.all(s.dims_in == [1, 1])
        assert np.all(s.ranks == [3, 4])
        assert not s.is_orth

    def test_init_promotes_2d(self):
        spaces = [np.random.rand(5, 3), np.random.rand(7, 4)]
        s = tensap.TSpace(spaces)
        assert s.spaces[0].shape == (5, 1, 3)

    def test_init_is_orth(self):
        spaces = [np.random.rand(5, 1, 3), np.random.rand(7, 1, 4)]
        s = tensap.TSpace(spaces, is_orth=True)
        assert s.is_orth

    def test_init_invalid_type(self):
        try:
            tensap.TSpace("not a list")
            assert False
        except TypeError:
            pass

    def test_init_invalid_ndim(self):
        try:
            tensap.TSpace([np.random.rand(5)])
            assert False
        except ValueError:
            pass

    def test_storage(self):
        spaces = [np.random.rand(5, 1, 3), np.random.rand(7, 1, 4)]
        s = tensap.TSpace(spaces)
        assert s.storage() == 5 * 1 * 3 + 7 * 1 * 4

    def test_sparse_storage(self):
        spaces = [np.zeros((5, 1, 3)), np.ones((7, 1, 4))]
        s = tensap.TSpace(spaces)
        assert s.sparse_storage() == 7 * 1 * 4

    def test_representation_rank(self):
        spaces = [np.random.rand(5, 1, 3), np.random.rand(7, 1, 4)]
        s = tensap.TSpace(spaces)
        r = s.representation_rank()
        assert np.all(r == [3, 4])
        r[0] = 99
        assert s.ranks[0] == 3

    def test_cat(self):
        s1 = tensap.TSpace([np.random.rand(5, 1, 3), np.random.rand(7, 1, 4)])
        s2 = tensap.TSpace([np.random.rand(5, 1, 2), np.random.rand(7, 1, 5)])
        s = s1.cat(s2)
        assert np.all(s.ranks == [5, 9])
        assert np.all(s.dims_out == [5, 7])
        assert np.all(s.dims_in == [1, 1])

    def test_cat_error_order(self):
        s1 = tensap.TSpace([np.random.rand(5, 1, 3)])
        s2 = tensap.TSpace([np.random.rand(5, 1, 3), np.random.rand(7, 1, 4)])
        try:
            s1.cat(s2)
            assert False
        except ValueError:
            pass

    def test_cat_error_dims_out(self):
        s1 = tensap.TSpace([np.random.rand(5, 1, 3), np.random.rand(7, 1, 4)])
        s2 = tensap.TSpace([np.random.rand(9, 1, 3), np.random.rand(7, 1, 4)])
        try:
            s1.cat(s2)
            assert False
        except ValueError:
            pass

    def test_dot_all_dims(self):
        s1 = tensap.TSpace([np.random.rand(5, 1, 3), np.random.rand(7, 1, 4)],
                           is_orth=True)
        s2 = tensap.TSpace([np.random.rand(5, 1, 3), np.random.rand(7, 1, 4)])
        M = s1.dot(s2)
        assert len(M) == 2
        assert M[0].shape == (3, 3)

    def test_dot_subset_dims(self):
        s1 = tensap.TSpace([np.random.rand(5, 1, 3), np.random.rand(7, 1, 4)])
        s2 = tensap.TSpace([np.random.rand(5, 1, 3), np.random.rand(7, 1, 4)])
        M = s1.dot(s2, dims=0)
        assert len(M) == 1
        assert M[0].shape == (3, 3)

    def test_dot_self(self):
        s1 = tensap.TSpace([np.random.rand(5, 1, 3), np.random.rand(7, 1, 4)])
        M = s1.dot(s1)
        assert np.allclose(M[0], s1.spaces[0][:, 0, :].T @ s1.spaces[0][:, 0, :])

    def test_dot_error_dims_out(self):
        s1 = tensap.TSpace([np.random.rand(5, 1, 3), np.random.rand(7, 1, 4)])
        s2 = tensap.TSpace([np.random.rand(9, 1, 3), np.random.rand(7, 1, 4)])
        try:
            s1.dot(s2)
            assert False
        except ValueError:
            pass

    def test_matrix_times_space_all(self):
        spaces = [np.random.rand(5, 1, 3), np.random.rand(7, 1, 4)]
        s = tensap.TSpace(spaces)
        M = [np.random.rand(3, 5), np.random.rand(4, 7)]
        s2 = s.matrix_times_space(M)
        assert s2.dims_out[0] == 3
        assert s2.dims_out[1] == 4
        assert np.all(s2.ranks == [3, 4])

    def test_matrix_times_space_subset(self):
        spaces = [np.random.rand(5, 1, 3), np.random.rand(7, 1, 4)]
        s = tensap.TSpace(spaces)
        M = [np.random.rand(3, 7)]
        s2 = s.matrix_times_space(M, dims=1)
        assert s2.dims_out[0] == 5
        assert s2.dims_out[1] == 3

    def test_matrix_times_space_error_count(self):
        spaces = [np.random.rand(5, 1, 3), np.random.rand(7, 1, 4)]
        s = tensap.TSpace(spaces)
        M = [np.random.rand(3, 5)]
        try:
            s.matrix_times_space(M, dims=[0, 1])
            assert False
        except ValueError:
            pass

    def test_matrix_times_space_error_shape(self):
        spaces = [np.random.rand(5, 1, 3)]
        s = tensap.TSpace(spaces)
        try:
            s.matrix_times_space([np.random.rand(3, 9)])
            assert False
        except ValueError:
            pass

    def test_space_times_matrix_all(self):
        spaces = [np.random.rand(5, 1, 3), np.random.rand(7, 1, 4)]
        s = tensap.TSpace(spaces)
        M = [np.random.rand(3, 2), np.random.rand(4, 5)]
        s2 = s.space_times_matrix(M)
        assert np.all(s2.ranks == [2, 5])
        assert np.all(s2.dims_out == [5, 7])

    def test_space_times_matrix_subset(self):
        spaces = [np.random.rand(5, 1, 3), np.random.rand(7, 1, 4)]
        s = tensap.TSpace(spaces)
        s2 = s.space_times_matrix([np.random.rand(3, 6)], dims=0)
        assert s2.ranks[0] == 6
        assert s2.ranks[1] == 4

    def test_space_times_matrix_error_rank(self):
        spaces = [np.random.rand(5, 1, 3)]
        s = tensap.TSpace(spaces)
        try:
            s.space_times_matrix([np.random.rand(9, 2)])
            assert False
        except ValueError:
            pass

    def test_eval_in_space(self):
        spaces = [np.random.rand(5, 1, 3)]
        s = tensap.TSpace(spaces)
        coefs = np.array([1.0, 2.0, 3.0])
        result = s.eval_in_space(0, coefs)
        assert result.shape == (5, 1)
        expected = s.spaces[0][:, 0, :] @ coefs
        assert np.allclose(result[:, 0], expected)

    def test_eval_in_space_error_ndim(self):
        spaces = [np.random.rand(5, 1, 3)]
        s = tensap.TSpace(spaces)
        try:
            s.eval_in_space(0, np.ones((2, 2)))
            assert False
        except ValueError:
            pass

    def test_eval_in_space_error_count(self):
        spaces = [np.random.rand(5, 1, 3)]
        s = tensap.TSpace(spaces)
        try:
            s.eval_in_space(0, np.ones(5))
            assert False
        except ValueError:
            pass

    def test_orth(self):
        rng = _rng(0)
        spaces = [rng.rand(5, 1, 3), rng.rand(7, 1, 4)]
        s = tensap.TSpace(spaces)
        q, mats = s.orth()
        assert q.is_orth
        assert len(mats) == 2
        assert mats[0].shape == (3, 3)
        assert np.allclose(q.spaces[0][:, 0, :].T @ q.spaces[0][:, 0, :],
                           np.eye(3), atol=1e-10)

    def test_orth_subset_dims(self):
        rng = _rng(0)
        spaces = [rng.rand(5, 1, 3), rng.rand(7, 1, 4)]
        s = tensap.TSpace(spaces)
        q, mats = s.orth(dims=1)
        assert not q.is_orth
        assert len(mats) == 1

    def test_truncate_tight(self):
        rng = _rng(0)
        spaces = [rng.rand(5, 1, 3), rng.rand(7, 1, 4)]
        s = tensap.TSpace(spaces)
        t, mats = s.truncate(tol=1e-16)
        assert t.is_orth
        assert np.all(t.ranks == [3, 4])

    def test_truncate_loose(self):
        rng = _rng(0)
        spaces = [rng.rand(5, 1, 3), rng.rand(7, 1, 4)]
        s = tensap.TSpace(spaces)
        t, mats = s.truncate(tol=0.5)
        assert np.all(t.ranks <= [3, 4])
        assert np.any(t.ranks < [3, 4])

    def test_permute(self):
        spaces = [np.random.rand(5, 1, 3), np.random.rand(7, 1, 4),
                  np.random.rand(9, 1, 2)]
        s = tensap.TSpace(spaces)
        p = s.permute([2, 0, 1])
        assert p.dims_out[0] == 9
        assert p.dims_out[1] == 5
        assert p.dims_out[2] == 7

    def test_permute_error_length(self):
        spaces = [np.random.rand(5, 1, 3), np.random.rand(7, 1, 4)]
        s = tensap.TSpace(spaces)
        try:
            s.permute([0])
            assert False
        except ValueError:
            pass

    def test_keep_space(self):
        spaces = [np.random.rand(5, 1, 3), np.random.rand(7, 1, 4),
                  np.random.rand(9, 1, 2)]
        s = tensap.TSpace(spaces)
        k = s.keep_space([0, 2])
        assert k.order == 2
        assert np.all(k.dims_out == [5, 9])

    def test_remove_space(self):
        spaces = [np.random.rand(5, 1, 3), np.random.rand(7, 1, 4),
                  np.random.rand(9, 1, 2)]
        s = tensap.TSpace(spaces)
        k = s.remove_space([1])
        assert k.order == 2
        assert np.all(k.dims_out == [5, 9])


class TestTSpaceVectors:
    """Tests for TSpaceVectors."""

    def test_init(self):
        spaces = [np.random.rand(5, 3), np.random.rand(7, 4)]
        s = tensap.TSpaceVectors(spaces)
        assert s.order == 2
        assert np.all(s.dims_in == 1)
        assert np.all(s.ranks == [3, 4])

    def test_init_error_dims_in(self):
        try:
            tensap.TSpaceVectors([np.random.rand(5, 2, 3)])
            assert False
        except ValueError:
            pass

    def test_diag_cat(self):
        rng = _rng(0)
        vx = tensap.TSpaceVectors([rng.rand(5, 1, 3), rng.rand(7, 1, 4)])
        vy = tensap.TSpaceVectors([rng.rand(2, 1, 1), rng.rand(3, 1, 2)])
        v = vx.diag_cat(vy)
        assert np.all(v.ranks == [4, 6])
        assert np.all(v.dims_out == [7, 10])
        assert np.array_equal(v.spaces[0][:5, :, :3], vx.spaces[0])
        assert np.array_equal(v.spaces[0][5:, :, 3:], vy.spaces[0])

    def test_diag_cat_error_order(self):
        vx = tensap.TSpaceVectors([np.random.rand(5, 1, 3)])
        vy = tensap.TSpaceVectors([np.random.rand(5, 1, 3),
                                   np.random.rand(7, 1, 4)])
        try:
            vx.diag_cat(vy)
            assert False
        except ValueError:
            pass

    def test_dot_with_metrics_all(self):
        rng = _rng(0)
        v = tensap.TSpaceVectors([rng.rand(5, 1, 3), rng.rand(7, 1, 4)])
        eps = tensap.TSpaceOperators(
            [rng.rand(5, 5, 2), rng.rand(7, 7, 3)])
        M = v.dot_with_metrics(v, eps)
        assert len(M) == 2
        assert M[0].shape == (2, 3, 3)

    def test_dot_with_metrics_subset(self):
        rng = _rng(0)
        v = tensap.TSpaceVectors([rng.rand(5, 1, 3), rng.rand(7, 1, 4)])
        eps = tensap.TSpaceOperators(
            [rng.rand(5, 5, 2), rng.rand(7, 7, 3)])
        M = v.dot_with_metrics(v, eps, dims=0)
        assert len(M) == 1

    def test_dot_with_metrics_error_square(self):
        v = tensap.TSpaceVectors([np.random.rand(5, 1, 3)])
        non_square = tensap.TSpaceOperators([np.random.rand(5, 3, 2)])
        try:
            v.dot_with_metrics(v, non_square)
            assert False
        except ValueError:
            pass

    def test_eval_at_indices_all(self):
        rng = _rng(0)
        v = tensap.TSpaceVectors([rng.rand(10, 1, 3), rng.rand(10, 1, 4)])
        indices = np.array([[0, 1], [2, 3]])
        ve = v.eval_at_indices(indices)
        assert ve.spaces[0].shape == (2, 1, 3)
        assert np.all(ve.spaces[0][:, 0, :] == v.spaces[0][[0, 2], 0, :])

    def test_eval_at_indices_subset(self):
        v = tensap.TSpaceVectors([np.random.rand(10, 1, 3),
                                  np.random.rand(10, 1, 4)])
        indices = np.array([[0, 1], [2, 3]])
        ve = v.eval_at_indices(indices, dims=0)
        assert ve.spaces[0].shape == (2, 1, 3)
        assert ve.spaces[1].shape == (10, 1, 4)

    def test_unvectorize(self):
        rng = _rng(0)
        v = tensap.TSpaceVectors([rng.rand(6, 1, 3), rng.rand(12, 1, 2)])
        sz = np.array([[2, 4], [3, 3]])
        o = v.unvectorize(sz)
        assert isinstance(o, tensap.TSpaceOperators)
        assert o.spaces[0].shape == (2, 3, 3)
        assert o.spaces[1].shape == (4, 3, 2)

    def test_create(self):
        s = tensap.TSpaceVectors.create(lambda x: np.ones(x), [5, 7],
                                        nb_vectors=[3, 4])
        assert np.all(s.ranks == [3, 4])
        assert np.all(s.dims_out == [5, 7])
        assert np.allclose(s.spaces[0][:, 0, :], 1.0)

    def test_create_default_nb_vectors(self):
        s = tensap.TSpaceVectors.create(lambda x: np.ones(x), [5, 7])
        assert np.all(s.ranks == [1, 1])

    def test_zeros(self):
        s = tensap.TSpaceVectors.zeros([5, 7], nb_vectors=[3, 4])
        assert np.allclose(s.spaces[0][:, 0, :], 0.0)

    def test_ones(self):
        s = tensap.TSpaceVectors.ones([5, 7], nb_vectors=[3, 4])
        assert np.allclose(s.spaces[0][:, 0, :], 1.0)

    def test_rand(self):
        s = tensap.TSpaceVectors.rand([5, 7], nb_vectors=[3, 4])
        assert s.spaces[0].shape == (5, 1, 3)
        assert s.spaces[1].shape == (7, 1, 4)

    def test_randn(self):
        s = tensap.TSpaceVectors.randn([5, 7], nb_vectors=[3, 4])
        assert s.spaces[0].shape == (5, 1, 3)

    def test_eye(self):
        s = tensap.TSpaceVectors.eye([5, 7], nb_vectors=[3, 4])
        assert s.spaces[0].shape == (5, 1, 3)
        assert np.allclose(s.spaces[0][:, 0, :],
                           np.eye(5, 3))

    def test_eye_default_nb_vectors(self):
        s = tensap.TSpaceVectors.eye([5])
        assert s.spaces[0].shape == (5, 1, 5)
        assert np.allclose(s.spaces[0][:, 0, :], np.eye(5))


class TestTSpaceOperators:
    """Tests for TSpaceOperators."""

    def test_init(self):
        spaces = [np.random.rand(5, 3, 2), np.random.rand(7, 4, 3)]
        s = tensap.TSpaceOperators(spaces)
        assert s.order == 2
        assert np.all(s.dims_in == [3, 4])
        assert np.all(s.ranks == [2, 3])

    def test_init_promotes_2d(self):
        s = tensap.TSpaceOperators([np.random.rand(5, 3), np.random.rand(7, 4)])
        assert s.spaces[0].shape == (5, 1, 3)

    def test_init_error_dims_in_zero(self):
        try:
            tensap.TSpaceOperators([np.random.rand(5, 0, 3)])
            assert False
        except ValueError:
            pass

    def test_mtimes_operator_operator(self):
        rng = _rng(0)
        op = tensap.TSpaceOperators(
            [rng.rand(5, 3, 2), rng.rand(7, 4, 3)])
        other = tensap.TSpaceOperators(
            [rng.rand(3, 6, 2), rng.rand(4, 8, 3)])
        result = op.mtimes(other)
        assert isinstance(result, tensap.TSpaceOperators)
        assert result.spaces[0].shape == (5, 6, 4)

    def test_mtimes_operator_vector(self):
        rng = _rng(0)
        op = tensap.TSpaceOperators(
            [rng.rand(5, 3, 2), rng.rand(7, 4, 3)])
        vec = tensap.TSpaceVectors([rng.rand(3, 1, 2), rng.rand(4, 1, 3)])
        result = op.mtimes(vec)
        assert isinstance(result, tensap.TSpaceVectors)
        assert result.spaces[0].shape == (5, 1, 4)

    def test_mtimes_subset_dims(self):
        rng = _rng(0)
        op = tensap.TSpaceOperators(
            [rng.rand(5, 3, 2), rng.rand(7, 4, 3)])
        vec = tensap.TSpaceVectors([rng.rand(3, 1, 2), rng.rand(4, 1, 3)])
        result = op.mtimes(vec, dims=0)
        assert isinstance(result, tensap.TSpaceVectors)

    def test_mtimes_error_mismatch(self):
        op = tensap.TSpaceOperators([np.random.rand(5, 3, 2)])
        vec = tensap.TSpaceVectors([np.random.rand(9, 1, 2)])
        try:
            op.mtimes(vec)
            assert False
        except ValueError:
            pass

    def test_transpose(self):
        rng = _rng(0)
        op = tensap.TSpaceOperators([rng.rand(5, 3, 2)])
        t = op.transpose()
        assert t.spaces[0].shape == (3, 5, 2)
        assert np.allclose(t.spaces[0], op.spaces[0].transpose(1, 0, 2))

    def test_T_property(self):
        op = tensap.TSpaceOperators([np.random.rand(5, 3, 2)])
        assert np.allclose(op.T.spaces[0], op.spaces[0].transpose(1, 0, 2))

    def test_H_property(self):
        op = tensap.TSpaceOperators([np.random.rand(5, 3, 2)])
        expected = np.conj(op.spaces[0]).transpose(1, 0, 2)
        assert np.allclose(op.H.spaces[0], expected)

    def test_vectorize(self):
        rng = _rng(0)
        op = tensap.TSpaceOperators([rng.rand(5, 3, 2), rng.rand(7, 4, 3)])
        v = op.vectorize()
        assert isinstance(v, tensap.TSpaceVectors)
        assert v.spaces[0].shape == (15, 1, 2)
        assert v.spaces[1].shape == (28, 1, 3)
        assert np.all(v.dims_in == 1)

    def test_vectorize_conserves_orth(self):
        op = tensap.TSpaceOperators([np.random.rand(5, 3, 2)], is_orth=True)
        v = op.vectorize()
        assert v.is_orth

    def test_create(self):
        s = tensap.TSpaceOperators.create(
            lambda x: np.ones(x), [5, 7], sz2=[3, 4], nb_vectors=[2, 3])
        assert np.all(s.ranks == [2, 3])
        assert s.spaces[0].shape == (5, 3, 2)
        assert np.allclose(s.spaces[0][:, :, 0], 1.0)

    def test_create_default_sz2(self):
        s = tensap.TSpaceOperators.create(
            lambda x: np.ones(x), [5, 7], nb_vectors=[2, 3])
        assert s.spaces[0].shape == (5, 5, 2)

    def test_create_default_nb_vectors(self):
        s = tensap.TSpaceOperators.create(lambda x: np.ones(x), [5, 7])
        assert np.all(s.ranks == [1, 1])

    def test_zeros(self):
        s = tensap.TSpaceOperators.zeros([5, 7], sz2=[3, 4], nb_vectors=[2, 3])
        assert np.allclose(s.spaces[0], 0.0)

    def test_ones(self):
        s = tensap.TSpaceOperators.ones([5, 7], sz2=[3, 4], nb_vectors=[2, 3])
        assert np.allclose(s.spaces[0], 1.0)

    def test_rand(self):
        s = tensap.TSpaceOperators.rand([5, 7], sz2=[3, 4], nb_vectors=[2, 3])
        assert s.spaces[0].shape == (5, 3, 2)

    def test_randn(self):
        s = tensap.TSpaceOperators.randn([5, 7], sz2=[3, 4], nb_vectors=[2, 3])
        assert s.spaces[0].shape == (5, 3, 2)

    def test_eye(self):
        s = tensap.TSpaceOperators.eye([5, 7], nb_vectors=[2, 3])
        assert s.spaces[0].shape == (5, 5, 2)
        assert np.allclose(s.spaces[0][:, :, 0], np.eye(5))

    def test_eye_default_nb_vectors(self):
        s = tensap.TSpaceOperators.eye([5, 7])
        assert np.all(s.ranks == [1, 1])
