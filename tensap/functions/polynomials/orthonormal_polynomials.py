# Copyright (c) 2020, Anthony Nouy, Erwan Grelier
# This file is part of tensap (tensor approximation package).

# tensap is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# tensap is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with tensap.  If not, see <https://www.gnu.org/licenses/>.

"""
Module orthonormal_polynomials.

"""

from abc import abstractmethod
import math
import numpy as np
import tensap


class OrthonormalPolynomials(tensap.UnivariatePolynomials):
    """
    Class OrthonormalPolynomials.

    Attributes
    ----------
    measure : tensap.Measure
        The measure associated with the orthonormal polynomials.

    """

    @abstractmethod
    def __init__(self):
        """
        Constructor for the class OrthonormalPolynomials.

        Returns
        -------
        None.

        """
        self.measure = None

    #
    def __repr__(self):
        return "<{}>".format(self.__class__.__name__)

    @staticmethod
    def is_orthonormal():
        return True

    @staticmethod
    def one():
        return 1, 0

    def __eq__(self, poly_2):
        if not isinstance(poly_2, OrthonormalPolynomials):
            out = False
        else:
            out = (
                self.measure == poly_2.measure
                and np.array_equal(
                    self._recurrence_coefficients, poly_2._recurrence_coefficients
                )
                and np.array_equal(
                    self._orthogonal_polynomials_norms,
                    poly_2._orthogonal_polynomials_norms,
                )
                and self.is_orthonormal == poly_2.is_orthonormal
            )
        return out

    def domain(self):
        """
        Return the support of the associated measure.

        Returns
        -------
        numpy.ndarray
            The support of the associated measure.

        """
        return self.measure.support()

    def truncated_domain(self):
        """
        Return the truncated support of the associated measure.

        Returns
        -------
        numpy.ndarray
            The truncated support of the associated measure.

        """
        return self.measure.truncated_support()

    def moment(self, ind, measure=None):
        ind = np.atleast_2d(ind)

        if np.isin(ind.shape[1], [1, 2]) and (
            measure is None or measure == self.measure
        ):
            out = np.zeros(ind.shape[0])
            if ind.shape[1] == 1:
                out[ind[:, 0] == 0] = 1
            else:
                out[ind[:, 0] == ind[:, 1]] = 1
        else:
            out = super().moment(ind, measure)
        return out

    def poly_coeff(self, ind):
        max_ind = np.max(ind)
        recurr = self.recurrence_monic(max_ind)[0]
        a = recurr[0, : max_ind + 1]
        b = recurr[1, : max_ind + 1]

        coef = np.zeros([max_ind + 1] * 2)
        coef[0, 0] = 1

        if max_ind > 0:
            coef[1, 1:] = coef[0, :-1]
            coef[1, :] = (coef[1, :] - a[0] * coef[0, :]) / np.sqrt(b[1])

        if max_ind > 1:
            for deg in np.arange(1, max_ind):
                coef[deg + 1, 1:] = coef[deg, :-1]
                coef[deg + 1, :] = (
                    coef[deg + 1, :]
                    - a[deg] * coef[deg, :]
                    - np.sqrt(b[deg]) * coef[deg - 1, :]
                ) / np.sqrt(b[deg + 1])

        return coef[ind, :]

    def polyval(self, ind, x):
        x = np.ravel(x)
        max_ind = np.max(ind)
        recurr = self.recurrence_monic(max_ind)[0]

        a = recurr[0, : max_ind + 1]
        b = recurr[1, : max_ind + 1]

        out = np.zeros((x.size, max_ind + 1))
        out[:, 0] = 1
        if max_ind > 0:
            out[:, 1] = (x - a[0]) / np.sqrt(b[1])
            for deg in np.arange(2, max_ind + 1):
                out[:, deg] = (x - a[deg - 1]) * out[:, deg - 1] - np.sqrt(
                    b[deg - 1]
                ) * out[:, deg - 2]
                out[:, deg] /= np.sqrt(b[deg])
        out /= np.sqrt(self.measure.mass())
        return out[:, ind]

    def d_polyval(self, ind, x):
        return self.dn_polyval(1, ind, x)

    def dn_polyval(self, n, ind, x):
        x = np.ravel(x)
        max_ind = np.max(ind)
        recurr = self.recurrence_monic(max_ind)[0]
        a = recurr[0, : max_ind + 1]
        b = recurr[1, : max_ind + 1]

        out = self.polyval(np.arange(max_ind + 1), x)

        for k in np.arange(1, n + 1):
            out_0 = np.array(out)

            out = np.zeros((x.size, max_ind + 1))
            out[:, 0] = 0
            if max_ind > 0:
                if k == 1:
                    out[:, 1] = 1 / np.sqrt(b[1])
                else:
                    out[:, 1] = 0
            for deg in np.arange(2, max_ind + 1):
                out[:, deg] = (
                    k * out_0[:, deg - 1]
                    + (x - a[deg - 1]) * out[:, deg - 1]
                    - np.sqrt(b[deg - 1]) * out[:, deg - 2]
                ) / np.sqrt(b[deg])
        return out[:, ind]

    def random(self, ind, n=1, measure=None):
        """
        Return an array of size n of random evaluations of the polynomials for
        which the degree is in ind. If measure is not provided, the
        random generation is performed using self.measure.

        Parameters
        ----------
        ind : list or numpy.ndarray
            The indices of the polynomials to be evaluated.
        n : int, optional
            The number of random evaluations. The default is 1.
        measure : tensap.ProbabilityMeasure, optional
            The probability measure used for the generation of the input
            points. The default is None, indicating to use
            self.measure.

        Returns
        -------
        out : numpy.ndarray
            The random evaluations of the polynomials.
        x : numpy.ndarray
            The randomly drawn input points.

        """
        if measure is None:
            measure = self.measure

        assert isinstance(
            measure, tensap.ProbabilityMeasure
        ), "Must provide a ProbabilityMeasure."
        n = np.atleast_1d(int(n))
        x = measure.random(np.prod(n))

        out = np.zeros((np.prod(n), len(ind)))
        for i, ind_loc in enumerate(ind):
            out[:, i] = self.polyval(ind_loc, x)
        if n.size > 1 and not (n.size == 2 and n[1] == 1):
            out = np.reshape(out, (-1, len(ind)))
            x = np.reshape(x, n)
        return out, x

    def roots(self, deg):
        """
        Return the roots of the polynomial of degree deg.

        Parameters
        ----------
        deg : int
            The degree of the polynomial for which the roots are to be
            computed.

        Returns
        -------
        numpy.ndarray
            The roots of the polynomial of degree deg.

        """
        reccur = self.recurrence_monic(deg - 1)[0]
        a = reccur[0]
        b = reccur[1]

        # Jacobi matrix
        if deg == 1:
            jacobi_matrix = np.diag(a)
        else:
            jacobi_matrix = (
                np.diag(a) + np.diag(np.sqrt(b[1:]), -1) + np.diag(np.sqrt(b[1:]), 1)
            )
        return np.sort(np.linalg.eig(jacobi_matrix)[0])


class ShiftedOrthonormalPolynomials(tensap.UnivariatePolynomials):
    """
    Class ShiftedOrthonormalPolynomials.

    Attributes
    ----------
    measure : tensap.Measure
        The measure associated with the ShiftedOrthonormalPolynomials.
    polynomials : tensap.OrthonormalPolynomials
        The OrthonormalPolynomials which are shifted.
    shift : float
        The shifting parameter.
    scaling : float
        The scaling parameter.

    """

    def __init__(self, polynomials, shift, scaling):
        """
        Constructor for the class ShiftedOrthonormalPolynomials.

        Parameters
        ----------
        polynomials : tensap.OrthonormalPolynomials
            The OrthonormalPolynomials which are shifted.
        shift : float
            The shifting parameter.
        scaling : float
            The scaling parameter.

        Returns
        -------
        None.

        """
        self.measure = polynomials.measure.shift(shift, scaling)
        self.polynomials = polynomials
        self.shift = shift
        self.scaling = scaling

    def __repr__(self):
        return "<{}>".format(self.__class__.__name__)

    @staticmethod
    def is_orthonormal():
        return True

    @staticmethod
    def one():
        return 1, 0

    def __eq__(self, poly_2):
        if not isinstance(poly_2, ShiftedOrthonormalPolynomials):
            out = False
        else:
            out = (
                (self.polynomials == poly_2.polynomials)
                and (self.shift == poly_2.shift)
                and (self.scaling == poly_2.scaling)
            )
        return out

    def domain(self):
        """
        Return the support of the associated shifted measure.

        Returns
        -------
        numpy.ndarray
            The support of the associated measure.

        """
        return self.shift + self.scaling * self.polynomials.domain()

    def truncated_domain(self):
        """
        Return the truncated support of the associated shifted measure.

        Returns
        -------
        numpy.ndarray
            The truncated support of the associated measure.

        """
        return self.shift + self.scaling * self.polynomials.truncated_domain()

    def mean(self, ind, *measure):
        return self.polynomials.mean(ind, *measure)

    def moment(self, ind, *measure):
        if measure:
            if measure[0] == self.measure:
                measure = ()
            else:
                measure = tuple(
                    [measure[0].shift(-self.shift / self.scaling, 1 / self.scaling)]
                )
        return self.polynomials.moment(ind, *measure)

    def polyval(self, ind, x):
        x = (x - self.shift) / self.scaling
        return self.polynomials.polyval(ind, x)

    def d_polyval(self, ind, x):
        x = (x - self.shift) / self.scaling
        return self.polynomials.d_polyval(ind, x) / self.scaling

    def dn_polyval(self, n, ind, x):
        x = (x - self.shift) / self.scaling
        return self.polynomials.dn_polyval(n, ind, x) / (self.scaling**n)

    def random(self, ind, n=1, measure=None):
        """
        Return an array of size n of random evaluations of the shifted
        polynomials for which the degree is in ind. If measure is not provided,
        the random generation is performed using self.measure.

        Parameters
        ----------
        ind : list or numpy.ndarray
            The indices of the polynomials to be evaluated.
        n : int, optional
            The number of random evaluations. The default is 1.
        measure : tensap.ProbabilityMeasure, optional
            The probability measure used for the generation of the input
            points. The default is None, indicating to use
            self.measure.

        Returns
        -------
        fx : numpy.ndarray
            The random evaluations of the polynomials.
        x : numpy.ndarray
            The randomly drawn input points.

        """
        fx, x = self.polynomials.random(ind, n, measure)
        x = self.shift + self.scaling * x
        return fx, x

    def roots(self, n):
        """
        Return the roots of the shifted polynomial of degree deg.

        Parameters
        ----------
        deg : int
            The degree of the polynomial for which the roots are to be
            computed.

        Returns
        -------
        numpy.ndarray
            The roots of the polynomial of degree deg.

        """
        return self.shift + self.scaling * self.polynomials.roots(n)

    @staticmethod
    def ndim():
        return 1

    @staticmethod
    def poly_coeff():
        raise NotImplementedError("Method not implemented.")


class HermitePolynomials(OrthonormalPolynomials):
    """
    Class HermitePolynomials.

    Polynomials defined on R and orthonormal with respect to the standard
    gaussian measure 1/sqrt(2*pi)*exp(-x^2/2).

    """

    def __init__(self):
        """
        Constructor for the class HermitePolynomials.

        Parameters
        ----------
        n : int, optional
            The highest degree for which a polynomial can be computed with the
            stored recurrence coefficients. The default is 50.

        Returns
        -------
        None.

        """
        self.measure = tensap.NormalRandomVariable(0, 1)

    def recurrence_monic(self, n):
        """
        Computes the coefficients of the three-term recurrence used
        to construct the monic polynomials
        :math::`p_{n+1}(x) = (x-a_n)p_n(x) - b_n p_{n-1}(x)`, an and bn are
        the three-term recurrence coefficients

        Parameters
        ----------
        n : int

        Returns
        -------
        recurr : 2-by-(n+1) numpy.ndarray
            recurr[0] contains [a_0 , ..., a_n]
            recurr[1] contains [b_0 , ..., b_n]
        norms : 1-by-(n+1) numpy.ndarray
            norms array
        """
        recurr = np.zeros((2, n + 1))
        recurr[1, :] = np.arange(n + 1)
        norms = np.array([np.sqrt(float(math.factorial(x))) for x in range(n + 1)])
        return recurr, norms


class LegendrePolynomials(OrthonormalPolynomials):
    """
    Class LegendrePolynomials.

    Polynomials defined on [-1,1], orthonormal with respect to the standard
    uniform measure 1/2.

    """

    def __init__(self):
        """
        Constructor for the class LegendrePolynomials.

        Parameters
        ----------
        n : int, optional
            The highest degree for which a polynomial can be computed with the
            stored recurrence coefficients. The default is 50.

        Returns
        -------
        None.

        """
        self.measure = tensap.UniformRandomVariable(-1, 1)

    def recurrence_monic(self, n):
        """
        Computes the coefficients of the three-term recurrence used
        to construct the monic polynomials
        :math::`p_{n+1}(x) = (x-a_n)p_n(x) - b_n p_{n-1}(x)`, an and bn are
        the three-term recurrence coefficients

        Parameters
        ----------
        n : int

        Returns
        -------
        recurr : 2-by-(n+1) numpy.ndarray
            recurr[0] contains [a_0 , ..., a_n]
            recurr[1] contains [b_0 , ..., b_n]
        norms : 1-by-(n+1) numpy.ndarray
            norms array
        """
        recurr = np.zeros((2, n + 1))
        recurr[1, :] = np.arange(n + 1) ** 2 / (4 * np.arange(n + 1) ** 2 - 1)
        norms = np.array(
            [
                np.sqrt(1 / (2 * x + 1))
                * 2**x
                * math.factorial(x) ** 2
                / math.factorial(2 * x)
                for x in range(n + 1)
            ]
        )
        return recurr, norms


class LegendrePolynomialsLebesgue(OrthonormalPolynomials):
    """
    Class LegendrePolynomialsLebesgue.

    Polynomials defined on [-1,1], orthonormal with respect to the Lebesgue
      measure.

    """

    def __init__(self):
        """
        Constructor for the class LegendrePolynomialsLebesgue.

        Parameters
        ----------
        n : int, optional
            The highest degree for which a polynomial can be computed with the
            stored recurrence coefficients. The default is 50.

        Returns
        -------
        None.

        """
        self.measure = tensap.LebesgueMeasure(-1, 1)

    def recurrence_monic(self, n):
        """
        Computes the coefficients of the three-term recurrence used
        to construct the monic polynomials
        :math::`p_{n+1}(x) = (x-a_n)p_n(x) - b_n p_{n-1}(x)`, an and bn are
        the three-term recurrence coefficients

        Parameters
        ----------
        n : int

        Returns
        -------
        recurr : 2-by-(n+1) numpy.ndarray
            recurr[0] contains [a_0 , ..., a_n]
            recurr[1] contains [b_0 , ..., b_n]
        norms : 1-by-(n+1) numpy.ndarray
            norms array
        """
        recurr = np.zeros((2, n + 1))
        recurr[1, :] = np.arange(n + 1) ** 2 / (4 * np.arange(n + 1) ** 2 - 1)

        norms = np.array(
            [
                np.sqrt(1 / (2 * x + 1))
                * 2**x
                * math.factorial(x) ** 2
                / math.factorial(2 * x)
                * np.sqrt(2)
                for x in range(n + 1)
            ]
        )
        return recurr, norms


class EmpiricalPolynomials(OrthonormalPolynomials):
    """
    Class EmpiricalPolynomials.

    Polynomials defined on R and orthonormal with respect to the gaussian
    kernel smoothed distribution based on a sample x, which was centered and
    normalized (unit variance).

    """

    def __init__(self, sample, n=None):
        """
        Constructor for the class EmpiricalPolynomials.

        Parameters
        ----------
        sample : numpy.ndarray or list
            The sample used used to fit the probability density function using
            Scott's rule.
        n : int, optional
            The highest degree for which a polynomial can be computed with the
            stored recurrence coefficients. The default is the size of the sample.

        Returns
        -------
        None.

        """
        if isinstance(sample, tensap.EmpiricalRandomVariable):
            self.measure = sample.get_standard_random_variable()
        else:
            # Standardization of the sample
            x = np.reshape(self.sample, [-1, 1])
            x = (x - np.tile(np.mean(x, 0), (x.shape[0], 1))) / np.tile(
                np.std(x, 0), (x.shape[0], 1)
            )
            self.measure = tensap.EmpiricalRandomVariable(np.ravel(x))
        if n is None:
            n = self.measure.sample.shape[0] - 1

        (
            self._recurrence_coefficients,
            self._orthogonal_polynomials_norms,
        ) = self._precompute_recurrence_monic(self.measure, n)

    def recurrence_monic(self, n):
        recurr = self._recurrence_coefficients[:, np.arange(n + 1)]
        norms = self._orthogonal_polynomials_norms[np.arange(n + 1)]
        return recurr, norms

    @staticmethod
    def _precompute_recurrence_monic(measure, n):
        """
        Computes the coefficients of the three-term recurrence used
        to construct the monic polynomials
        p_{n+1}(x) = (x-a_n)p_n(x) - b_n p_{n-1}(x), a_n and b_n are
        the three-term recurrence coefficients

        Parameters
        ----------
        n : int

        Returns
        -------
        recurr : 2-by-(n+1) numpy.ndarray
            recurr[0] contains [a_0 , ..., a_n]
            recurr[1] contains [b_0 , ..., b_n]
        norms : 1-by-(n+1) numpy.ndarray
            norms array
        """

        def is_orth(pnp1, pn, pnm1, weights):
            """
            Determine if the polynomial pnp1 is orthogonal to the polynomials
            pn and pnm1, according to the DiscreteRandomVariable r.

            Parameters
            ----------
            pnp1 : function
                The evaluations of the first function at integration points.
            pn : function
                The evaluations of the second function at integration points.
            pnm1 : function
                The evaluations of the third function at integration points.
            weights : numpy.ndarray
                The weights used for the numerical integration.

            Returns
            -------
            bool
                Boolean equal to True if the polynomial pnp1 is orthogonal to
                the polynomials pn and pnm1, False otherwise.

            """
            tol = 1e-4  # Tolerance for the inner product to be considered as 0

            d1 = np.abs(np.sum(np.matmul(pnp1 * pn, weights)))
            d2 = np.abs(np.sum(np.matmul(pnp1 * pnm1, weights)))

            if d1 < tol and d2 < tol:
                return True
            return False

        def peval(i, a, b, x):
            """
            Evaluate the polynomial of degree i defined by the coefficients a
            and b at points x.

            Parameters
            ----------
            i : int
                The degree of the polynomial.
            a : numpy.ndarray
                The first recurrence coefficients.
            b : numpy.ndarray
                The second recurrence coefficients.
            x : numpy.ndarray
                The points of evaluation.

            Returns
            -------
            p_n_loc : numpy.ndarray
                The evaluations of the polynomial at points x.

            """

            if i < 0:
                p_n_loc = np.zeros(x.shape)
            elif i == 0:
                p_n_loc = np.ones(x.shape)
            else:
                p_n_loc_m2 = 1
                p_n_loc_m1 = x - a[0]
                p_n_loc = np.array(p_n_loc_m1)
                for N in np.arange(2, i + 1):
                    p_n_loc = (x - a[N - 1]) * p_n_loc_m1 - b[N - 1] * p_n_loc_m2
                    p_n_loc_m2 = np.array(p_n_loc_m1)
                    p_n_loc_m1 = np.array(p_n_loc)
            return p_n_loc

        if n is None:
            check = True
            n = 10
        else:
            check = False

        norms = np.zeros(n + 2)
        a = np.zeros(n + 2)
        b = np.zeros(n + 2)

        G = tensap.NormalRandomVariable().gauss_integration_rule(
            int(np.ceil((2 * n + 3) / 2))
        )
        xi = measure.sample
        weights = G.weights / xi.size

        x_ij = measure.bandwidth * np.tile(
            np.reshape(G.points, [1, -1]), (xi.size, 1)
        ) + np.tile(np.reshape(xi, [-1, 1]), (1, G.points.size))

        i = 0
        cond = True
        norms[0] = 1
        a[0] = np.sum(np.matmul(x_ij, weights))
        b[0] = 0

        while cond and i <= n:
            i += 1

            p_n_m1 = np.reshape(peval(i - 1, a, b, x_ij), x_ij.shape)
            p_n = np.reshape(peval(i, a, b, x_ij), x_ij.shape)

            norms[i] = np.sum(np.matmul(p_n**2, weights))
            a[i] = np.sum(np.matmul(p_n * x_ij * p_n, weights)) / norms[i]
            b[i] = norms[i] / norms[i - 1]

            p_n_p1 = (x_ij - a[i]) * p_n - b[i] * p_n_m1

            if check:
                # Orthogonality condition, only if the number of polynomials
                # is not specified by the user
                cond = is_orth(p_n_p1, p_n, p_n_m1, weights)

        if not cond and i - 2 != n:
            raise ValueError("Maximum degree: %i (%i asked)" % (i - 2, n))

        return np.vstack((a[:i], b[:i])), np.sqrt(norms[:i])


class DiscretePolynomials(OrthonormalPolynomials):
    """
    Class DiscretePolynomials.

    Polynomials orthonormal with respect to a discrete measure.

    """

    def __init__(self, measure=None):
        """
        Constructor for the class DiscretePolynomials.

        Polynomials orthonormal with respect to a discrete measure.

        Parameters
        ----------
        measure : tensap.DiscreteRandomVariable or tensap.DiscreteMeaasure
            The discrete measure with respect to which the polynomials form an
            orthonormal basis.

        Returns
        -------
        None.

        """
        assert isinstance(measure, tensap.DiscreteRandomVariable) or isinstance(
            measure, tensap.DiscreteMeasure
        ), "Must specify a DiscreteRandomVariable or DiscreteMeasure."

        self.measure = measure
        n = np.size(measure.values) - 1
        (
            self._recurrence_coefficients,
            self._orthogonal_polynomials_norms,
        ) = tensap.DiscretePolynomials._precompute_recurrence_monic(measure, n)

    def recurrence_monic(self, n):
        recurr = self._recurrence_coefficients[:, np.arange(n + 1)]
        norms = self._orthogonal_polynomials_norms[np.arange(n + 1)]
        return recurr, norms

    @staticmethod
    def _precompute_recurrence_monic(measure, n):
        """
        Precompute the coefficients of the three-term recurrence used
        to construct the monic polynomials
        p_{n+1}(x) = (x-a_n)p_n(x) - b_n p_{n-1}(x), a_n and b_n are
        the three-term recurrence coefficients

        Parameters
        ----------
        n : int

        Returns
        -------
        recurr : 2-by-(n+1) numpy.ndarray
            recurr[0] contains [a_0 , ..., a_n]
            recurr[1] contains [b_0 , ..., b_n]
        norms : 1-by-(n+1) numpy.ndarray
            norms array
        """

        def dot_product(p1, p2, r):
            """
            Compute the inner product between two polynomials p1 and p2,
            according to the DiscreteRandomVariable r.

            Parameters
            ----------
            p1 : function
                The first function of the inner product.
            p2 : function
                The second function of the inner product.
            r : tensap.DiscreteRandomVariable
                The random variable used to compute the inner product.

            Returns
            -------
            float
                The inner product between p1 and p2.

            """
            G = r.integration_rule()
            return G.integrate(lambda x: p1(x) * p2(x))

        def is_orth(pnp1, pn, pnm1, r):
            """
            Determine if the polynomial pnp1 is orthogonal to the polynomials
            pn and pnm1, according to the DiscreteRandomVariable r.

            Parameters
            ----------
            pnp1 : function
                The first function.
            pn : function
                The second function.
            pnm1 : function
                The third function.
            r : tensap.DiscreteRandomVariable
                The random variable used to compute the inner product.

            Returns
            -------
            bool
                Boolean equal to True if the polynomial pnp1 is orthogonal to
                the polynomials pn and pnm1, False otherwise.

            """
            tol = 1e-5  # Tolerance for the inner product to be considered as 0

            d1 = np.abs(dot_product(pnp1, pn, r))
            d2 = np.abs(dot_product(pnp1, pnm1, r))

            if d1 < tol and d2 < tol:
                return True
            return False

        norms = np.zeros(n + 2)
        a = np.zeros(n + 2)
        b = np.zeros(n + 2)

        i = 0
        cond = True
        norms[0] = dot_product(lambda x: x**0, lambda x: x**0, measure)
        a[0] = dot_product(lambda x: x**0, lambda x: x, measure) / norms[0]
        b[0] = 0
        pnm1 = []
        pn = [lambda x: 1]
        pnp1 = [lambda x, a=a[0]: (x - a) * pn[0](x)]

        while cond and i <= n:
            i += 1
            pnm1.append(pn[i - 1])
            pn.append(pnp1[i - 1])

            norms[i] = dot_product(pn[i], pn[i], measure)
            if norms[i] == 0:
                a[i] = np.nan
            else:
                a[i] = dot_product(pn[i], lambda x: x * pn[i](x), measure) / norms[i]
            b[i] = norms[i] / norms[i - 1]

            pnp1.append(
                lambda x, a=a[i], b=b[i], pn=pn[i], pnm1=pnm1[i - 1]: (x - a) * pn(x)
                - b * pnm1(x)
            )

            # Orthogonality condition
            cond = is_orth(pnp1[i], pn[i], pnm1[i - 1], measure)

        return np.vstack((a[:i], b[:i])), np.sqrt(norms[:i])
