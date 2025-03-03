# This file is part of tensap (tensor approximation package).

"""
Tutorial on orthonormal polynomials.

"""

import tensap as tp
import numpy as np
import matplotlib.pyplot as plt


def plot_poly(phi, s):
    x = np.linspace(s[0], s[1], 100)
    plt.figure(1)
    plt.plot(x, phi.eval(x))
    plt.legend([i for i in range(deg + 1)])
    plt.title(type(phi.basis).__name__)


def check_orthonormality(phi, g):
    A = np.dot(np.diag(g.weights), phi.eval(g.points))
    G = A.T @ phi.eval(g.points)
    print(
        "Distance from the Gram to identity = ",
        np.linalg.norm(G - np.eye(phi.cardinal())),
    )


# %% Legendre polynomials orthonormal with respect to the uniform probability measure on (-1,1)

X = tp.UniformRandomVariable(-1, 1)
p = X.orthonormal_polynomials()
# evaluate the polynomials of degree 1 and 2 and their derivatives at
# points 0 and 1
print(p.polyval([1, 2], [0, 1]))
print(p.d_polyval([1, 2], [0, 1]))
print(p.dn_polyval(2, [1, 2], [0, 1]))

deg = 5
phi = tp.PolynomialFunctionalBasis(p, range(deg + 1))
print(phi)
s = phi.measure.support()
plot_poly(phi, s)

# check orthonormality
g = phi.measure.gauss_integration_rule(deg + 1)
check_orthonormality(phi, g)

# %% Legendre polynomials orthonormal with respect to the uniform probability measure on (1,3)

X = tp.UniformRandomVariable(1, 3)
p = X.orthonormal_polynomials()
deg = 5
phi = tp.PolynomialFunctionalBasis(p, range(deg + 1))
print(phi)
s = phi.measure.support()
plot_poly(phi, s)

# check orthonormality
g = phi.measure.gauss_integration_rule(deg + 1)
check_orthonormality(phi, g)


# %% Legendre polynomials orthonormal with respect to the Lebesgue measure on (-1,1)

X = tp.LebesgueMeasure(-1, 1)
p = X.orthonormal_polynomials()
deg = 5
phi = tp.PolynomialFunctionalBasis(p, range(deg + 1))
print(phi)
s = phi.measure.support()
plot_poly(phi, s)

# check orthonormality
g = phi.measure.gauss_integration_rule(deg + 1)
check_orthonormality(phi, g)


# %% Legendre polynomials orthonormal with respect to the Lebesgue measure on (1,3)

X = tp.LebesgueMeasure(1, 3)
p = X.orthonormal_polynomials()
deg = 5
phi = tp.PolynomialFunctionalBasis(p, range(deg + 1))
print(phi)
s = phi.measure.support()
plot_poly(phi, s)

# check orthonormality
g = phi.measure.gauss_integration_rule(deg + 1)
check_orthonormality(phi, g)


# %% Hermite polynomials orthonormal with respect to the Normal distribution N(0,1)

X = tp.NormalRandomVariable()
p = X.orthonormal_polynomials()
deg = 5
phi = tp.PolynomialFunctionalBasis(p, range(deg + 1))
print(phi)
s = [-3, 3]
plot_poly(phi, s)

# check orthonormality
g = phi.measure.gauss_integration_rule(deg + 1)
check_orthonormality(phi, g)

# %% Hermite polynomials orthonormal with respect to the Normal distribution N(2,3)

X = tp.NormalRandomVariable(2, 3)
p = X.orthonormal_polynomials()
deg = 5
phi = tp.PolynomialFunctionalBasis(p, range(deg + 1))
print(phi)
s = [-5, 9]
plot_poly(phi, s)

# check orthonormality
g = phi.measure.gauss_integration_rule(deg + 1)
check_orthonormality(phi, g)


# %% Orthonormal polynomials with respect to a discrete measure

X = tp.DiscreteMeasure(np.random.random(10), np.random.randint(1, 10, 10))
p = X.orthonormal_polynomials()
deg = 5
phi = tp.PolynomialFunctionalBasis(p, range(deg + 1))
print(phi)
s = phi.measure.support()
plot_poly(phi, s)
plt.scatter(X.values[:, 0], np.zeros(X.values.shape[0]))

# check orthonormality
g = phi.measure.integration_rule()
check_orthonormality(phi, g)


# %% Orthonormal polynomials with respect to a discrete random variable

points = np.random.random(10)
probabilities = np.random.randint(1, 10, 10)
probabilities = probabilities / np.sum(probabilities)
X = tp.DiscreteRandomVariable(points, probabilities)
p = X.orthonormal_polynomials()
deg = 5
phi = tp.PolynomialFunctionalBasis(p, range(deg + 1))
print(phi)
s = phi.measure.support()
plot_poly(phi, s)
plt.scatter(X.values[:, 0], np.zeros(X.values.shape[0]))

# check orthonormality
g = phi.measure.integration_rule()
check_orthonormality(phi, g)


# %% Empirical polynomials associated with an empirical random variable
# An empirical random variable is a mixture of gaussian random variables,
# obtained from a kernel density estimation of a sample

sample = np.random.random(100) + np.random.random(100)
X = tp.EmpiricalRandomVariable(sample)
p = X.orthonormal_polynomials()
deg = 5
phi = tp.PolynomialFunctionalBasis(p, range(deg + 1))
print(phi)
s = [0, 2]
plot_poly(phi, s)
plt.scatter(X.sample, np.zeros(X.sample.shape[0]))

# check orthonormality
# using gauss integration rule for the empirical measure
g = phi.measure.gauss_integration_rule(10)
check_orthonormality(phi, g)

# using an integration rule that exploits the gaussian mixture property,
# that is a sum of integration rules w.r.t. to gaussian distributions
g = phi.measure.integration_rule(deg + 1)
check_orthonormality(phi, g)
