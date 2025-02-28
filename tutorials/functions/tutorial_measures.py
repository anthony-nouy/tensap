# This file is part of tensap (tensor approximation package).

"""
Tutorial on measure and random variables.

"""

import tensap as tp
import numpy as np
import matplotlib.pyplot as plt


# %% Uniform probability measure on (a,b)

a = 1
b = 3
X = tp.UniformRandomVariable(a, b)

# support
print("support ", X.support())
print("truncated support ", X.truncated_support())

# plot
plt.figure()
plt.title("probability density function")
X.pdf_plot()

plt.figure()
plt.title("cumulative distribution function")
X.cdf_plot()

plt.figure()
plt.title("inverse cumulative distribution function (quantile function)")
X.icdf_plot()

# probability density function
print("X.pdf(2) ", X.pdf([2]))
# cumulative distribution function
print("X.cdf(2) ", X.cdf([2]))
# inverse cumulative distribution function (quantile function)
print("X.icdf(.5) ", X.icdf([0.5]))

# pseudo-random generator
print("X.random(2)= \n", X.random(2))

# latin hypercube sampling
print("X.lhs_random(2) = \n", X.lhs_random(2))

# %% Normal (Gaussian) probability measure with mean mu and standard deviation sigma
mu = 1
sigma = 3
X = tp.NormalRandomVariable(mu, sigma)

# support
print("support ", X.support())
print("truncated support ", X.truncated_support())
print("truncated support with probability 1-2e-8", X.truncated_support(p=1 - 2e-8))


# plot
plt.figure()
plt.title("probability density function")
X.pdf_plot()

# %% Gauss quadrature
X = tp.NormalRandomVariable(0, 2)
G = X.gauss_integration_rule(6)


def fun(x):
    return x**2


v = np.sum(fun(G.points) * G.weights)
print("E(X^2) = ", v)
print("integration error = ", np.abs(v - X.variance()) / X.variance())


# %% Discretisation of a random variable

X = tp.NormalRandomVariable(0, 1)

# returns a discrete random variable taking n values with equal probability
Xd = X.discretize(1000)
plt.figure()
Xd.pdf_plot()


def fun(x):
    return x**2


G = Xd.integration_rule()
v = G.integrate(fun)
print("E(Xd^2) = ", v)
print("integration error = ", np.abs(v - X.variance()) / X.variance())

# %% Discretisation of a random variable from a discretisation of its support

X = tp.NormalRandomVariable(0, 1)

# returns a discrete random variable taking n values from a uniform discretisation
# of a truncated support
Xd2 = X.discretize_support(50)
plt.figure(1)
plt.title("uniform discretization of the truncated support")
Xd2.pdf_plot()


Xd2 = X.discretize_support(50, [-8, 8])

plt.figure()
plt.title("uniform discretization of the provided support")
Xd2.pdf_plot()

# returns a discrete random variable taking n specified values
Xd3 = X.discretize_support(np.linspace(-6, 6, 100))
plt.figure()
plt.title("given discretization")
Xd3.pdf_plot()


# %% Random vector

X1 = tp.UniformRandomVariable(-1, 1)
X2 = tp.NormalRandomVariable(1, 1)
X = tp.RandomVector([X1, X2])
print("random vector X")
print(X)
print("X.random(3) = \n", X.random(3))
print("X.lhs_random(3) = \n", X.lhs_random(3))

# duplicate a random variable
X = tp.RandomVector(X1, 5)
print("random vector X")
print(X)
print("X.random(3) = \n", X.random(3))

# %% Lebesgue Measure
L = tp.LebesgueMeasure(2, 5)

# %% Product Measure
m1 = tp.UniformRandomVariable()
m2 = tp.NormalRandomVariable()
m3 = tp.LebesgueMeasure(-1, 1)
m = tp.ProductMeasure([m1, m2, m3])
print("product measure\n", m)

# get the marginals
print("marginal 1\n", m.marginal([1]))
print("marginal [1,2]\n", m.marginal([1, 2]))

# tensorize a random variable -> returns a random vector
m = tp.ProductMeasure.duplicate(m1, 3)
print("duplicate a random variable\n", m)

# tensorize a measure -> returns a product measure
m = tp.ProductMeasure.duplicate(m3, 3)
print("duplicate a Lebesgue measure\n", m)

# %%
