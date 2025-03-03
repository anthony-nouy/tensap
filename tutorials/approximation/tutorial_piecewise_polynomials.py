import numpy as np
import matplotlib.pyplot as plt
import tensap

# %% Piecewise polynomial basis with given points and degrees
points = [0, 0.2, 1]
p = [1, 2]
H = tensap.PiecewisePolynomialFunctionalBasis(points, p)
xplot = np.linspace(0, 1, 1000)
plt.figure(1)
plt.plot(xplot, H.eval(xplot))
plt.legend(np.arange(H.cardinal()))

plt.figure(2)
plt.plot(xplot, H.eval_derivative(1, xplot))
plt.legend(np.arange(H.cardinal()))


# %% Piecewise polynomial basis with constant degree and mesh size
h = 2 ** (-2)
p = 1
H = tensap.PiecewisePolynomialFunctionalBasis.hp(0, 1, h, p)
xplot = np.linspace(0, 1, 1000)
plt.plot(xplot, H.eval(xplot))
plt.legend(np.arange(H.cardinal()))

# %% Piecewise polynomial basis with constant degree and given number of elements
n = 2
p = 4
H = tensap.PiecewisePolynomialFunctionalBasis.np(0, 1, n, p)
xplot = np.linspace(0, 1, 1000)
plt.plot(xplot, H.eval(xplot))
plt.legend(np.arange(H.cardinal()))


# %% Interpolation of a function

H = tensap.PiecewisePolynomialFunctionalBasis.np(0, 1, 4, 3)
f = tensap.UserDefinedFunction("np.cos(4*np.pi*x0)", 1)

# using chebychev points in each interval
g = H.interpolation_points()
If = H.interpolate(f, g)
plt.plot(xplot, If.eval(xplot), xplot, f.eval(xplot))
plt.legend(["If", "f"])

X = tensap.UniformRandomVariable(0, 1)
ERR_L2, ERR_L_INF = f.test_error(If, 1000, X)
print("Mean squared error (chebychev points)  = %2.5e" % ERR_L2)

# using magic points in each interval
g = H.magic_points()
If = H.interpolate(f, g)
ERR_L2, ERR_L_INF = f.test_error(If, 1000, X)
print("Mean squared error (magic points) = %2.5e" % ERR_L2)


# %% Singularity adapted Piecewise polynomial basis
f = tensap.UserDefinedFunction("np.sqrt(x0)", 1)
h = 2 ** (-12)
H = tensap.PiecewisePolynomialFunctionalBasis.singularityhp_adapted(0, 1, [0], h)

xI = H.interpolation_points()
If = H.interpolate(f, xI)
xplot = np.linspace(0, 1, 1000)
plt.plot(xplot, If.eval(xplot), xplot, f.eval(xplot), xI, f.eval(xI), ".")
plt.legend(["log If", "log f"])
X = tensap.UniformRandomVariable(0, 1)
ERR_L2, ERR_L_INF = f.test_error(If, 100, X)
print("Mean squared error (random sample) = %2.5e" % ERR_L2)

g = tensap.IntegrationRule.gauss_legendre_composite(H.points, 20)
L1error = np.dot(g.weights, abs(f.eval(g.points) - If.eval(g.points))) / np.dot(
    g.weights, abs(f.eval(g.points))
)
print("L1 error = %2.5e" % L1error[0])
L2error = np.sqrt(
    np.dot(g.weights, abs(f.eval(g.points) - If.eval(g.points)) ** 2)
    / np.dot(g.weights, abs(f.eval(g.points)) ** 2)
)
print("L2 error = %2.5e" % L2error[0])

# %% Bivariate piecewise polynomials
d = 2
f = tensap.UserDefinedFunction("np.cos(4*np.pi*x0)*np.cos(2*np.pi*x1)", d)
f.evaluation_at_multiple_points = True
h = 2**-4
p = 3
bases = tensap.PiecewisePolynomialFunctionalBasis.hp(0, 1, h, p)
bases = tensap.FunctionalBases.duplicate(bases, d)
H = tensap.FullTensorProductFunctionalBasis(bases)
If, OUTPUT = H.tensor_product_interpolation(f)
ERR_L2, ERR_L_INF = f.test_error(If, 100, If.measure)

print("Mean squared error = %2.5e" % ERR_L2)

f.measure = If.measure
axf = f.surf([100, 100])
axf.set_title("f")
axIf = If.surf([100, 100])
axIf.set_title("If")
