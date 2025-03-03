import numpy as np
import matplotlib.pyplot as plt
import tensap

# %% Cardinal BSplines
plt.figure(1)
mmax = 5
fig, axes = plt.subplots(2, mmax, figsize=(15, 5))
xplot = np.linspace(0, mmax + 1, 400)
for m in range(mmax):
    h = tensap.BSplinesFunctionalBasis.cardinal_bspline(m)
    axes[0, m].plot(xplot, h.eval(xplot))
    axes[0, m].set_title(f"B_{m}")
    axes[0, m].set_xlim([0, m + 1])
    axes[1, m].plot(xplot, h.eval_derivative(1, xplot))
    axes[1, m].set_title(f"dB_{m}/dx")
    axes[1, m].set_xlim([0, m + 1])
plt.show()

# %% BSplines with extra knots
knots = np.linspace(-1, 1, 3)
s = 3  # degree
h = tensap.BSplinesFunctionalBasis.with_extra_knots(knots, s)

xplot = np.linspace(-1, 1, 400)

# Evaluate B-splines and their derivatives at the points in xplot
B_spline_values = h.eval(xplot)
B_spline_deriv_1 = h.eval_derivative(1, xplot)
B_spline_deriv_2 = h.eval_derivative(2, xplot)

# Create figure and subplots
plt.figure(1)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# First subplot: B-Splines
axes[0].plot(xplot, B_spline_values)
axes[0].set_title("BSplines")
axes[0].set_xlim([-1, 1])

# Second subplot: First derivative of B-Splines
axes[1].plot(xplot, B_spline_deriv_1)
axes[1].set_title("BSplines first derivative")
axes[1].set_xlim([-1, 1])

# Third subplot: Second derivative of B-Splines
axes[2].plot(xplot, B_spline_deriv_2)
axes[2].set_title("BSplines second derivative")
axes[2].set_xlim([-1, 1])

# Show the plot
plt.tight_layout()
plt.show()

# %% Interpolation of a function
knots = np.linspace(-1, 1, 30)
s = 4
H = tensap.BSplinesFunctionalBasis.with_extra_knots(knots, s)
f = tensap.UserDefinedFunction("np.cos(4*np.pi*x0)", 1)

X = tensap.UniformRandomVariable(-1, 1)
a = H.magic_points(np.linspace(-1, 1, 1000))[0]
If = H.interpolate(f, a)

plt.plot(xplot, If.eval(xplot), xplot, f.eval(xplot), a, f.eval(a), ".")
plt.legend(["If", "f"])

ERR_L2, ERR_L_INF = f.test_error(If, 1000, X)
print("Mean squared error (magic points)  = %2.5e" % ERR_L2)


# %% Dilated BSplines
s = 0  # degree
r = 2  # resolution
b = 2  # base of dilation
h = tensap.DilatedBSplinesFunctionalBasis.with_level_bounded_by(s, b, r)

plt.figure(2)
plt.clf()
x = np.linspace(0, 1, 100)
plt.plot(x, h.eval(x))
plt.legend(range(h.cardinal()))


# %% Dilated BSplines
s = 2  # degree
r = 2  # resolution
b = 2  # base of dilation
h = tensap.DilatedBSplinesFunctionalBasis.with_level_bounded_by(s, b, r)

plt.figure(3)
x = np.linspace(0, 1, 100)
y = h.eval(x)
plt.plot(x, y)
plt.title("BSplines")

plt.figure(4)
y_derivative_1 = h.eval_derivative(1, x)
plt.plot(x, y_derivative_1)
plt.title("BSplines first derivative")

plt.figure(5)
# Placeholder for second derivative evaluation
y_derivative_2 = h.eval_derivative(2, x)  # Example second derivative
plt.plot(x, y_derivative_2)
plt.title("BSplines second derivative")
