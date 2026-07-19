import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

os.makedirs("figures", exist_ok=True)

# ============================================================
# We want to approximate an integral using the Monte Carlo
# method, then reduce the error using a special technique
# called importance sampling. The integral we'll use
# represents 1/4 of the area of a circle, and its exact
# value is pi/4 = 0.7854
#
# I = ∫ sqrt(1-x²) dx over [0,1]
#
# So what we do is generate random points xi within this
# interval.
# ============================================================

# ============================================================
# BASIC MONTE CARLO
# ============================================================
# Here it's as if we want to estimate the integral just by
# drawing random points (xi) in the interval, without doing
# complicated calculations. For each one we compute g(xi),
# then we make an estimate with the formula (1/n) x sum of
# g(xi), and that's the Monte Carlo estimate. The problem
# with this method is that it converges slowly and has a
# lot of variance.

N = 1921

def g(x):
    return np.sqrt(1 - x**2)

xi = np.random.uniform(0, 1, N)
yi = g(xi)
I = (1 / N) * sum(yi)
exact = np.pi / 4
erreur = abs(I - exact)
ecart = np.std(yi, ddof=1)  # ddof=1 for an unbiased estimator
erreur_standard = ecart / np.sqrt(N)
variance = ecart**2  # since variance = standard deviation squared

print("Integral estimate:", I)
print("Exact value (pi/4):", exact)
print("Absolute error:", erreur)
print("Standard deviation:", ecart)
print("Variance:", variance)

# Now we want to find the right N so that our error is below
# 0.01 with 95% confidence. We follow the formula:
# Error <= Z x (sigma/sqrt(N)), with Z=1.96, sigma²=0.05 as
# given in the problem statement.

z = 1.96                  # Quantile for 95% confidence
sigma = np.sqrt(0.05)     # Standard deviation given in the problem
erreur_max = 0.01         # Maximum error
N_min = (z * sigma / erreur_max) ** 2
N_min = int(np.ceil(N_min))  # round up

print("Number of samples needed for error below 1e-2 at 95% confidence:", N_min)


# ============================================================
# VARIANCE REDUCTION FOR MONTE CARLO — DIFFERENT METHODS
# ============================================================

# ------------------------------------------------------------
# 1) IMPORTANCE SAMPLING
# ------------------------------------------------------------
# After using the naive Monte Carlo method, we improve it with
# a specific technique to reduce its variance: importance
# sampling. This doesn't mean computing the integral
# differently, it means finding a function f(x) that resembles
# g(x). The idea is that when using importance sampling, we
# want to draw more often where g(x) is large — but for that
# we need to invent a density f(x) that resembles g(x), since
# we can't directly draw from g(x) (that's already what basic
# Monte Carlo did, and we're trying to improve on it — we were
# drawing in useless zones, for example where g(x) is close to
# zero, which is like wasting our randomness).
#
# So we expand g(x) as a Taylor series around 0:
#
# g(x) = g(0) + g'(0).x + (g''(0)/2!).x² + (g'''(0)/3!).x³ + ...
#
# Evaluating around 0: g(0) = sqrt(1-0²) = 1, g'(0) = 0,
# g''(0) = -1, so the Taylor series up to x² is:
# g(x) = 1 + 0 + (-1/2).x² = 1 - (1/2).x², generally written
# as (1 - B.x²).
#
# Now we want a probability density f(x) that is: positive or
# zero over the whole interval, and integrates to 1, i.e.
# ∫f(x)dx = 1 over (0,1). So we compute ∫(1-Bx²) over (0,1),
# which gives 1-(B/3) ≠ 1. Since the area is 1-(B/3), we divide
# the whole function by this so the area becomes exactly 1:
# f(x) = (1-Bx²) / (1-(B/3))
#
# Here we can't just reuse the g(x) code from before, because
# in basic Monte Carlo we drew xi uniformly on [0,1], but here
# we want to draw xi following the density f(x). The estimator
# becomes (1/N).sum of g(xi)/f(xi), so we want x values between
# 0 and 1, but not all equally likely — values where f(x) is
# large should come up more often (rejection sampling, not
# uniform sampling).

def f_importance(x, B):
    return (1 - B * x**2) / (1 - B / 3)

accept = []
N = 1921
B = 0.5
while len(accept) < N:
    xii = np.random.uniform(0, 1)
    u = np.random.uniform(0, 1)
    if u <= f_importance(xii, B):
        accept.append(xii)

# Now we estimate the integral
II = sum(g(x) / f_importance(x, B) for x in accept) / N
poids = [g(x) / f_importance(x, B) for x in accept]
variancef = np.var(poids, ddof=1)
ecartf = np.sqrt(variancef / N)
erreurf = exact - II

print("Integral estimate with importance sampling:", II)
print("Variance with importance sampling:", variancef)
print("Standard deviation with importance sampling:", ecartf)
print("Absolute error with importance sampling:", erreurf)

# Plot of g(x), f(x) for B=0.5 and B=0.75, plus both variances

x_vals = np.linspace(0, 1, 300)
g_vals = g(x_vals)
f05 = f_importance(x_vals, 0.5)
f075 = f_importance(x_vals, 0.75)

plt.plot(x_vals, g_vals, label="g(x) = sqrt(1 - x²)", linewidth=2, color='#ffb6c1')
plt.plot(x_vals, f05, label="f(x), B=0.5", linestyle='--', color='#ffb6c1')
plt.plot(x_vals, f075, label="f(x), B=0.75", linestyle='--', color='#b2f2bb')
plt.title("g(x) and importance sampling functions")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.savefig("figures/importance_sampling_g_vs_f.png")
plt.show()

methodes = ['Basic Monte Carlo', 'Importance Sampling']
variances = [variance, variancef]
plt.bar(methodes, variances, color=['#d8b7dd', '#ffb6c1'])
plt.ylabel("Variance")
plt.title("Variance comparison between the two methods")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("figures/variance_comparison_importance_sampling.png")
plt.show()


# ------------------------------------------------------------
# 2) CONTROL VARIATE
# ------------------------------------------------------------
# E = mathematical expectation (when we run many random trials
# and compute the average result over the long run). We want to
# compute the integral I = ∫exp(x²) dx from 0 to 1, but we
# don't know how to compute this area easily with a formula, so
# we use random numbers to approximate it. We draw many random
# numbers U1, U2, ... between 0 and 1, and for each one we
# compute exp(U²), then average all these values. This average
# is an estimate of the area using basic Monte Carlo.
#
# Now we want to improve the variance with another method
# called control variate. Here, unlike importance sampling
# (where we try to draw more where g(x) is large, i.e. change
# how xi is drawn instead of uniform, following f tilde), in
# the control variate method, since the function is complicated,
# we subtract a simpler function h(x) that we know exactly.
# So we keep the uniform random draws (U), but we transform:
# f(U) = [f(U) - h(U)] + exact value of E[h(U)]
# Since h resembles f, their difference varies little -> less
# noise -> better estimate.
#
# basic Monte Carlo here: I = E[exp(U²)] = (1/N).sum(exp(U²))
# simple control variate: f(U) = [f(U)-h(U)] + exact value of E[h(U)]

# --- Basic Monte Carlo ---

def k(x):
    return np.exp(x**2)

N = 1000
U = np.random.uniform(0, 1, N)
yy = k(U)  # fixed: was k(U**2), which computed exp(U^4) instead of exp(U^2)
I3 = (1 / N) * sum(yy)
aa = np.std(yy, ddof=1)
ecart2 = aa / np.sqrt(N)  # fixed: was a/np.sqrt(N), 'a' was undefined
variance2 = aa**2

print("Integral estimate with basic Monte Carlo:", I3)
print("Standard deviation, basic Monte Carlo:", ecart2)
print("Variance, basic Monte Carlo:", variance2)

# --- Simple Control Variate ---

def h(x):
    return 1 + x**2

Ih = (1 / N) * sum(h(U))  # theoretically 4/3 if we integrate h(x) from 0 to 1
Ih_theor = 4 / 3
I4 = k(U) - h(U) + Ih_theor
I4f = sum(I4) / N
ecart4 = np.std(I4, ddof=1)
variance4 = ecart4**2

print("Integral estimate with simple control variate:", I4f)
print("Variance with simple control variate:", variance4)
print("Standard deviation with simple control variate:", ecart4)

methodess = ['Basic Monte Carlo', 'Control Variate']
variancess = [variance2, variance4]
plt.bar(methodess, variancess, color=['#f7c937', '#f76637'])
plt.ylabel("Variance")
plt.title("Variance comparison between the two methods")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("figures/variance_comparison_control_variate_simple.png")
plt.show()


# --- Sophisticated Control Variate ---
# Now we want to estimate a new integral X = exp((U+V)²), a
# two-variable function, so the integral will be a double
# integral, of the form E[X] = E[exp((U+V)²)] (equivalent to
# writing ∫∫f(U,V)dU dV). Let's first do it with basic Monte
# Carlo.

N = 1000
U1 = np.random.uniform(0, 1, N)
V1 = np.random.uniform(0, 1, N)

def X(U, V):
    return np.exp((U + V)**2)

Y = X(U1, V1)
I5 = (1 / N) * sum(Y)
ecart5 = np.std(Y, ddof=1)
variance5 = ecart5**2
erreur5 = ecart5 / np.sqrt(N)

print("Integral estimate with basic Monte Carlo:", I5)
print("Standard deviation:", ecart5)
print("Variance:", variance5)
print("Standard error:", erreur5)

# Control variates Y1, Y2, Y3
# As a reminder: estimating the expectation of X (E[X]) is the
# same as estimating the integral of X. Here we call this
# theta (theta = E[X], which is (1/N)*sum(xi) in basic Monte
# Carlo), and I (the integral estimate) is called Xc. But since
# our goal here is precisely to improve on the previous control
# variate method, we don't just set I = E[X(U)-Y(U)+E[Y]] like
# we did in the simplified case with k(x) and h(x) — here there
# will be a coefficient b that further improves the variance.
# b is computed as: b = Cov(X,Y)/Var(Y), in code:
# b = np.cov(X, Y, ddof=1)[0, 1] / np.var(Y, ddof=1)
# and the estimate becomes: I = Xc = X - b(Y - E[Y])

# Y1 = U + V
def Y1(U, V):
    return U + V

N = 1000
U1 = np.random.uniform(0, 1, N)
V1 = np.random.uniform(0, 1, N)
Y1y = Y1(U1, V1)
Xy = X(U1, V1)
EY1 = (1 / N) * sum(Y1y)
b1 = np.cov(Xy, Y1y, ddof=1)[0, 1] / np.var(Y1y, ddof=1)
Xc1 = Xy - b1 * (Y1y - EY1)
Ifinal1 = (1 / N) * sum(Xc1)
ecartY1 = np.std(Xc1, ddof=1)
varianceY1 = ecartY1**2
erreurY1 = ecartY1 / np.sqrt(N)

print("Integral estimate with control variate Y1=U+V:", Ifinal1)
print("Standard deviation:", ecartY1)
print("Variance:", varianceY1)
print("Standard error:", erreurY1)

# Y2 = (U+V)²
def Y2(U, V):
    return (U + V)**2

N = 1000
U1 = np.random.uniform(0, 1, N)
V1 = np.random.uniform(0, 1, N)
Y2y = Y2(U1, V1)
Xy = X(U1, V1)
EY2 = (1 / N) * sum(Y2y)
b2 = np.cov(Xy, Y2y, ddof=1)[0, 1] / np.var(Y2y, ddof=1)
Xc2 = Xy - b2 * (Y2y - EY2)
Ifinal2 = (1 / N) * sum(Xc2)
ecartY2 = np.std(Xc2, ddof=1)
varianceY2 = ecartY2**2
erreurY2 = ecartY2 / np.sqrt(N)

print("Integral estimate with control variate Y2=(U+V)^2:", Ifinal2)
print("Standard deviation:", ecartY2)
print("Variance:", varianceY2)
print("Standard error:", erreurY2)

# Y3 = exp(U+V)
def Y3(U, V):
    return np.exp(U + V)

N = 1000
U1 = np.random.uniform(0, 1, N)
V1 = np.random.uniform(0, 1, N)
Y3y = Y3(U1, V1)
Xy = X(U1, V1)
EY3 = (1 / N) * sum(Y3y)
b3 = np.cov(Xy, Y3y, ddof=1)[0, 1] / np.var(Y3y, ddof=1)
Xc3 = Xy - b3 * (Y3y - EY3)
Ifinal3 = (1 / N) * sum(Xc3)
ecartY3 = np.std(Xc3, ddof=1)
varianceY3 = ecartY3**2
erreurY3 = ecartY3 / np.sqrt(N)

print("Integral estimate with control variate Y3=exp(U+V):", Ifinal3)
print("Standard deviation:", ecartY3)
print("Variance:", varianceY3)
print("Standard error:", erreurY3)

methodesy = ['Basic Monte Carlo', 'Control Variate Y1=U+V', 'Control Variate Y2=(U+V)^2', 'Control Variate Y3=exp(U+V)']
variancesy = [variance5, varianceY1, varianceY2, varianceY3]
plt.bar(methodesy, variancesy, color=['#fbb1bd', '#cdb4db', '#ffdac1', '#b5ead7'])
plt.ylabel("Variance")
plt.title("Variance comparison using control variates")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/variance_comparison_control_variate_Y1Y2Y3.png")
plt.show()

# We immediately notice that Y3 performs best, probably
# because it resembles our studied function X=exp((U+V)²) the
# most closely — it follows the same growth pattern, which is
# why it's well correlated with X.


# --- Estimating another integral with control variate ---
# Here we proceed as before: basic Monte Carlo, then Monte
# Carlo with control variate, but this time we use a better
# optimization than the previous versions. We also change some
# notation here: we introduce an optimized formula to reduce
# the variance even further, with an optimal choice of the
# coefficient c:
# X* = X - c(Y - theta)  (this is the estimate I)
# with c = cov(X,Y)/var(Y)
# sigma_x = standard deviation of X
# sigma_y = standard deviation of Y
# rho_XY = correlation between X and Y = cov(X,Y)/(sigma_X sigma_Y)
# cov(X,Y) = covariance between X and Y
# theta = expectation of Y, i.e. E[Y]

# --- Basic Monte Carlo ---

N = 10_000_000

def f_t(t):
    return 1 / (1 + t**2)

xt = np.random.uniform(0, 1, N)
yt = f_t(xt)
It = (1 / N) * sum(yt)
exactT = np.log(2)
erreurT = abs(It - exactT)
sigmaX = np.std(yt, ddof=1)
varianceT = sigmaX**2
erreursT = sigmaX / np.sqrt(N)

print("Expectation (integral) estimate with basic Monte Carlo:", It)
print("Exact value:", exactT)
print("Absolute error:", erreurT)
print("Standard error:", erreursT)
print("Variance:", varianceT)
print("Standard deviation sigma X:", sigmaX)

# --- Optimized control variate ---

N = 150_000
Ut = np.random.uniform(0, 1, N)
yt = f_t(Ut)
It = (1 / N) * sum(yt)
exactT = np.log(2)
sigmaX = np.std(yt, ddof=1)

def Yt(U):
    return 1 + U

Yi = Yt(Ut)
theta = (1 / N) * sum(Yi)
sigmaY = np.std(Yi, ddof=1)

rhoXY = np.cov(yt, Yi, ddof=1)[0, 1] / (np.std(yt, ddof=1) * np.std(Yi, ddof=1))
c = (sigmaX / sigmaY) * rhoXY
Xstar = yt - c * (Yi - theta)
Ifinalstar = (1 / N) * sum(Xstar)
covariance = np.cov(yt, Yi)[0, 1]

ecartstar = np.std(Xstar, ddof=1)
varXstar = (1 - rhoXY**2) * sigmaX**2
erreurstar = abs(Ifinalstar - exactT)

print("Expectation (integral estimate) with optimized control variate:", Ifinalstar)
print("Exact value:", exactT)
print("Absolute error:", erreurstar)
print("Standard deviation:", ecartstar)
print("Variance:", varXstar)
print("Correlation between f(t) and Y:", rhoXY)
print("Sigma X:", sigmaX)
print("Sigma Y:", sigmaY)
print("theta:", theta)
print("sigmaY**2:", sigmaY**2)

methodesg = ['Basic Monte Carlo', 'Optimized Control Variate']
variancesg = [varianceT, varXstar]
plt.bar(methodesg, variancesg, color=['#76dcff', '#5d20fd'])
plt.ylabel("Variance")
plt.title("Variance comparison between the two methods")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("figures/variance_comparison_control_variate_optimized.png")
plt.show()


# ------------------------------------------------------------
# 3) ANTITHETIC VARIABLES
# ------------------------------------------------------------
# The idea is simple: if the random variable U gives noisy
# results, we use its counterpart 1-U to balance out the
# fluctuations. Let's estimate:
# V = ∫ln(1+x²).exp(-x) dx from 0 to infinity
# but instead we're asked to do a change of variable, since
# otherwise the integral is explosive and unpleasant to work
# with directly. We're given t = 1-exp(-x), so dt = exp(-x)dx.
# For the bounds: when x=0, t=0, and as x tends to infinity, t
# tends to 1, so x = -ln(1-t). The integral becomes:
# V = ∫ln(1+x²) dt from 0 to 1, using x=-ln(1-t).

# --- Basic Monte Carlo ---

def g_ex1(t):
    x = -np.log(1 - t)
    return np.log(1 + x**2)

N = 10000
U = np.random.uniform(0, 1, N)
Y = g_ex1(U)
I = (1 / N) * sum(Y)
var = np.var(Y, ddof=1)

print("Expectation, basic Monte Carlo:", I)
print("Variance:", var)

# --- Antithetic variable ---

UA = 1 - U
YA = g_ex1(UA)
W = (Y + YA) / 2
IA = np.mean(W)
varA = np.var(W, ddof=1)  # variance of the estimators

print("Expectation, antithetic variable:", IA)
print("Variance, antithetic variable:", varA)

methodesg = ['Basic Monte Carlo', 'Antithetic Variable']
variancesg = [var, varA]
plt.bar(methodesg, variancesg, color=['#8a2982', '#c1235f'])
plt.ylabel("Variance")
plt.title("Variance comparison between the two methods")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("figures/variance_comparison_antithetic_example1.png")
plt.show()


# --- SECOND EXAMPLE ---

# --- Basic Monte Carlo ---

N = 1500
U = np.random.uniform(0, 1, 2 * N)
EXACT = np.log(2)

def F(x):
    return 1 / (1 + x**2)

Y = F(U)
I = (1 / (2 * N)) * sum(Y)
variance = np.var(Y, ddof=1)

print("Basic Monte Carlo estimator:", I)
print("Variance:", variance)

# --- Antithetic ---

u = np.random.uniform(0, 1, N)
ua = 1 - u
Y1_ex2 = F(u)
Y2_ex2 = F(ua)
D = (Y1_ex2 + Y2_ex2) / 2
Ia = np.mean(D)
varr = np.var(D, ddof=1)

print("Antithetic variable estimator:", Ia)
print("Variance:", varr)

methodesg = ['Basic Monte Carlo', 'Antithetic Variable']
variancesg = [variance, varr]
plt.bar(methodesg, variancesg, color=['yellow', 'red'])
plt.ylabel("Variance")
plt.title("Variance comparison between the two methods")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("figures/variance_comparison_antithetic_example2.png")
plt.show()
