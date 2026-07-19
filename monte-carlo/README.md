# Monte Carlo Integration and Variance Reduction Techniques

## Context

We want to approximate an integral using the Monte Carlo method, then reduce the error using variance reduction techniques.

The main example integral represents 1/4 of the area of a circle:

I = ∫ sqrt(1-x²) dx over [0,1], exact value π/4 ≈ 0.7854

We generate random points xᵢ in the interval and estimate the integral from them.

## 1) Basic Monte Carlo

The simplest approach: draw random points xᵢ in the interval without any special adjustments, compute g(xᵢ) for each, then average them: I ≈ (1/n) Σ g(xᵢ). This is the basic Monte Carlo estimate. The problem with this method is that it converges slowly and has high variance.

### Determining the required sample size

To get an error below 0.01 with 95% confidence, we use:

Error ≤ Z × (σ/√N), with Z = 1.96

Solving for N gives the minimum number of samples needed.

## Variance Reduction Techniques

### 1. Importance Sampling

After the naive Monte Carlo estimate, we improve it by finding a density f(x) that resembles g(x), so we draw more often where g(x) is large instead of wasting random draws in regions where g(x) is small or zero.

We build f(x) from a Taylor expansion of g(x) around 0 up to the x² term, normalize it so it integrates to 1 over [0,1], then draw samples following f(x) via rejection sampling. The estimator becomes:

I ≈ (1/N) Σ g(xᵢ)/f(xᵢ)

### 2. Control Variates

#### Simple control variate

When the target function is complex, we subtract a simpler function h(x) that we know the exact expectation of. Since h resembles f, their difference varies little, giving a lower-variance estimator:

f(U) = [f(U) - h(U)] + E[h(U)]

Applied here to I = ∫exp(x²)dx over [0,1].

#### Sophisticated control variate (two variables)

For a two-variable integral X = exp((U+V)²), we test three candidate control variates — Y1 = U+V, Y2 = (U+V)², Y3 = exp(U+V) — each combined with an optimal coefficient b = Cov(X,Y)/Var(Y), giving the estimator:

I = Xc = X - b(Y - E[Y])

Y3 performs best, since it most closely resembles the growth pattern of X = exp((U+V)²), making it the most strongly correlated control variate of the three.

#### Optimized control variate

Applied to I = ∫1/(1+t²)dt over [0,1] (exact value ln 2), using the optimal coefficient c = (σ_X/σ_Y) × ρ_XY, where ρ_XY is the correlation between X and the control variate Y.

### 3. Antithetic Variables

If a random variable U gives noisy results, we pair it with its counterpart 1-U to balance out fluctuations. Demonstrated on two integrals:

- V = ∫ln(1+x²)·exp(-x) dx from 0 to ∞, handled via the change of variable t = 1-exp(-x) to avoid the divergent original form.
- A second integral, ∫1/(1+x²)dx over [0,1], compared directly against its basic Monte Carlo estimate.

## Figures

| File | Description |
|---|---|
| `figures/importance_sampling_g_vs_f.png` | g(x) plotted against the importance sampling density f(x) for B=0.5 and B=0.75 |
| `figures/variance_comparison_importance_sampling.png` | Variance: basic Monte Carlo vs. importance sampling |
| `figures/variance_comparison_control_variate_simple.png` | Variance: basic Monte Carlo vs. simple control variate |
| `figures/variance_comparison_control_variate_Y1Y2Y3.png` | Variance comparison across the three control variate candidates (Y1, Y2, Y3) |
| `figures/variance_comparison_control_variate_optimized.png` | Variance: basic Monte Carlo vs. optimized control variate |
| `figures/variance_comparison_antithetic_example1.png` | Variance: basic Monte Carlo vs. antithetic variable (example 1) |
| `figures/variance_comparison_antithetic_example2.png` | Variance: basic Monte Carlo vs. antithetic variable (example 2) |