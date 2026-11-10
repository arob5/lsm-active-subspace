# Notation

Every docstring in `lsmas` uses these symbols with these meanings. A symbol defined only inside one
function (such as the leaf carbon fraction $c$ in a parameter map) is defined there.

## Parameters and coordinates

| Symbol | Meaning |
|---|---|
| $\theta \in \mathbb R^d$ | the calibration parameters, in their own units; $d$ of them |
| $z = b^{-1}(\theta)$ | unconstrained coordinates; $b$ applies each parameter's support bijector (exp, logistic, identity) |
| $\xi = L^{-1}(z - m)$ | whitened coordinates; $m$ and $L L^\top$ are the prior mean and covariance of $z$, so $\xi$ has prior mean $0$ and covariance $I$ |
| $\pi_0$ | the prior; $\pi(\cdot \mid y)$ the posterior |

## Data, forward map and likelihood

| Symbol | Meaning |
|---|---|
| $y \in \mathbb R^n$ | the observed features; $n = 400$ for the Harvard Forest problems |
| $x(\theta) \in \mathbb R^S$ | SIPNET's NEE rate at each of its $S$ output steps |
| $A$ | the observation operator, $n \times S$: each feature averages its steps |
| $G(\theta) = A\, x(\theta)$ | the forward map; written $G(\xi)$ when evaluated at $\theta(\xi)$ |
| $k(i)$ | the class of observation $i$ (day or night) |
| $v_k$, $v_i = v_{k(i)}$ | the reference variance of a class |
| $s^2_k$ | the noise scale of class $k$ |
| $\eta_i \in (0, 1]$ | the learning rate of observation $i$ |
| $\Gamma = \operatorname{diag}\!\left(s^2_{k(i)} v_i / \eta_i\right)$ | the tempered noise covariance: for fixed noise scales, the generalized posterior of $\theta$ is $\pi_0(\theta)\, \mathcal N(y \mid G(\theta), \Gamma)$ up to a constant |

## Ensembles and tempering

| Symbol | Meaning |
|---|---|
| $J$ | the number of particles; particles are indexed by $j$ |
| $\beta \in [0, 1]$ | the tempering level: $\beta = 0$ is the prior, $\beta = 1$ the posterior |
| $\Delta\beta_t$ | the increment taken at step $t$, from level $\beta_t$ to $\beta_{t+1}$ |
| $\mu_\beta$ | the tempered posterior at level $\beta$, $\propto \pi_0(\theta)\, \mathcal N(y \mid G(\theta), \Gamma / \beta)$ |

## Subspaces

| Symbol | Meaning |
|---|---|
| $DG(\xi)$ | the Jacobian of $G$ in whitened coordinates, $n \times d$ |
| $\widehat{DG}$ | a statistical linearization of $G$ over an ensemble, an estimate of an average Jacobian |
| $H(\mu) = \mathbb E_\mu\!\left[DG(\xi)^\top \Gamma^{-1} DG(\xi)\right]$ | the information matrix averaged over a law $\mu$; $d \times d$ |
| $\lambda_1 \ge \dots \ge \lambda_d$ | the eigenvalues of an information matrix |
| $r$ | the rank of a subspace |
| $W$, $W_\perp$ | orthonormal bases of the subspace ($d \times r$) and of its complement ($d \times (d - r)$) |
| $a = W^\top \xi$, $b = W_\perp^\top \xi$ | the active and inactive variables, so $\xi = W a + W_\perp b$ |
