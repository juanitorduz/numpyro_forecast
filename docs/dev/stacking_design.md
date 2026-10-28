# Design: model stacking for `numpyro_forecast`

Status: design, with a throwaway prototype (Appendix A), an independent adversarial review whose findings changed the design (Section 8), and a type-checked scaffold on the `stacking-scaffold` branch (signatures, docstrings, stubbed kernels, and the test suite; see `docs/dev/stacking_scaffolding.md`). No kernel is implemented yet.

Scope: how to combine several `numpyro_forecast` models into one forecast distribution with weights learned from backtests, following Yao, Vehtari, Simpson and Gelman (2018) for stacking of predictive distributions and Yao, Pirš, Vehtari and Gelman (2022) for hierarchical (input-dependent) stacking weights, adapted to the time-ordered, horizon-indexed setting of a forecasting library.

## 1. Summary

The package has every ingredient of stacking but one. `backtest()` produces the rolling-origin, leave-future-out evaluation the stacking literature asks for in time series (Yao et al. 2022, Section 2.5), `forecast()` produces the forecast draws a mixture recombines, and the metrics module scores draws. What is missing is a **pointwise log predictive density per backtest cell**: everything the evaluation stack computes today is sample-based (MAE, RMSE, CRPS, coverage, pinball, interval score, MASE), and `to_datatree()` writes no `log_likelihood` group, so neither the package nor ArviZ can run log-score stacking on these models as they stand.

The design adds three layers, each usable without the next:

1. **A density driver.** `forecast_log_density()` mirrors `forecast()` and returns, for every posterior draw, the log density of the realized future values under the model's forecast distribution given the sampled future latent path, shape `(sample, *batch, future, obs)`. `reduce_log_density()` turns it into one of two h-step scores, `log_density_ess()` diagnoses the reduction, and `conditional_log_density()` is the exact chain-rule route for error-feedback models. `backtest(log_density=...)` stores the reduced cells on each `BacktestResult` without the closure ever seeing the test block: `forecast_fn` returns a `ForecastOutput(draws, posterior)` and `backtest` scores the posterior itself. `backtest_vectorized(log_density=...)` does the same inside its fused pass. A `DensityReader` covers the two kinds of models in the package: `predict_density` for models built on `predict()`, which register an `obs_future` site, and `ssoe_density(name)` for additive models built on `ssoe()`, which register a deterministic `forecast` and need one more deterministic.
2. **Weight solvers.** Pure JAX functions on a `(cells, models)` matrix: `stacking_weights()` (complete-pooling stacking of predictive distributions), `pseudo_bma_weights()` (the pseudo-BMA+ baseline, bootstrapped over windows), `crps_stacking_weights()` with `mixture_crps_terms()` (a convex QP that needs draws only), `log_score()` for held-out comparisons with a paired standard error across windows, and `window_design_effect()` to measure how dependent the cells of a window are. `stacking_cells()` builds the matrix from per-model backtest results.
3. **Hierarchical weights as model building blocks.** `hierarchical_weights()` samples the partial-pooling parameters of Yao et al. (2022) and returns input-dependent log weights; `stacking_factor()` registers the stacking likelihood. A user's stacking model is three lines that call them, fit with their own `MCMC` or `SVI`, exactly as a forecasting model calls `innovations()` and `predict()`. `rectified_features()` builds the paper's median-split features; `stack_forecasts()` samples the weighted mixture of forecast draws.

The decisions that shape everything else: densities come from the model's own `obs_future` site (no re-implementation of the likelihood); the per-cell quantity is the **horizon marginal** $\log p(y_{t+j} \mid y_{\le t})$, the distribution the forecast draws represent and the one that stacks per horizon step, with the joint block density available under its own name but not the default; weights are learned from backtest cells and never from PSIS-LOO; the **window is the unit of exchangeability**, so every standard error, bootstrap and likelihood tempering is per window, never per cell; closures never receive the test block; and everything new is a plain function with `rng_key` first that composes with the existing drivers, or a building block called inside a user model, never a stacker object or a shipped model function.

## 2. Context

### 2.1 What the package has

A model is a plain function `(covariates, data=None) -> None` built from blocks. `predict(h, obs_dist, prediction)` registers `obs` over the training window and, when the horizon is positive, samples `obs_future` from `prefix_condition(full_dist, data)` and exposes it as the `forecast` deterministic. The future latents of `innovations` and `markov_series` live on separate `_future` sites that `Predictive` draws from the prior. `ssoe()` samples future errors at `{name}_future` and returns `SSOEResult(mu, mu_future, y_future)`; the caller writes the likelihood against `mu` and registers `forecast` as a deterministic, so these models expose no density over the horizon. Eleven of the sixteen example notebooks use `predict()`; the five single-source-of-error examples (ARMA, Croston, TSB, exponential smoothing, VAR) and the availability TSB use the deterministic route.

`backtest()` iterates `(t0, t1, t2)` windows (expanding or rolling), calls the user's `forecast_fn(rng_key, model, train_data, train_covariates, full_covariates, num_samples, *, batch_size)` and scores the returned draws against `data[t1:t2]` with sample metrics; the closure never receives the test block, which is what makes a backtest leak-free by construction. `backtest_vectorized()` fits every window in one vmapped SVI run with a shared guide and owns the posterior of every window. Metrics are pure JAX scalar kernels `(pred, truth) -> Array` so the vectorized path can `vmap` them.

### 2.2 What the papers say, and what transfers

**Stacking of predictive distributions** (Yao et al. 2018). Given $K$ models and held-out predictive densities $p_{k,-i} = p(y_i \mid y_{-i}, M_k)$, choose $w$ on the simplex $S_K$ to maximize $\sum_i \log \sum_k w_k \, p_{k,-i}$. The combined predictive is $\sum_k w_k \, p(\tilde y \mid y, M_k)$. The densities come from PSIS-LOO so no refit is needed; the paper recommends exact LOO or $K$-fold where the Pareto $\hat k$ exceeds 0.7. Stacking is recommended in the M-open setting over BMA (which asymptotically selects one model) and over pseudo-BMA (which inherits the instability of the point elpd); pseudo-BMA+ with the Bayesian bootstrap is the cheap alternative. The paper assumes exchangeable data and does not treat time series.

**Bayesian hierarchical stacking** (Yao et al. 2022). The weights become a function of the input, $w(x) = \text{softmax}(\alpha(x))$ with $\alpha_K \equiv 0$ as the reference. For a discrete input with $J$ cells, $\alpha_{jk} \sim \text{Normal}(\mu_k, \sigma_k)$ with hyperpriors $\mu_k \sim \text{Normal}(0, \tau_\mu)$ and $\sigma_k \sim \text{Normal}^+(0, \tau_\sigma)$ (eqs. 8 to 10), so complete pooling ($\sigma_k \to 0$) and no pooling ($\sigma_k \to \infty$) are the limiting cases. Continuous inputs enter additively, $\alpha_k(x) = \mu_k + \sum_m \beta_{mk} f_m(x)$ with $\beta_{mk} \sim \text{Normal}(0, \sigma_k)$ (eqs. 13 and 14), through rectified features $(x_d - \text{med}(x_d))_+$ and $(\text{med}(x_d) - x_d)_+$ (eq. 16). The log posterior is the stacking objective evaluated with $w(x_i)$ plus the log prior (eq. 11); the MAP is degenerate on the complete-pooling subspace, so the model is fit by MCMC and the **posterior mean of the weights** is reported. The time-series section (2.5) replaces LOO with expanding-window one-step-ahead predictive densities, optionally reweights the likelihood by a nondecreasing sequence $\pi_i$ to emphasize recent data, and appends time to the input so the weights can drift. The election case study is an expanding-window one-week-ahead backtest where hierarchical stacking beats complete pooling, no pooling and model selection, with the largest gains early in the season and in small cells. The paper also gives an approximate LOO for the stacking model itself by PSIS-reweighting its posterior draws (eq. 28), and notes that hierarchical stacking weights are generally not sparse.

**The NumPyro tutorial** implements the well-switching case study: six logistic regressions fit with NUTS, `az.loo(..., pointwise=True)` for the pointwise LOO log densities, a `stacking` model function with the eqs. 8 to 10 priors (non-centered) over four education cells plus rectified continuous features, NUTS on the stacking model, and `trace["w_test"].mean(axis=0)` as the test-set weights.

What transfers directly: the objective, the softmax parameterization, the hierarchical priors, the rectified features, MCMC on the stacking model, and the posterior-mean weights. What must change for forecasting: the held-out densities must be leave-future-out, which `backtest()` already produces; the natural discrete input is the **horizon step** (and the series, for panels), which has no analogue in the paper; the cells of one window are dependent because they share one realized future path, which the paper's one-step-ahead windows never are; and the forecast distribution over a block of $H$ steps admits two different scores (Section 3.2).

### 2.3 What ArviZ offers and why it is not enough here

`arviz_stats.compare(..., method="stacking")` solves the complete-pooling stacking problem, but on `ELPDData` objects produced by `loo()` from a `log_likelihood` group, and `loo()` is PSIS-LOO: on a time series it conditions on the future. `arviz_stats.loo.lfo_cv` computes leave-future-out elpd (Bürkner et al. 2020) by carrying the posterior forward with PSIS and refitting on demand, but it needs a `SamplingWrapper` that refits the model, scores the chain-rule block density, and returns one value per forecast origin, not per horizon step. None of this reaches hierarchical stacking. The package should produce the cell-level densities itself from its backtests and solve the weights in JAX; `compare` on hand-built `ELPDData` from those cells is a cross-check the test suite keeps (the reviewer verified it agrees with the prototype to three decimals). Exporting a `log_likelihood` group from `to_datatree()` is an independent improvement (Section 10) for in-sample diagnostics, not for forecasts.

## 3. The quantities

Notation: $K$ candidate models $M_k$, backtest windows $w = 1, \dots, W$ with forecast origin $t_1^{(w)}$ and test length $H$ (the `test_window`), horizon steps $j = 1, \dots, H$, and for a panel a series index $b$. A **cell** is one $(w, j)$ (and $b$). The per-window truth block is $y^{(w)} = (y_{t_1+1}, \dots, y_{t_1+H})$.

### 3.1 Per-draw log densities

For posterior draw $s$ of model $k$ fit on window $w$, with parameters $\theta^{(s)}$ and a sampled future latent path $z^{(s)}$ (the `_future` sites, drawn from the prior exactly as `forecast()` draws them), the model's forecast distribution has a density at the realized value, and the driver returns

$$
\ell^{(s)}_{k,w,j} = \log p\!\left(y_{t_1+j} \,\middle|\, z^{(s)}, \theta^{(s)}, y_{\le t_1}, M_k\right),
\qquad \text{shape } (S, *\text{batch}, H, \text{obs}).
$$

For a `predict()` model this is `trace["obs_future"]["fn"].log_prob(truth)` after conditioning `obs_future` on the truth block: for an elementwise observation family `prefix_condition` reduces to the horizon marginal given the path, so the value factorizes over cells and the array has one entry per cell, with no conditioning on earlier truth cells (the surgery slices, it does not condition). For a correlated family (`MultivariateNormal` over time) `obs_future` has the time axis in its event shape and `log_prob` returns one joint value per draw; the reader returns it as `(*batch, 1, 1)` and only the joint reduction below is meaningful. For an additive `ssoe()` model the same quantity is `noise_dist.log_prob(truth_j - mu_future_j)` along the sampled error path, which needs `mu_future` in the trace (Section 7.1).

### 3.2 Two reductions, two scores

The per-draw array supports two estimators, and they estimate different things.

**Horizon marginal** (the default). Averaging the density over draws cell by cell,

$$
\hat\ell_{k,w,j} = \log \frac{1}{S} \sum_{s=1}^S \exp \ell^{(s)}_{k,w,j}
\;\approx\; \log p\!\left(y_{t_1+j} \mid y_{\le t_1}, M_k\right),
$$

is the log density of the $j$-step-ahead forecast distribution at the realized value, with the intermediate steps $y_{t_1+1}, \dots, y_{t_1+j-1}$ *not* revealed. This is the distribution that `forecast()` samples and that CRPS, coverage and the pinball loss score, so a stacking weight learned from $\hat\ell$ at horizon $j$ weighs the same objects the user compares with the existing metrics. Implemented as `logsumexp(axis=0) - log(S)`.

**Joint block.** Summing over the block first,

$$
\hat L_{k,w} = \log \frac{1}{S} \sum_{s=1}^S \exp \sum_{j=1}^H \ell^{(s)}_{k,w,j}
\;\approx\; \log p\!\left(y^{(w)} \mid y_{\le t_1}, M_k\right)
= \sum_{j=1}^H \log p\!\left(y_{t_1+j} \mid y_{\le t_1+j-1}, M_k\right),
$$

is the joint density of the block, equal by the chain rule to the sum of one-step-ahead densities each conditional on the realized predecessors. This is the quantity `arviz_stats.loo.lfo_cv` computes and the one Bürkner et al. (2020) call the $M$-step-ahead elpd. It is an importance-sampling-like average: the draws whose sampled path happens to be close to the whole realized block dominate the sum, and at $H = 14$ the effective sample size of this estimator was 11% of the draws against 77% to 85% for the marginal (Appendix A.3). For `ssoe()` models the chain-rule form is instead exact and cheap: run the model on the concatenated series with `future = 0` and read the `obs` site's log density over the last $H$ steps, because the error-feedback recursion conditioned on the realized block is the in-sample filter extended by $H$ steps. That shortcut does not exist for `innovations` or `markov_series` models, whose in-sample latent sites change shape with the window.

The two scores coincide at $H = 1$ **only**: for any longer block the chain rule conditions on realized intermediate values that also update the posterior over the parameters, so they differ even for an iid model (the first draft of this document claimed otherwise; the reviewer corrected it). Results computed one way must never be compared with results computed the other way; the API keeps them under different names (`reduction="marginal"` and `reduction="joint"`), the default is the marginal, `backtest` records which one a result holds, and `stacking_cells` refuses to mix them.

### 3.3 The stacking objective and its baselines

With $L$ the matrix of per-cell log densities, shape `(cells, K)`, complete-pooling stacking solves

$$
\hat w = \arg\max_{w \in S_K} \sum_{c=1}^{N} \pi_c \log \sum_{k=1}^K w_k \exp L_{ck}
= \arg\max_{w \in S_K} \sum_c \pi_c \operatorname{logsumexp}_k \left(L_{ck} + \log w_k\right),
$$

with optional per-cell weights $\pi_c$ (a recency sequence, Yao et al. 2022 Section 2.5, or a tempering by the design effect of Section 3.4). The objective is concave in $w$ (the log of an affine function). The solver parameterizes $w = \text{softmax}(\eta)$ with $\eta_K = 0$, adds a $\text{Dirichlet}(\alpha)$ log prior that is flat at $\alpha = 1$ and keeps weights strictly positive for $\alpha > 1$, and runs BFGS from the uniform point; the shift by $\max_k L_{ck}$ per cell is built into `logsumexp`, so a cell with densities of order $e^{-700}$ (far-tail intermittent demand) is as stable as any other. The optimum is on the boundary when one model dominates everywhere, and the softmax parameterization reaches it only in the limit, so the solver reports the weights at convergence, not an exact zero. Duplicated models are not separately identified: the objective is flat along any split of their combined weight (the reviewer verified this to six decimals), so only the sum is a meaningful quantity, and that is what the test suite checks. Pseudo-BMA+ (Yao et al. 2018, Section 3.4) is the baseline: $w_k \propto \exp(\sum_c L_{ck})$, averaged over Dirichlet(1) Bayesian-bootstrap reweightings of the **windows**. Equal weights are the baseline every stacking result has to beat.

### 3.4 Dependent cells and the window as the unit

The cells of one window share one realized future path and one fitted posterior, so the per-cell stacking terms are serially dependent whatever the stride: on the prototype (stride equal to the test window, so no overlap at all) the lag-one autocorrelation of the per-cell difference between the two models was 0.62, and the variance of the window means was 9.97 times what independent cells would give, so a fourteen-step window carried about 1.4 independent cells (Appendix A.4). The point estimate of the weights is unaffected in expectation (an M-estimator with dependent terms stays consistent), but anything that reads the curvature of the objective is not: the posterior standard deviation of the complete-pooling weight computed as if cells were independent was 0.056 against a window-bootstrap standard deviation of 0.186, a factor 3.4, and the hierarchical model's partial pooling is set by the same curvature, so it under-shrinks the group offsets by the same factor.

Yao et al. (2022) never meet this because their time-series windows are one step long. The design makes the **window the unit of exchangeability** everywhere: `window_design_effect()` estimates the factor from the data, `stacking_factor()` and the solvers take `cell_weights` so the likelihood can be tempered by its inverse, `pseudo_bma_weights()` bootstraps windows and not cells, and `log_score(..., unit=window)` returns per-window means so that every held-out comparison carries a paired standard error across windows. A comparison quoted per cell understates its error by the square root of the design effect.

### 3.5 Hierarchical weights for forecasts

Each cell carries a discrete group index $g(c) \in \{-1, 0, \dots, J-1\}$ (the horizon step, or the series, or a user-supplied key such as the hierarchy level in the M5 example; $-1$ for a cell with no group, see below) and optional continuous features $f_m(c)$ (intermittency statistics of the training window, trend strength, calendar position). The unconstrained score of model $k < K$ at cell $c$ is

$$
\alpha_{k}(c) = \mu_k + \sigma_k \, z_{g(c),k} + \sum_m \beta_{mk} f_m(c),
\qquad \alpha_K(c) = 0,
$$

with $\mu_k \sim \text{Normal}(0, \tau_\mu)$, $\sigma_k \sim \text{HalfNormal}(\tau_\sigma)$, $z_{jk} \sim \text{Normal}(0, 1)$ (the non-centered form of eq. 9) and $\beta_{mk} \sim \text{Normal}(0, \tau_\beta)$. Several group indices (horizon and series) enter additively, each with its own $\sigma$. The weights are $w(c) = \text{softmax}(\alpha(c))$ and the log likelihood is $\sum_c \pi_c \operatorname{logsumexp}_k (L_{ck} + \log w_k(c))$, registered with `numpyro.factor`. The block exposes the weights as a deterministic site; the user reads its posterior mean, which is the estimator the paper recommends over the MAP.

The horizon is an **ordinal** group, which the exchangeable prior of eq. 9 ignores. The block offers a second group prior, a centered Gaussian random walk over the ordered groups, $z_{jk} = \sum_{i \le j} \epsilon_{ik}$ with $\epsilon \sim \text{Normal}(0, 1)$, so neighboring horizon steps share strength. In the prototype this prior produced the monotone weight profile the data generating process implies (0.86 at $j = 1$ down to 0.40 at $j = 14$) where the exchangeable prior shrank everything to a band (0.74 to 0.57) and the no-pooling solution was noisy, with a bootstrap standard deviation of 0.2 per step and a quarter of the bootstrap replicates on the boundary (Appendix A.2). Per-horizon weights are **not identified from 30 windows**; the hierarchical prior is what makes them reportable at all, and even then the held-out gain over complete pooling was nil on that toy (Appendix A.1). Horizon-dependent weights earn their keep only when the models' relative accuracy actually changes with the horizon, and the design makes the complete-pooling and equal-weights scores a one-line comparison so the user can check.

Prediction at new cells is the softmax of the same scores at those cells' groups and features. The block is called on the training **and** held-out cells together (the tutorial's `w_test`): the factor scores the training rows and the posterior of the deterministic holds the held-out weights. A new series with no backtest history carries group index $-1$ and gets the population weights $\text{softmax}(\mu)$, which is the partial-pooling payoff for cold-start series in a panel.

### 3.6 CRPS stacking from draws only

For a mixture $F_w = \sum_k w_k F_k$ the CRPS at $y$ has the closed form

$$
\text{CRPS}(F_w, y) = \sum_k w_k \, a_k - \tfrac{1}{2} \sum_{k,l} w_k w_l \, B_{kl},
\qquad a_k = \mathbb{E}|X_k - y|, \quad B_{kl} = \mathbb{E}|X_k - X_l|,
$$

which follows from the energy form $\text{CRPS}(F, y) = \mathbb{E}|X - y| - \tfrac12 \mathbb{E}|X - X'|$ with $X, X'$ independent draws from the mixture. Since $\text{CRPS}(F, y) = \int (F(x) - \mathbf{1}\{y \le x\})^2 \, dx$ is a convex quadratic in $F$ and $F_w$ is affine in $w$, the mixture CRPS is a convex quadratic in $w$ and the minimization over the simplex is a small convex QP that projected gradient solves reliably. The restriction of $-B$ to the simplex tangent space is positive semidefinite while the unrestricted $-B$ is indefinite (the reviewer found an eigenvalue of $-15.2$), so a naive definiteness check would wrongly fail; the test checks midpoint convexity on the simplex instead. The terms $a$ and $B$ are computed once from the backtest draws, with $a_k - \tfrac12 B_{kk}$ equal to `crps_empirical` of model $k$ alone, and the identity against `crps_empirical` on a sample pooled **in proportion to the weights** holds to $10^{-6}$ (a plain concatenation is only the $1/K$ mixture).

This solver exists for the models that have no density and for users who evaluate with CRPS. Yao et al. (2018) argue for the log score on propriety grounds and warn that stacking of means is not strictly proper and can be non-identified; the CRPS is a strictly proper score for the mixture, so the objection does not apply, but the two solvers optimize different things and a notebook should show both. In the prototype the CRPS weights leaned further toward the sharper model (0.86 on the random walk against 0.66 for the log score), which is the known behavior of the CRPS rewarding sharpness less than the log score penalizes tails.

### 3.7 Sampling the mixture

Given per-model forecast draws, shape `(K, S, *batch, future, obs)`, and weights, `stack_forecasts()` draws a model index per sample path from $\text{Categorical}(w)$ and takes that model's draw. With weights that vary across cells (horizon-dependent, series-dependent) the choice is per cell instead; the marginal at every cell is then correct but a sample path mixes models along the horizon, which breaks path statistics (cumulative sums, peak timing) and, for a panel, cross-series coherence within a path. The function takes a `per` argument (`"path"` or `"cell"`), defaults to `"path"` for a single weight vector, and documents the trade-off; a `Categorical` draw rather than a weighted average of quantiles is what makes the result a sample from the mixture distribution.

Linear pools are overdispersed: a mixture of calibrated components is wider than any component (Gneiting and Ranjan 2013, Theorem 3.1), so a stacked forecast of two well-calibrated models has coverage above nominal. Recalibration (a beta-transformed pool, or the PIT-based recalibration the electricity notebook introduces) is out of scope for the first iteration and noted in Section 10; the coverage metric in `DEFAULT_METRICS` makes the effect visible immediately.

## 4. Design principles

- **Densities come from the model's own sites.** The driver reads the `obs_future` density that `predict()` already builds through `prefix_condition`; it never reconstructs a likelihood from `prediction` and `obs_dist`, so link functions, Student-t tails, discrete families and the MVN layout check are covered by the existing surgery code and its tests.
- **Leave-future-out, never leave-one-out.** Weights are learned from `backtest()` cells. The design does not offer PSIS-LOO on the in-sample `obs` site as a shortcut, because that conditions each point on its own future and rewards models that interpolate rather than forecast.
- **The horizon marginal is the default score.** It is the distribution the rest of the package evaluates, it factorizes over cells so weights can depend on the horizon, and its Monte Carlo estimator is well behaved except at surprising cells. The joint block score stays available under its own name.
- **Closures never see the test block.** `backtest` computes the density itself from the posterior the closure returns in a `ForecastOutput`; the leak barrier of `forecast_fn` is kept, and the model is fit once per window.
- **The window is the unit of exchangeability.** Standard errors, bootstraps and likelihood tempering are per window (Section 3.4). Every reported gain carries a paired standard error across held-out windows, and equal weights are the baseline.
- **Plain functions, `rng_key` first, the user owns inference.** The hierarchical weights are a building block the user calls inside their own three-line NumPyro model and fits with their own `MCMC` or `SVI`, as they do with a forecasting model; the solvers are pure jitted kernels; nothing holds state. No stacker class, no shipped model function, no fit method.
- **Same layouts as the rest of the package.** Per-draw densities are `(sample, *batch, future, obs)` like forecast draws; the model axis is last in every stacking matrix (`(cells, K)`, as `lpd_point` in the tutorial) and in per-cell weights `(*batch, future, obs, K)`; host-committed and NumPy inputs are accepted wherever draws are.
- **Compose with the drivers, do not fork them.** `forecast_log_density` reuses the chunking, device placement and compile-once machinery of `forecast`.
- **Every claim about a score is a test identity.** Known-parameter Normal, Poisson and AR(1) densities; the two reductions against their NumPy definitions; the chain-rule route against the in-sample filter; one-model stacking returning weight one; the combined weight of duplicated models equal to the merged model's; complete pooling recovered at zero spread; mixture CRPS equal to the CRPS of a proportionally pooled sample; the solver against an independent SciPy reference and against ArviZ. Snapshot tests of fitted weights are not accepted.

## 5. Approaches considered

### A. Density driver, pure solvers, and weight blocks (recommended)

The three layers of Section 1. Each is independently useful: the density driver alone gives a log score column in backtests (a metric the package lacks), the solvers alone give complete-pooling stacking and pseudo-BMA+ in a few lines, and the blocks give hierarchical weights to users who want them inside a model they write. The user keeps control of inference and of which backtest windows feed the weights.

### B. Export `log_likelihood` and defer to ArviZ

Add a `log_likelihood` group to `to_datatree()` and let users call `arviz_stats.compare(method="stacking")`. Rejected as the stacking route: the group would hold the in-sample `obs` densities, so `loo()` would do PSIS-LOO on a time series (future leakage), `lfo_cv` needs a refitting wrapper the package would have to provide anyway, and ArviZ offers no hierarchical stacking. The export is still worth doing for in-sample diagnostics and is listed in Section 10.

### C. A `Stacker` object

A class holding the candidate models, running the backtests, fitting the weights and forecasting. Rejected: the package removed its class hierarchy deliberately (PR #81), inference is the user's, and the object would have to re-implement window iteration and forecast caching that `backtest()` already has.

### D. A shipped hierarchical stacking model function

The first draft of this design proposed `hierarchical_stacking(lpd, groups, features, ...)` as a complete NumPyro model function in the package. The reviewer objected that it has no precedent here (`AGENTS.md` ships blocks called inside user models and keeps full models in tests and notebooks) and would grow a configuration surface (priors, tempering, recency weights, feature handling) that a fifteen-line user model does not need. Replaced by the two blocks of approach A, which is the library's own idiom: the user composes `hierarchical_weights()` and `stacking_factor()` with whatever else they want in the model, and the package tests the blocks' identities.

### E. Fit a mixture model jointly

Write the mixture of the $K$ models as one NumPyro model with a latent assignment and fit it once. Rejected for the reasons in Yao et al. (2022, Section 4): the cost is $K$ times a single fit at every evaluation, the component models are typically already fit and backtested, and a jointly fit mixture does not use held-out predictive performance at all.

### F. Per-horizon no-pooling stacking only

Run complete-pooling stacking separately at each horizon step with the pure solver and skip the hierarchical blocks. Kept as the baseline the blocks are compared against (one `vmap` of the solver over the horizon axis), but not as the only option: with 30 windows per cell the no-pooling weights had a bootstrap standard deviation of 0.2 and sat on the boundary in a quarter of the replicates (Appendix A.2).

## 6. The API

Full signatures, docstrings and types are on the scaffold branch and in `docs/dev/stacking_scaffolding.md`; this section states the contracts.

| Layer | Where | Public names |
| --- | --- | --- |
| Density driver | `predictive.py` | `forecast_log_density`, `reduce_log_density`, `log_density_ess`, `conditional_log_density`, `predict_density`, `ssoe_density` |
| Backtest hook | `evaluate.py` | `backtest(log_density=, density_reader=)`, `BacktestResult.log_density`, `BacktestResult.log_density_reduction`, `backtest_vectorized(log_density=, density_reader=)`, `VectorizedBacktestResult.log_density` |
| Cells | `stacking.py` | `StackingCells`, `stacking_cells`, `window_design_effect` |
| Solvers | `stacking.py` | `stacking_weights`, `pseudo_bma_weights`, `log_score`, `mixture_crps_terms`, `crps_stacking_weights` |
| Blocks | `stacking.py` | `hierarchical_weights`, `stacking_factor`, `rectified_features` |
| Mixture | `stacking.py` | `stack_forecasts`, `MixturePer` |
| Types and errors | `typing.py`, `exceptions.py` | `ForecastOutput`, `DensityReader`, `Reduction`, `GroupPrior`, `DensitySiteError`, `StackingShapeError` |

### 6.1 `forecast_log_density`

`forecast_log_density(rng_key, model, posterior, data, covariates, truth, *, reader=predict_density, batch_size=None, parallel=True, device=None)` returns the per-draw array of Section 3.1. The call signature is `forecast()` plus `truth` (shape `(*batch, future, obs)`) and a `reader`. One traced pass per posterior draw: `substitute` the draw, `seed` the pass so every `_future` site and every reparameterization auxiliary is drawn from the prior (verified for `LocScaleReparam`, `time_reparam`, `markov_series`, an enclosing series plate, NumPy leaves and a Poisson link, Appendix A.5), condition the observation site on `truth`, and hand the trace to the reader. The pass is vmapped over the sample axis in chunks of `batch_size` exactly as `_predict` is, with the same device placement contract. It never returns forecast draws; `forecast()` does that, with its own key.

Two guards the review added. A `posterior` that carries `_future` keys (the output of a `Predictive` call over a horizon) would pin the future path by name and return the same densities for every key with no error; the driver rejects such keys with `ValueError` listing them. And the reader branches on the site's `event_shape` **before** calling `log_prob`, because a `MultivariateNormal` with event `(future,)` evaluated at a `(future, 1)` truth broadcasts each scalar against the event and returns `(future,)` finite garbage; elementwise families return per cell, the time-leading MVN layout returns the joint as `(*batch, 1, 1)`, anything else raises.

A `DensityReader` is `(trace, truth) -> Array`. `predict_density` reads `trace["obs_future"]["fn"].log_prob(truth)` and raises `DensitySiteError` naming the fix when the site is missing; its `site=` keyword reads a predictive the model registered under another name. `ssoe_density(name)` returns a reader for **additive** `ssoe()` models (Section 7.1).

### 6.2 The backtest hook

`backtest(..., log_density: Reduction | None = None, density_reader=predict_density)`. When `log_density` is set, `forecast_fn` must return a `ForecastOutput(draws, posterior)`; `backtest` reads the draws as before, scores the window's truth block with `forecast_log_density` on that posterior (its own key substream, `fold_in(rng_key, 3)`), reduces it, and stores the per-cell result on `BacktestResult.log_density` with the reduction name on `BacktestResult.log_density_reduction`. The closure never sees the test block and the model is fit once per window; a closure that returns bare draws when a density is requested gets a `ValueError` naming the fix. The density is computed on the untransformed truth: `transform` applies to the sample metrics, not to the density, and the docstring says the two then sit on different scales. Nothing changes for callers that pass neither. `backtest_vectorized(..., log_density=..., density_reader=...)` owns every window's posterior already and computes the array inside its fused pass.

`stacking_cells(results_by_model, data)` turns `{name: list[BacktestResult]}` from $K$ backtests with identical windows into a `StackingCells` frozen dataclass of host arrays: `log_density` `(cells, K)`, `horizon`, `window`, `series`, `truth` (all `(cells,)`), the reduction, and `draws` `(K, S, cells)` when every backtest kept predictions. It validates that every model saw the same `(t0, t1, t2)` sequence and the same reduction and raises `StackingShapeError` otherwise. This is the only function that knows the backtest result layout; the solvers see matrices.

### 6.3 Solvers

`stacking_weights(log_density, *, cell_weights=None, dirichlet_alpha=1.0, tol, max_iter)` and `pseudo_bma_weights(rng_key, log_density, *, unit=None, num_bootstrap=1_000)` return `(K,)` on the simplex; `mixture_crps_terms(draws, truth)` returns `a` `(cells, K)` and `B` `(cells, K, K)`, and `crps_stacking_weights(a, b, *, cell_weights=None, tol, max_iter)` returns `(K,)`; `log_score(log_density, weights, *, unit=None)` evaluates the per-cell score for any weights broadcastable to `(cells, K)`, or the per-unit means when `unit` is given; `window_design_effect(log_density, window)` returns the factor of Section 3.4. All are pure JAX, jitted with static shapes and `vmap`-able over a leading axis (per-horizon no pooling is `jax.vmap(stacking_weights)` on `(H, cells_per_h, K)`).

### 6.4 The blocks

`hierarchical_weights(name, num_models, groups, num_groups, *, features=None, group_prior="exchangeable" | "random_walk", tau_mu, tau_sigma, tau_beta) -> log_weights (cells, K)` samples `{name}_mu`, `{name}_sigma_i` and `{name}_z_i` per group index $i$, `{name}_beta`, and registers the deterministic `name` holding the weights. `num_groups` is a Python integer because it sizes a plate (a traced `groups.max()` cannot). `stacking_factor(name, log_density, log_weights, *, cell_weights=None)` registers $\sum_c \pi_c \operatorname{logsumexp}_k (L_{ck} + \log w_k(c))$ as a `numpyro.factor` and returns the value. A user's stacking model is

```python
def stacking_model(lpd_train, groups, num_groups, cell_weights):
    log_w = hierarchical_weights("w", lpd_train.shape[-1], groups, num_groups)
    stacking_factor("stacking", lpd_train, log_w[: lpd_train.shape[0]], cell_weights)
```

fit with `MCMC(NUTS(stacking_model))`, the posterior mean of `w` read off `mcmc.get_samples()`. The blocks' test identities: zero spread is complete pooling (all rows of the weights equal); group $-1$ gets $\text{softmax}(\mu)$; the random-walk offsets average to zero; the factor's value is the objective at the given weights; a NUTS run on simulated cells with group-varying truth recovers the weights within 0.1 (a slow test).

### 6.5 Mixture draws

`stack_forecasts(rng_key, draws, weights, *, per="auto")` takes `(K, S, *batch, future, obs)` draws and weights `(K,)` or `(*batch, future, obs, K)` and returns `(S, *batch, future, obs)` draws from the mixture (Section 3.7). `StackingShapeError` for negative, unnormalized or non-broadcastable weights, and for `per="path"` with cell-varying weights.

## 7. Pitfalls and how the design handles them

### 7.1 Which models have a forecast density

`predict()` models do, through `obs_future`. **Additive** `ssoe()` models (ARMA, exponential smoothing, a local level) do once they register `numpyro.deterministic(f"{name}_mu_future", r.mu_future)` when forecasting: the forecast distribution at step $j$ along a sampled error path is `noise_dist` shifted by `mu_future[j]`, and the reviewer verified that `ssoe_density` reproduces the AR(1) closed form $\text{Normal}(\phi^h y_t, \sigma^2 (1 - \phi^{2h}) / (1 - \phi^2))$ at $S = 20{,}000$. The deterministic is one line in the model; an opt-in `register_mean` flag on `ssoe()` is the alternative, deferred until a second notebook needs it.

The **intermittent-demand models** (Croston, TSB, availability TSB) have no scorable forecast at all as written, and the first draft of this design proposed them as the showcase. Their `forecast` is `z.y_future * p.y_future`, the product of a size channel and a probability channel that are each a level plus Normal noise (the probability is not confined to $[0, 1]$): it is a noisy rate forecast, not a draw of intermittent demand, so CRPS stacking on it scores a rate distribution against zero-or-positive demand, and the `mu_future` recipe gives a channel's density, not the demand density, whose zero-inflated predictive $(1 - p)\,\mathbf{1}[y = 0] + p \, \text{Normal}(y; z, \sigma)$ exists nowhere in the model. The per-model obligation is therefore: a model is stackable when it registers, over the horizon, an observed-style site holding the predictive the user wants scored (`predict_density(site=...)` reads it), or when it is a `predict()` model, or an additive `ssoe()` model with the deterministic. The intermittent-demand notebooks need a demand predictive before they can be stacked, which is a modeling change outside this design.

### 7.2 Two h-step scores

Section 3.2. The reduction is an explicit argument, recorded on the result by `backtest` itself, and `stacking_cells` refuses to stack results with different reductions. The docstrings state which one each quantity is, that the ArviZ `lfo_cv` is the joint one, and that they coincide at horizon one only.

### 7.3 Correlated observation families and discrete support

Section 6.1. The event-shape branch runs before `log_prob`; the time-leading MVN returns the joint block as `(*batch, 1, 1)`, `stacking_cells` records `horizon = -1` for such cells, and `hierarchical_weights` rejects a group index containing $-1$ when it is meant as the horizon (a group of $-1$ means "no group"). Discrete families work unchanged; `truth` must have the integer dtype `predict()` already enforces for `data`, and the driver raises the same `ValueError`. Side finding of the review, a pre-existing bug independent of this design: `forecast()` on the same MVN model returns `(sample, future)` with no obs axis and trips its own jaxtyping return annotation; to be filed as an issue.

### 7.4 Dependent cells

Section 3.4. Intrinsic to multi-step windows, not an overlap artifact; handled by making the window the unit everywhere and by `window_design_effect()`.

### 7.5 Identifiability of per-horizon weights

Section 3.5. With $K$ models and $H$ horizon steps the no-pooling solution has $H (K - 1)$ free parameters fit from $W$ windows per step, and $W = 30$ is not enough; the hierarchical prior is the fix, and the notebook must show the complete-pooling and equal-weights scores next to the hierarchical one.

### 7.6 Monte Carlo error of the densities

The marginal estimator is a plain Monte Carlo average and its bootstrap standard error was 0.023 per cell with 400 draws at every horizon on a typical window (Appendix A.3). It is not uniformly benign: at a cell whose realized value is far in the tail of most sampled paths the effective sample size collapsed to 8 of 400 draws with a standard error of 0.35 nats, and on a window whose fitted drift scale was large the ESS at $j = 14$ was 72. The log-mean-exp estimator is biased downward by about half its squared standard error, and the bias falls on the model with the wider latent path. `forecast_log_density` returns the per-draw array, `log_density_ess()` reports the effective sample size per cell, the docstrings say that the number of draws must grow with the horizon, and the test suite pins the estimator against closed forms. The joint reduction is worse by construction and its docstring points at the exact `ssoe()` route.

### 7.7 Path coherence of the mixture

Section 3.7. `per="path"` for a single weight vector; the docstring explains what `per="cell"` breaks.

### 7.8 Overdispersion of linear pools

Section 3.7. Not handled in the first iteration; documented, and the coverage metric exposes it.

### 7.9 Host offload and NumPy inputs

Every new function that consumes draws or densities accepts `Array | np.ndarray` and host-committed arrays, following the `draw_posterior` contract, and `forecast_log_density` takes the same `device` argument as `forecast`. The solvers move inputs to device memory with `_device_view` as the metrics do.

### 7.10 Weights are not model probabilities

The docstrings of the solvers state, as Yao et al. (2018) do, that stacking weights optimize the predictive score of the mixture and are not posterior model probabilities, and that a model with weight zero is not thereby wrong (Theorem 2 of Yao et al. 2022 bounds its local winning probability but does not make it zero).

### 7.11 Posteriors that carry future sites

Section 6.1. Rejected with `ValueError`; `forecast()` has the same hole and should get the same guard (Section 10).

## 8. Adversarial review

An independent reviewer with no access to the author's reasoning was given the first draft of the proposal, the prototype and its cached backtest, the library clone and the papers, and asked to attack the statistics, the mechanics, the API fit, the test design and the scope, with code. Nine findings came back, ranked; every one is resolved in the design above. The scripts are summarized in Appendix A.4 and A.5.

| # | Finding (severity) | Resolution |
| --- | --- | --- |
| 1 | The prototype's held-out gain of 0.077 nats over the best single model has a paired standard error of 0.075 across the 20 held-out windows ($t \approx 1$); fixed equal weights score at least as well as every stacking variant; the toy is complementary by construction, and the decaying per-horizon profile is mostly the random-walk prior smoothing an unidentified no-pooling estimate (blocker for the showcase). | Appendix A.1 now carries the standard errors and the equal-weights row. Equal weights are the baseline (Section 3.3), every comparison is per window (`log_score(unit=)`, Section 3.4), and the showcase changed (Section 9). The prototype numbers are quoted only with their error. |
| 2 | Weight-posterior overconfidence is intrinsic to multi-step windows, not an overlap artifact: design effect 9.97, lag-one autocorrelation 0.62, posterior sd 0.056 against bootstrap sd 0.186 (major). | Section 3.4: the window is the unit of exchangeability; `window_design_effect()`, `cell_weights` tempering on the solvers and `stacking_factor()`, `pseudo_bma_weights(unit=)`, `log_score(unit=)`. |
| 3 | The intermittent-demand trio cannot be scored by either route: their `forecast` is a product of two noisy channels, not a demand draw, and no demand predictive exists in the trace (major). | Section 7.1 states the per-model obligation; `ssoe_density` is documented as correct for additive single-channel models only; `predict_density(site=)` reads a user-registered predictive; the showcase moved to models with well-defined densities (Section 9). |
| 4 | An MVN `obs_future` with event `(future,)` evaluated at `(future, 1)` truth broadcasts into finite garbage with no error; side finding: `forecast()` on that model violates its own return annotation (major). | Section 6.1: the reader branches on `event_shape` before `log_prob`, returns the joint as `(*batch, 1, 1)` for the time-leading layout and raises otherwise; a test builds the MVN case against the conditional closed form. The `forecast()` bug is listed for an issue (Section 10). |
| 5 | The marginal estimator is correct against closed forms, but its ESS collapses at surprising cells (8 of 400) and the log-mean-exp bias is model dependent (major). | Section 7.6: per-draw array returned, `log_density_ess()`, docstring guidance on $S$ growing with the horizon, closed-form tests for Normal, Poisson, random walk and AR(1). |
| 6 | A `log_density_fn` closure receiving the truth breaks `backtest`'s leak barrier and either refits or changes the return type; densities belong on `backtest_vectorized`, which owns the posterior (major, API). | Section 6.2: `ForecastOutput(draws, posterior)`; `backtest` scores the posterior itself and never hands the closure the test block; `backtest_vectorized(log_density=)`; `transform` documented as not applying to the density. |
| 7 | A posterior dict carrying `_future` keys silently pins the path (minor). | Section 6.1: `ValueError` listing the keys; the same guard proposed for `forecast()` (Section 10). |
| 8 | Two wrong statements: the two h-step scores coincide only at $H = 1$ (not whenever the likelihood ignores lagged responses), and "duplicated models split evenly" is not an identity (the objective is flat along the split). | Section 3.2 and all docstrings corrected; the test checks the invariance of the combined weight and the objective, and the even split only under $\alpha > 1$ (Section 3.3). |
| 9 | A shipped hierarchical stacking model function has no precedent in the package; keep it in the notebook until a second use (minor, fit). | Approach D rejected; replaced by the two blocks of Section 6.4, which follow the library's own block idiom and are what the tests pin. |

Claims that survived the review, verified by its scripts: the `substitute` + `seed` + `condition` + `trace` pattern pins exactly the in-sample sites and draws the future auxiliaries; for elementwise families the per-cell value is the density along the path with no conditioning on earlier truth cells; the pattern works under `vmap`, `jit`, nested `vmap` over windows, `time_reparam("dct")`, `markov_series`, an enclosing series plate, NumPy leaves and a Poisson link; the marginal estimator matches the closed-form h-step density of a random walk within two standard errors at $S = 400$ and $S = 20{,}000$; the `ssoe` recipe matches the AR(1) closed form; the mixture CRPS identity holds against a proportionally pooled sample for three models and three weight vectors; the CRPS objective is a convex QP on the simplex; and `arviz_stats.compare(method="stacking")` on hand-built `ELPDData` returns the prototype's complete-pooling weight (0.66 against 0.656).

## 9. Phased plan

**PR 1, densities.** `forecast_log_density` with `predict_density` and `ssoe_density`, `reduce_log_density`, `log_density_ess`, `conditional_log_density`, `ForecastOutput`, `backtest(log_density=, density_reader=)` and the result fields, `backtest_vectorized(log_density=, density_reader=)`, the two errors, and the `_future`-key guard. Tests: `tests/test_density.py` of the scaffold (closed forms, the MVN joint case, the discrete dtype error, the guard, chunking and placement, the two reductions, the chain-rule route, the backtest fields). The `forecasting_univariate` notebook gains a log score column in its backtest table and a `ForecastOutput`-returning closure.

**PR 2, cells and solvers.** `StackingCells`, `stacking_cells`, `window_design_effect`, `stacking_weights`, `pseudo_bma_weights`, `log_score`, `stack_forecasts`. Tests: the solver against the SciPy reference and ArviZ, the duplicate invariance, concavity, the shift invariance, the design effect on independent and shared-term cells, the mixture sampler's one-hot and whole-path properties. Notebook `model_stacking.ipynb`, part one: a random-walk level, an ARMA (`ssoe` with the deterministic) and a seasonal mean on one of the existing example series (weekly BART ridership, or the Victoria electricity demand), non-overlapping windows, weights fit on the first half of the windows and every method scored on the second half **per window with a paired standard error**, equal weights and the best single model as baselines, pseudo-BMA+ next to stacking.

**PR 3, hierarchical blocks and CRPS.** `hierarchical_weights`, `stacking_factor`, `rectified_features`, `mixture_crps_terms`, `crps_stacking_weights`. Tests: the block identities, the NUTS recovery test (slow), the CRPS identities and convexity. Notebook part two: horizon-group weights with the random-walk prior and the design-effect tempering, compared with complete pooling and no pooling on the held-out windows; per-series weights on the fresh-retail panel with series-level paired standard errors and the cold-start group; the CRPS weights next to the log-score weights.

Each PR updates `reference:` in `great-docs.yml` (already done on the scaffold branch), the `skill:` block (a new decision-table row: "combine several models" leads to `backtest(log_density=)`, `stacking_cells` and `stacking_weights`), and `AGENTS.md` (the two-score rule, the window-as-unit rule, the `_mu_future` convention, and the `ForecastOutput` return of `forecast_fn`).

## 10. Open questions and follow-ups

- Whether `ssoe()` should register `{name}_mu_future` itself behind an opt-in flag. Leaning to the one-line deterministic in the model until a second notebook needs the flag.
- A `log_likelihood` group on `to_datatree()` for in-sample ArviZ diagnostics. Independent of stacking; cheap; a separate PR.
- Two issues to file from the review: `forecast()` returns `(sample, future)` for the time-leading MVN layout and violates its own return annotation; `forecast()` and `predict_in_sample()` accept a posterior carrying `_future` keys and silently pin the path.
- Recalibration of the stacked forecast (beta-transformed pool, or the PIT recalibration of the electricity notebook) once the overdispersion is measured on a real example.
- A demand predictive for the intermittent-demand models, so they become stackable (Section 7.1); a modeling change with its own design.
- Whether the posterior draws of the weights (not only their mean) should feed `stack_forecasts`, propagating weight uncertainty into the mixture. The paper reports the mean; propagating draws is a one-line change the user can make on the block's deterministic.
- Stacking across inference methods (NUTS, SVI, Pathfinder of the same model, as in the inference comparison notebook) is a degenerate case of this design with $K$ posteriors of one model; worth a paragraph in the notebook, not an API.

## Appendix A. Prototype and review evidence

The prototype (`stacking_poc.py`, kept outside the repository) backtested two models built from the package's blocks on a simulated series $y_t = 10 + 1.5 \sin(2\pi t / 7) + u_t$, $u_t = 0.9\, u_{t-1} + \text{Normal}(0, 0.5)$, $T = 840$: a random-walk level with Fourier seasonality (`innovations` + `predict`, `LocScaleReparam`) and a constant mean with Fourier seasonality. Rolling windows of 140 training days, 14 test days, stride 14, 50 windows; SVI with `AutoNormal`, 3,000 Adam steps, 400 posterior draws per window. Weights were fit on the first 30 windows and every method was scored on the last 20. The toy is complementary by construction: an AR(1) with $\phi = 0.9$ is a random walk at short lags and mean-reverting at long lags, and the random walk wins at horizons 1 to 9 and the constant mean at 10 to 14.

### A.1 Held-out mean log density per cell, with the standard error the first draft omitted

Unit: the 20 held-out windows. Differences are against the random walk alone; the paired standard error is across windows.

| Method | Score | Gain over random walk | Paired SE | $t$ |
| --- | --- | --- | --- | --- |
| Random walk alone | $-1.881$ | | | |
| Constant mean alone | $-1.931$ | $-0.050$ | | |
| Complete-pooling stacking | $-1.804$ | $+0.077$ | 0.075 | 1.03 |
| No pooling (one weight vector per horizon step) | $-1.811$ | $+0.071$ | 0.071 | 1.00 |
| Hierarchical, exchangeable groups over horizon | $-1.802$ | $+0.079$ | 0.076 | 1.05 |
| Hierarchical, random-walk prior over horizon | $-1.803$ | $+0.078$ | 0.075 | 1.04 |
| Equal weights (no fitting) | $-1.801$ | $+0.080$ | 0.101 | 0.79 |

Complete-pooling weight on the random walk: 0.66. Swapping the split (fit on the last 30, test on the first 20) gives a gain of $+0.038$ with standard error 0.069. No variant is distinguishable from equal weights on this toy.

### A.2 Weight on the random walk by horizon step

| $j$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hierarchical, exchangeable | 0.74 | 0.73 | 0.72 | 0.72 | 0.71 | 0.69 | 0.70 | 0.66 | 0.65 | 0.62 | 0.58 | 0.54 | 0.56 | 0.57 |
| Hierarchical, random walk | 0.86 | 0.85 | 0.84 | 0.82 | 0.80 | 0.76 | 0.72 | 0.67 | 0.61 | 0.54 | 0.48 | 0.43 | 0.41 | 0.40 |
| No pooling | 1.00 | 1.00 | 0.91 | 0.91 | 0.88 | 0.77 | 0.80 | 0.67 | 0.61 | 0.52 | 0.44 | 0.33 | 0.33 | 0.36 |

The window-bootstrap standard deviation of the no-pooling weight is about 0.2 at every step, and 24% of the bootstrap replicates sit on the boundary (below 0.02 or above 0.98): per-horizon weights are not identified from 30 windows, and the smooth profile of the random-walk prior is the prior doing most of the work.

### A.3 Monte Carlo error of the two reductions

One typical window, the random-walk model, 4,000 posterior draws. Bootstrap standard error of the marginal estimator with 400 draws: 0.023 to 0.026 at every horizon step; with 4,000 draws: 0.006 to 0.009. Effective sample size of the marginal estimator as a fraction of the draws: 0.85 at $j = 1$ to 0.77 at $j = 14$. Effective sample size of the joint block estimator over the 14 steps: 0.11.

The reviewer's known-parameter check (`mc_error.py`): against the closed form $\text{Normal}(\text{level}_t, \sqrt{\sigma^2 + h \, \text{drift\_scale}^2})$ every cell's error was within two standard errors at $S = 400$ and $S = 20{,}000$, so the estimator is right; at the one cell where the realized value was surprising (closed-form log density $-6.42$) the ESS was 8 of 400 and the standard error 0.35 nats. On the fitted model, the window with the largest fitted drift scale (1.55) had ESS 72 at $j = 14$ against about 350 on benign windows. The average absolute error over all cells was 0.03 nats, small against the (noisy) gain of A.1.

### A.4 Dependence of the cells (`holdout.py`)

Lag-one autocorrelation of the per-cell difference between the two models within windows, averaged over windows: 0.62. Design effect, the variance of the window means over the variance an independent sample of 14 cells would give: 9.97. Posterior standard deviation of the complete-pooling weight on the random walk treating cells as independent: 0.056; window-bootstrap standard deviation: 0.186.

### A.5 Mechanics verified (`mechanics.py`, `misc_checks.py`, `ssoe_density.py`, `crps_check.py`)

- `substitute(model, draw)` pins exactly the in-sample sites (`bias`, `weight`, `drift_scale`, `sigma`, `drift_decentered`, `drift`); `drift_future_decentered` and `drift_future` are drawn from the prior and differ across keys; `obs_future` is observed at the truth.
- The pattern runs under `vmap`, `jit`, nested `vmap` over windows, `time_reparam("dct")`, `markov_series`, an enclosing series plate (result shape `(50, 3, 14, 1)`), NumPy leaves and a Poisson link.
- A posterior dict that also holds `drift_future` and `drift_future_decentered` gives identical densities under two different keys (maximum difference 0.000 against 15.5 for a clean dict), with no error.
- The MVN `predict()` model: `obs_future` has batch `()` and event `(14,)`; `log_prob` of a `(14, 1)` truth returns shape `(14,)` by broadcasting; `forecast()` returns `(sample, 14)` and raises `BeartypeCallHintReturnViolation` on its own return annotation.
- The `ssoe` recipe reproduces the AR(1) closed form at $S = 20{,}000$.
- The mixture CRPS identity holds to $10^{-6}$ for $K = 3$ and $w \in \{(0.5, 0.25, 0.25), (0.6, 0.3, 0.1), (0.1, 0.1, 0.8)\}$ against a pool replicated in proportion to the weights; a plain concatenation differs by 0.027. $-B$ projected on the simplex tangent space has non-negative eigenvalues on every cell checked and on random $K = 5$ data; the unprojected $-B$ has an eigenvalue of $-15.2$.
- `arviz_stats.compare(method="stacking")` on hand-built `ELPDData` from the cached cells gives weight 0.66 on the random walk, matching the prototype's 0.656.
- A first version of the prototype substituted the posterior into `Predictive` and then into `log_likelihood`; it double-counted a key split and gave wrong densities for the reparameterized model. The single traced pass is the correct mechanism and the one the design specifies; the test suite pins it with the known-parameter identities.

## Appendix B. References

- Yuling Yao, Aki Vehtari, Daniel Simpson, Andrew Gelman (2018). Using stacking to average Bayesian predictive distributions (with discussion). *Bayesian Analysis* 13(3). [arXiv:1704.02030](https://arxiv.org/abs/1704.02030).
- Yuling Yao, Gregor Pirš, Aki Vehtari, Andrew Gelman (2022). Bayesian hierarchical stacking: Some models are (somewhere) useful. *Bayesian Analysis* 17(4). [arXiv:2101.08954](https://arxiv.org/abs/2101.08954).
- Paul-Christian Bürkner, Jonah Gabry, Aki Vehtari (2020). Approximate leave-future-out cross-validation for Bayesian time series models. *Journal of Statistical Computation and Simulation* 90(14). [arXiv:1902.06281](https://arxiv.org/abs/1902.06281).
- Tilmann Gneiting, Roopesh Ranjan (2013). Combining predictive distributions. *Electronic Journal of Statistics* 7. [arXiv:1106.1638](https://arxiv.org/abs/1106.1638).
- Jonathan Berrisch, Florian Ziel (2023). CRPS learning. *Journal of Econometrics* 237(2). [arXiv:2102.00968](https://arxiv.org/abs/2102.00968).
- Marco Del Negro, Raiden B. Hasegawa, Frank Schorfheide (2016). Dynamic prediction pools. *Journal of Econometrics* 192(2). [NBER w20575](https://www.nber.org/papers/w20575).
- NumPyro tutorial, Bayesian Hierarchical Stacking: Well Switching Case Study. [num.pyro.ai](https://num.pyro.ai/en/stable/tutorials/bayesian_hierarchical_stacking.html).
