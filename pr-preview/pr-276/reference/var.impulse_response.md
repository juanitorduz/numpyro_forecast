## var.impulse_response()


Compute the impulse responses (moving-average coefficients) of a VAR.


Usage

``` python
var.impulse_response(
    phi,
    horizon,
    *,
    scale_tril=None,
    cumulative=False,
)
```


Inverting the lag polynomial of y_t = c + \sum\_{l=1}^{p} \Phi_l y\_{t-l} + \varepsilon_t gives the moving-average representation y_t = \mu + \sum\_{h \ge 0} \Psi_h \varepsilon\_{t-h} whose coefficients follow the recursion

 \Psi_0 = I, \qquad \Psi_h = \sum\_{j=1}^{\min(h, p)} \Phi_j \Psi\_{h-j} \quad (h \ge 1). 

`out[..., h, i, j]` is the response of variable `i`, `h` steps after a unit shock to variable `j` at step `0` (a unit increase of the reduced-form residual \varepsilon\_{t, j} with the other residuals held at zero). With `scale_tril` the responses are to *orthogonalized* shocks: for a factor B with \Sigma = B B^\top (the Cholesky factor L of the shock covariance, so \varepsilon_t = L u_t with u_t standard normal) the function returns \Theta_h = \Psi_h B, the response to a one-standard- deviation shock u\_{t, j}. With the Cholesky factor this is the recursive identification: the first series in the ordering responds to its own shock only at h = 0, the second to the first two, and so on, so the ordering of the series is part of the model. The recursion is defined for any `phi`; the moving-average representation, and \Psi_h \to 0, need a stable VAR (see [companion_matrix()](var.companion_matrix.md#numpyro_forecast.var.companion_matrix)). The left recursion above and the right recursion \Psi_h = \sum_j \Psi\_{h-j} \Phi_j agree, because both are the two-sided inverse of the lag polynomial.


## Parameters


`phi: Float[Array, ``" *#batch lags obs obs"]`  
Coefficient tensor `(*batch, lags, obs, obs)`. Posterior draws pass through the leading batch axis unchanged; no `vmap` is needed.

`horizon: int`  
Largest step H \ge 0 to compute; the result holds `horizon + 1` steps including step `0` (like [acf()](acf.acf.md#numpyro_forecast.acf.acf), which includes lag `0`).

`scale_tril: Float[Array, ``" *#batch obs obs"] | None`` = None`  
Optional square factor `(*batch, obs, obs)` of the shock covariance (typically the Cholesky factor, `sigma[..., :, None] * l_omega` for an LKJ correlation factor). Its batch axes broadcast against those of `phi`.

`cumulative: bool = ``False`  
If `True`, return the running sum over steps, the response of the *level* when the modeled series are differences.


## Returns


`Float[Array, ``"*batch steps obs obs"]`  
\Psi_0, \ldots, \Psi_H (or \Theta_h, or their running sums) with the step axis at `-3`: the package's "time at `-2`" convention counts from the trailing event dims, and the event here is an `(obs, obs)` matrix. Reshape to `(*batch, steps, obs * obs)` to plot the grid with [predictions_to_datatree()](convert.predictions_to_datatree.md#numpyro_forecast.convert.predictions_to_datatree).


## Raises


`ValueError`  
If `horizon` is negative.


## Notes

The posterior mean of the impulse responses is not the impulse response at the posterior mean of `phi`: the recursion is nonlinear in `phi`, and a single explosive draw dominates the mean at long horizons, so summarize draws with quantiles and check stability first.


## References

Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*, chapter 2. Springer.

Orduz, J. *Bayesian VAR in NumPyro*, [juanitorduz.github.io/var_numpyro](https://juanitorduz.github.io/var_numpyro/), whose `compute_irf` this function generalizes.

Pinder, T. *Impulso*, [github.com/thomaspinder/impulso](https://github.com/thomaspinder/impulso): the broadcast-over-draws recursion of its `compute_ma_phi` is the shape of this implementation, ported to a JAX scan.
