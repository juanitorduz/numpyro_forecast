## var.companion_matrix()


Stack the VAR coefficients into the companion form of a VAR(1).


Usage

``` python
var.companion_matrix(phi)
```


 F = \begin{bmatrix} \Phi_1 & \Phi_2 & \cdots & \Phi\_{p-1} & \Phi_p \\ I & 0 & \cdots & 0 & 0 \\ 0 & I & \cdots & 0 & 0 \\ \vdots & & \ddots & & \vdots \\ 0 & 0 & \cdots & I & 0 \end{bmatrix}, 

a square matrix of size `lags * obs`. The companion state stacks the lag window **most recent first**, the reverse of the `lags` layout: `s = lags[..., ::-1, :].reshape(*batch, lags * obs)`, and then `(F @ s)[..., :obs]` equals [var_mean()](var.var_mean.md#numpyro_forecast.var.var_mean) without intercept. The VAR is stable exactly when every eigenvalue of F has modulus below one; that is the condition for a finite unconditional mean (I - \sum_l \Phi_l)^{-1} c and for the impulse responses of [impulse_response()](var.impulse_response.md#numpyro_forecast.var.impulse_response) to die out.


## Parameters


`phi: Float[Array, ``" *#batch lags obs obs"]`  
Coefficient tensor `(*batch, lags, obs, obs)`.


## Returns


`Float[Array, ``"*batch lags_obs lags_obs"]`  
The companion matrix per batch element.


## Notes

JAX implements the eigenvalues of a nonsymmetric matrix on CPU only, so compute the spectral radius of posterior draws with NumPy: `np.abs(np.linalg.eigvals(np.asarray(companion_matrix(phi)))).max(-1)`.
