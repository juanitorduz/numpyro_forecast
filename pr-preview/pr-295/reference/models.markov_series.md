## models.markov_series()


Sample a Markov (state-space) latent over the full horizon.


Usage

``` python
models.markov_series(
    h,
    name,
    init_carry,
    transition,
    xs=None,
    *,
    advance=None,
    plates=(),
    reparam_config=None
)
```


In-sample steps run in a `numpyro.contrib.control_flow.scan` with site `name`; when forecasting, horizon steps run in a second scan with site `f"{name}_future"` **seeded by the final in-sample carry**. The guide never sees the future site (same invariant as [innovations()](models.innovations.md#numpyro_forecast.models.innovations)), and under posterior replay the carry is a deterministic function of the replayed draws, so the forecast is conditioned through the state.


## Parameters


`h: Horizon`  
The horizon for the current model call (see [Horizon](models.Horizon.md#numpyro_forecast.models.Horizon)).

`name: str`  
Base sample-site name for the in-sample latent scan.

`init_carry: Carry`  
Initial carry passed to the first transition.

`transition: Transition[Carry]`  
Per-step `(carry, x_t) -> dist_t` callable returning the distribution of the next latent (see [Transition](models.Transition.md#numpyro_forecast.models.Transition)); the wrapper owns the `numpyro.sample` statement.

`xs: PyTree[Array] | None = None`  
Optional exogenous inputs over the full horizon: a PyTree of arrays with time at axis `-2` and `duration` rows (a single array, a tuple, a dict, …), moved leaf by leaf into scan layout internally; `None` for autonomous dynamics. Slice each leaf from `covariates`, which has `duration` rows in every model call (training included), or extend an in-sample input with [pad_future()](arrays.pad_future.md#numpyro_forecast.arrays.pad_future).

`advance: Advance[Carry] | None = None`  
Optional `(carry, z_t, x_t) -> carry` (see [Advance](models.Advance.md#numpyro_forecast.models.Advance)) that builds the next carry from the sampled latent; `None` means the carry *is* the latent, `carry_{t+1} = z_t`, so `init_carry` must be a single array shaped like one draw. A vector autoregression with `p` lags keeps a `(p, obs)` window here.

`plates: Sequence[tuple[str, int]] = ()`  
`(name, size)` pairs opened **inside** the scan body around the sample statement (the only placement NumPyro supports for scan + plate), outermost first and with distinct names. The last pair goes at `dim=-2` and each earlier pair one dim to its left, so for an elementwise transition `[("store", 2), ("item", 3)]` draws `(2, 3, obs)` per step and returns a `(2, 3, duration, obs)` latent. The plate dims count from the right of the batch shape, so a transition with an `obs` event (a `MultivariateNormal`) needs a trailing size-1 batch axis, which the latent keeps: `(2, 3, 1, duration, obs)`; drop it with `z[..., 0, :, :]` before [predict()](models.predict.md#numpyro_forecast.models.predict). NumPyro autoguides do not support `plates` yet (issue \#294): fit such a model with MCMC or a hand-written guide.

`reparam_config: Mapping[str, Reparam] | None = None`  
Site-name -\> `numpyro.infer.reparam.Reparam` mapping applied **inside** the scan body.


## Returns


`Array`  
The latent over the full horizon in package layout `(*plate_batch, duration, obs)` (with the size-1 axis of a transition with an `obs` event before `duration`).


## Raises


`ValueError`  
If forecasting without observed data (only reachable with a hand-built [Horizon](models.Horizon.md#numpyro_forecast.models.Horizon): [Horizon.from_data()](models.Horizon.md#numpyro_forecast.models.Horizon.from_data) never sets `future > 0` without data), if an `xs` leaf lacks the time or observation axis or does not span `h.duration` rows, if the per-step shape lacks the observation dimension, if an enclosing plate is detected, if two `plates` pairs share a name, if the transition's batch shape has a size other than `1` or the plate size at a plate's dim, or if `advance` is omitted and `init_carry` is not a single array with the shape and dtype of a draw.
