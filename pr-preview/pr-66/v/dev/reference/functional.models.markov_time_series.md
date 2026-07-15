## functional.models.markov_time_series()


Sample a Markov (state-space) latent over the full horizon.


Usage

``` python
functional.models.markov_time_series(
    h,
    name,
    init_carry,
    transition,
    xs=None,
    *,
    plates=(),
    reparam_config=None
)
```


In-sample steps run in a `numpyro.contrib.control_flow.scan` with site `name`; when forecasting, horizon steps run in a second scan with site `f"{name}_future"` **seeded by the final in-sample carry**. The guide never sees the future site (same invariant as [time_series()](functional.models.time_series.md#numpyro_forecast.functional.models.time_series)), and under posterior replay the carry is a deterministic function of the replayed draws, so the forecast is conditioned through the state.


## Parameters


`h: Horizon`  
The horizon for the current model call (see [Horizon](functional.models.Horizon.md#numpyro_forecast.functional.models.Horizon)).

`name: str`  
Base sample-site name for the in-sample latent scan.

`init_carry: Any`  
Initial carry passed to the first transition.

`transition: Transition`  
Per-step `(carry, x_t) -> (dist_t, carry_fn)` callable; the wrapper owns the `numpyro.sample` statement.

`xs: Array | None = None`  
Optional exogenous inputs over the full horizon with time at axis `-2`, moved into scan layout internally; `None` for autonomous dynamics.

`plates: Sequence[tuple[str, int]] = ()`    
`(name, size)` pairs opened **inside** the scan body around the sample statement (the only placement NumPyro supports for scan + plate).

`reparam_config: Mapping[str, Reparam] | None = None`  
Site-name -\> `~numpyro.infer.reparam.Reparam` mapping applied **inside** the scan body.


## Returns


`Array`  
The latent over the full horizon in package layout `(*plate_batch, duration, obs)`.


## Raises


`ValueError`  
If forecasting without observed data, if the per-step shape lacks the observation dimension, or if an enclosing plate is detected.
