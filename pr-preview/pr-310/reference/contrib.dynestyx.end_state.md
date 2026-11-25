## contrib.dynestyx.end_state()


Take the filtered state at the last step as the start of the next window.


Usage

``` python
contrib.dynestyx.end_state(
    samples,
    name,
)
```


The function reads the two sites that a dynestyx Gaussian filter records, `f"{name}_filtered_states_mean"` of shape `(*sample, time, state)` and `f"{name}_filtered_states_cov"` of shape `(*sample, time, state, state)`, with any number of leading sample axes. It returns their last row on the time axis under the names of the two sample sites that [start_state()](contrib.dynestyx.start_state.md#numpyro_forecast.contrib.dynestyx.start_state) reads. Add the result to the posterior of a driver call on a model function that calls [start_state()](contrib.dynestyx.start_state.md#numpyro_forecast.contrib.dynestyx.start_state): the model then continues the filter on new observations and does not read the earlier window again.

- dynestyx records each of the two sites by default only while it has at most `100_000` elements per draw (`time * state` for the mean, `time * state**2` for the covariance). Set `record_filtered_states_mean=True` and `record_filtered_states_cov=True` in the filter configuration to record them at any size.
- The function reads the covariance site, not the `f"{name}_filtered_states_chol_cov"` site: only the cuthbert backend records that one, and the diagonal of its factor can be negative.
- The function reads the whole `(sample, time, ...)` arrays to take one row, so the earlier run holds them in memory once.
- The state at an earlier step comes from the same samples. A filter is causal: the first `t` rows are the filter of the first `t` observations. Slice both sites to `[..., :t, :]` and `[..., :t, :, :]` before the call. One filter pass then serves several forecast origins.
- The two sites of a resumed run give the start of the next window, so the updates chain: each new window needs only the output of the window before.
- The tested filters are `KFConfig()`, `KFConfig(filter_source="cuthbert")` and `EKFConfig()`.


## Parameters


`samples: Mapping[str, Array | np.ndarray]`  
Draws that hold the two filtered sites: `mcmc.get_samples()` or the output of [draw_posterior()](predictive.draw_posterior.md#numpyro_forecast.predictive.draw_posterior) for a fit under a `Filter`, or the output of a plain `numpyro.infer.Predictive` call with `return_sites` under a `Filter`. Other keys are ignored.

`name: str`  
The site prefix of the [state_space()](contrib.dynestyx.state_space.md#numpyro_forecast.contrib.dynestyx.state_space) call.


## Returns


`dict[str, Array | np.ndarray]`  
`f"{name}_start_mean"`, shape `(*sample, state)`, and `f"{name}_start_cov"`, shape `(*sample, state, state)`. The two rows are copies, so `samples` can be dropped afterwards. A NumPy input gives NumPy values.


## Raises


`ValueError`  
If `samples` does not hold both filtered sites.


## See Also

[start_state()](contrib.dynestyx.start_state.md#numpyro_forecast.contrib.dynestyx.start_state)  
The model-side function that reads the two returned sites.

[state_space()](contrib.dynestyx.state_space.md#numpyro_forecast.contrib.dynestyx.state_space)  
The building block that runs the filter.


## Examples

The filtered state at the end of a window, from one pass of the filter at the parameter draws of a fit:

``` python
sites = ["f_filtered_states_mean", "f_filtered_states_cov"]
with Filter(filter_config=KFConfig(filter_source="cuthbert")):
    samples = Predictive(model, posterior_samples=parameters, return_sites=sites)(
        rng_key, series[:t_obs], series[:t_obs]
    )

posterior = parameters | end_state(samples, "f")

# the state at an earlier forecast origin t, from the same pass
earlier = {
    "f_filtered_states_mean": samples["f_filtered_states_mean"][..., :t, :],
    "f_filtered_states_cov": samples["f_filtered_states_cov"][..., :t, :, :],
}
posterior_at_t = parameters | end_state(earlier, "f")
```
