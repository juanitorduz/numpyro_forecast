"""Validated example forecasting models, kept for end-to-end tests.

Reference models for `tests.test_examples`: plain functional models (a pure
``(covariates, data=None)`` callable opening with ``Horizon.from_data``) that
mirror the example notebooks under ``docs/examples/`` and act as a regression
target for the full fit-draw-forecast path.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jaxtyping import Float
from numpyro.handlers import scope
from numpyro.infer.reparam import LocScaleReparam

from numpyro_forecast.arrays import pad_future
from numpyro_forecast.features import periodic_repeat
from numpyro_forecast.models import Horizon, SSOEResult, innovations, predict, ssoe
from numpyro_forecast.typing import Array, ForecastModel


def univariate_model(covariates: Array, data: Array | None = None) -> None:
    """Local level + regression model with Student-T observations.

    The mean is ``bias + level_t + weight @ covariates_t`` where ``level`` is a
    Gaussian random walk (``LocScaleReparam`` improves the SVI geometry). The
    regression design ``covariates`` is supplied by the caller (e.g. Fourier
    features from `numpyro_forecast.features.fourier_features()`).

    Parameters
    ----------
    covariates
        Regression design with time at axis ``-2``, shape
        ``(duration, num_features)``.
    data
        Observed data with time at axis ``-2`` (``None`` for prior sampling).
    """
    h = Horizon.from_data(covariates, data)
    num_features = covariates.shape[-1]

    bias = numpyro.sample("bias", dist.Normal(0.0, 10.0))
    weight = numpyro.sample("weight", dist.Normal(0.0, 0.1).expand([num_features]).to_event(1))
    drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-20.0, 5.0))
    nu = numpyro.sample("nu", dist.Gamma(10.0, 2.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(-5.0, 5.0))
    centered = numpyro.sample("centered", dist.Uniform(0.0, 1.0))

    drift = innovations(
        h,
        "drift",
        dist.Normal(0.0, drift_scale),
        reparam=LocScaleReparam(centered=centered),
    )
    level = jnp.cumsum(drift, axis=-2)
    regression = (weight * covariates).sum(axis=-1, keepdims=True)
    prediction = level + bias + regression

    predict(h, dist.StudentT(df=nu, loc=0.0, scale=sigma), prediction)


def make_hierarchical_model(period: int = 24 * 7) -> ForecastModel:
    """Build a hierarchical OD model with per-station seasonality and drift.

    Data and covariates use the ``(origin, time, destin)`` layout. ``covariates``
    is only used for its shape (a dummy zero panel, as in the tutorial).

    Parameters
    ----------
    period
        Seasonal period in time steps (default ``24 * 7`` for hourly data).

    Returns
    -------
    ForecastModel
        A callable ``(covariates, data=None) -> None`` closed over ``period``.
    """

    def hierarchical_model(covariates: Array, data: Array | None = None) -> None:
        """Define the hierarchical forecasting model.

        Parameters
        ----------
        covariates
            Dummy covariates shaped ``(origin, duration, destin)``.
        data
            Observed data with time at axis ``-2`` (``None`` for prior sampling).
        """
        h = Horizon.from_data(covariates, data)
        n_origin = covariates.shape[-3]
        n_destin = covariates.shape[-1]
        duration = covariates.shape[-2]

        origin_plate = numpyro.plate("origin", n_origin, dim=-3)
        destin_plate = numpyro.plate("destin", n_destin, dim=-1)
        hour_plate = numpyro.plate("hour_of_week", period, dim=-2)

        drift_scale = numpyro.sample("drift_scale", dist.LogNormal(-20.0, 5.0))
        destin_centered = numpyro.sample("destin_centered", dist.Uniform(0.0, 1.0))

        with origin_plate, hour_plate:
            origin_seasonal = numpyro.sample("origin_seasonal", dist.Normal(0.0, 5.0))
        with hour_plate, destin_plate:
            destin_seasonal = numpyro.sample("destin_seasonal", dist.Normal(0.0, 5.0))

        with destin_plate:
            drift = innovations(
                h,
                "drift",
                dist.Normal(0.0, drift_scale),
                reparam=LocScaleReparam(centered=destin_centered),
            )
        level = jnp.cumsum(drift, axis=-2)

        with origin_plate, destin_plate:
            pairwise = numpyro.sample("pairwise", dist.Normal(0.0, 1.0))

        with origin_plate:
            origin_scale = numpyro.sample("origin_scale", dist.LogNormal(-5.0, 5.0))
        with destin_plate:
            destin_scale = numpyro.sample("destin_scale", dist.LogNormal(-5.0, 5.0))
        scale = origin_scale + destin_scale

        seasonal = jnp.asarray(origin_seasonal + destin_seasonal)
        seasonal_repeat = periodic_repeat(seasonal, duration, axis=-2)
        prediction = level + seasonal_repeat + pairwise

        predict(h, dist.Normal(0.0, scale), prediction)

    return hierarchical_model


def make_trend_inflation_model(
    num_measures: int,
    *,
    vol_of_vol: float | Array = 0.1,
    stochastic_volatility: bool = True,
    student_t: bool = False,
) -> ForecastModel:
    r"""Build the multi-measure trend inflation model of ``trend_inflation.ipynb``.

    Every measure is a common random-walk trend plus a measure offset (zero
    for the last measure) plus a transitory gap. The log-variances of the trend
    shock and of each gap are random walks with the fixed vol-of-vol
    ``vol_of_vol``, and the random walks are built from standardized steps with
    ``jnp.cumsum`` (non-centered form).

    Parameters
    ----------
    num_measures
        Number of price measures (the observation axis of ``data``).
    vol_of_vol
        Standard deviation of the log-variance random walk steps, $\gamma$: a fixed
        number or a value sampled by an enclosing model.
    stochastic_volatility
        Whether the log-variances move over time; ``False`` keeps them at their
        initial values (constant variances).
    student_t
        Whether the gaps are Student-t instead of Normal.

    Returns
    -------
    ForecastModel
        A callable ``(covariates, data=None) -> None`` that reads only the
        shape of ``covariates`` (pass ``jnp.zeros((duration, 0))``).
    """

    def trend_inflation_model(covariates: Array, data: Array | None = None) -> None:
        """Define the trend inflation model.

        Parameters
        ----------
        covariates
            Dummy covariates shaped ``(duration, 0)``; only the duration is read.
        data
            Observed rates ``(duration, num_measures)`` (``None`` for prior sampling).
        """
        h = Horizon.from_data(covariates, data)
        measure_plate = numpyro.plate("measure", num_measures, dim=-1)

        if num_measures > 1:
            alpha = numpyro.sample(
                "alpha", dist.Normal(0.0, 1.0).expand([num_measures - 1]).to_event(1)
            )
            offset = jnp.concatenate([jnp.asarray(alpha), jnp.zeros(1)])
        else:
            offset = jnp.zeros(1)
        numpyro.deterministic("offset", offset)

        tau0 = numpyro.sample("tau0", dist.Normal(3.0, 2.0))
        h0 = numpyro.sample("h0", dist.Normal(-1.5, 1.0))
        trend_z = innovations(h, "trend_z", dist.Normal(0.0, 1.0))
        if stochastic_volatility:
            trend_log_var_z = innovations(h, "trend_log_var_z", dist.Normal(0.0, 1.0))
            log_var_trend = h0 + vol_of_vol * jnp.cumsum(trend_log_var_z, axis=-2)
        else:
            log_var_trend = h0 * jnp.ones_like(trend_z)
        trend_vol = jnp.asarray(numpyro.deterministic("trend_vol", jnp.exp(log_var_trend / 2.0)))
        trend = jnp.asarray(
            numpyro.deterministic("trend", tau0 + jnp.cumsum(trend_vol * trend_z, axis=-2))
        )

        with measure_plate:
            g0 = numpyro.sample("g0", dist.Normal(0.0, 1.0))
        if stochastic_volatility:
            with measure_plate:
                gap_log_var_z = innovations(h, "gap_log_var_z", dist.Normal(0.0, 1.0))
            log_var_gap = g0 + vol_of_vol * jnp.cumsum(gap_log_var_z, axis=-2)
        else:
            log_var_gap = jnp.broadcast_to(g0, (h.duration, num_measures))
        gap_vol = numpyro.deterministic("gap_vol", jnp.exp(log_var_gap / 2.0))

        prediction = trend + offset
        if student_t:
            nu = numpyro.sample("nu", dist.Gamma(10.0, 1.0))
            predict(h, dist.StudentT(nu, 0.0, gap_vol), prediction)
        else:
            predict(h, dist.Normal(0.0, gap_vol), prediction)

    return trend_inflation_model


def low_frequency_basis(num_periods: int, num_terms: int) -> Float[Array, " periods terms"]:
    r"""Leading eigenvectors of a demeaned random walk, scaled by the root eigenvalues.

    Müller and Watson (2026, Section 3.5) write the covariance matrix of a
    demeaned random walk of length $T$ with unit innovation variance as
    $\bar\Sigma_{RW} = \sum_l \varphi_l \varphi_l'$ with
    $\varphi_l = \lambda_l^{1/2} e_l$, and keep the $q$ columns with the largest
    eigenvalues: the demeaned walk is approximately $\sum_{l \le q} \varphi_l \xi_l$
    with iid $\xi_l$ of the innovation variance. Column $l$ is close to a cosine
    of $l$ half-periods over the sample and has amplitude of order
    $\sqrt{2T}/(l\pi)$. The sign of each column is fixed by a positive first row.

    Parameters
    ----------
    num_periods
        Length $T$ of the walk.
    num_terms
        Number of columns $q$ kept (the largest eigenvalues first).

    Returns
    -------
    Float[Array, " periods terms"]
        The basis $(\varphi_1, \dots, \varphi_q)$ as a ``(num_periods, num_terms)`` array.
    """
    cumsum = np.tril(np.ones((num_periods, num_periods)))
    demean = np.eye(num_periods) - np.ones((num_periods, num_periods)) / num_periods
    covariance = demean @ cumsum @ cumsum.T @ demean.T
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:num_terms]
    basis = eigenvectors[:, order] * np.sqrt(eigenvalues[order])
    return jnp.asarray(basis * np.sign(basis[0])[None, :], dtype=jnp.float32)


MAX_LOG_NU = 8.0


def make_rts_model(
    y_init: Array,
    *,
    hierarchical: bool = True,
    student_t: bool = False,
    outliers: bool = False,
    stochastic_volatility: bool = False,
) -> ForecastModel:
    r"""Build the related time series ladder of ``related_time_series.ipynb``.

    Each series $j$ is an AR(12) around its mean, $y_{j,t} = \mu_j + u_{j,t} + o_{j,t}$
    with $u_{j,t} = \sum_{l \le 12} \phi_{j,l} u_{j,t-l} + \sigma_{j,t} \varepsilon_{j,t}$,
    run through `~~numpyro_forecast.models.ssoe()` on the observed history
    (Müller and Watson, 2026, Models I to V). The switches add the paper's
    features one at a time: hierarchical priors that pool $\phi_{j,l}$ and
    $\ln \sigma_j^2$ across series, Student-t innovations with pooled degrees of
    freedom, Student-t additive outliers $o_{j,t} = \kappa_j \eta_{j,t}$ with
    $\nu^o = 4$ that leave the lag window (the recursion runs on $y - o$), and
    a low-frequency random walk in $\ln \sigma^2_{j,t}$ on
    `low_frequency_basis` with $q = \lfloor (t_{\text{obs}} + 12) / 36 \rfloor$
    terms. The paper's common scale $\omega$ is absorbed into the location of
    the $\ln \sigma_j^2$ prior (site ``log_omega2``), the coefficient, degrees of
    freedom, outlier scale and volatility hierarchies are non-centered by hand (the
    innovation scales stay centered), and the first 12 rows of the panel, ``y_init``, seed
    the lag window instead of the paper's stationary prior on the initial values.

    Parameters
    ----------
    y_init
        The 12 rows before the first likelihood row, shape ``(12, states)``.
    hierarchical
        Pool $\phi_{j,l}$ and $\ln \sigma_j^2$ across series (Model II); otherwise
        the Minnesota prior $\phi_{j,l} \sim \text{Normal}(0, 0.2 / l)$ and
        $\ln \sigma_j^2 \sim \text{Normal}(\ln \omega^2, 0.3^2)$ of Model I.
    student_t
        Student-t innovations with series-specific $\nu_j$ (Model III);
        $\ln(\nu_j - 2)$ is capped at ``MAX_LOG_NU`` ($\nu_j \le 2 + e^8$), above which
        a Student-t is Normal to float32 precision and the float32
        ``StudentT.log_prob`` is inaccurate.
    outliers
        Student-t additive outliers (Model IV).
    stochastic_volatility
        Low-frequency random walk in the log innovation variance (Model V).

    Returns
    -------
    ForecastModel
        A callable ``(covariates, data=None) -> None`` that reads the observed
        history from ``covariates[..., :t_obs, :]`` (the panel itself, shape
        ``(duration, states)``); ``data`` is the same panel while training.
    """
    num_lags, num_states = y_init.shape
    lags = jnp.arange(1, num_lags + 1, dtype=jnp.float32)
    minnesota_scale = 0.2 / lags  # (12,): the prior sd of phi_{j,l} shrinks with the lag

    def rts_model(covariates: Array, data: Array | None = None) -> None:
        """Define the related time series model.

        Parameters
        ----------
        covariates
            The growth panel ``(duration, states)``; rows past ``t_obs`` are unread.
        data
            Observed growth ``(t_obs, states)`` (``None`` for prior sampling).
        """
        h = Horizon.from_data(covariates, data)
        y = covariates[..., : h.t_obs, :]  # (t_obs, states): the observed history
        state_plate = numpyro.plate("state", num_states, dim=-1)
        lag_plate = numpyro.plate("lag", num_lags, dim=-2)

        mu = jnp.asarray(
            numpyro.sample("mu", dist.Normal(0.0, 10.0).expand([num_states]).to_event(1))
        )
        log_omega2 = numpyro.sample("log_omega2", dist.Normal(0.0, 10.0))
        if hierarchical:
            phi_loc = jnp.asarray(
                numpyro.sample("phi_loc", dist.Normal(0.0, 0.5 * minnesota_scale).to_event(1))
            )
            log_v_phi = jnp.asarray(
                numpyro.sample(
                    "log_v_phi", dist.Normal(jnp.log(minnesota_scale**2), 0.5).to_event(1)
                )
            )
            with lag_plate, state_plate:
                phi_z = jnp.asarray(numpyro.sample("phi_z", dist.Normal(0.0, 1.0)))
            phi = jnp.asarray(
                numpyro.deterministic(
                    "phi", phi_loc[:, None] + jnp.exp(0.5 * log_v_phi)[:, None] * phi_z
                )
            )
            log_v_sigma = numpyro.sample("log_v_sigma", dist.Normal(jnp.log(0.3**2), 0.5))
            with state_plate:
                log_sigma2 = jnp.asarray(
                    numpyro.sample(
                        "log_sigma2", dist.Normal(log_omega2, jnp.exp(0.5 * log_v_sigma))
                    )
                )
        else:
            with lag_plate, state_plate:
                phi = jnp.asarray(
                    numpyro.sample("phi", dist.Normal(0.0, minnesota_scale[:, None]))
                )
            with state_plate:
                log_sigma2 = jnp.asarray(
                    numpyro.sample("log_sigma2", dist.Normal(log_omega2, 0.3))
                )

        if student_t:
            nu_loc = numpyro.sample("nu_loc", dist.Normal(jnp.log(10.0), 0.5))
            log_v_nu = numpyro.sample("log_v_nu", dist.Normal(jnp.log(0.5**2), 0.5))
            with state_plate:
                log_nu_z = jnp.asarray(numpyro.sample("log_nu_minus_2_z", dist.Normal(0.0, 1.0)))
            log_nu_minus_2 = jnp.minimum(nu_loc + jnp.exp(0.5 * log_v_nu) * log_nu_z, MAX_LOG_NU)
            nu = jnp.asarray(numpyro.deterministic("nu", 2.0 + jnp.exp(log_nu_minus_2)))

        if outliers:
            # The paper centers the kappa hierarchy at ln(0.1^2) relative to omega; with
            # omega absorbed the location moves with log_omega2 and the paper's two
            # 0.5^2 variances add up.
            kappa_loc = numpyro.sample(
                "kappa_loc", dist.Normal(log_omega2 + jnp.log(0.1**2), jnp.sqrt(0.5**2 + 0.5**2))
            )
            log_v_kappa = numpyro.sample("log_v_kappa", dist.Normal(jnp.log(0.3**2), 0.5))
            with state_plate:
                log_kappa2_z = jnp.asarray(numpyro.sample("log_kappa2_z", dist.Normal(0.0, 1.0)))
            kappa = jnp.asarray(
                numpyro.deterministic(
                    "kappa", jnp.exp(0.5 * (kappa_loc + jnp.exp(0.5 * log_v_kappa) * log_kappa2_z))
                )
            )
            with numpyro.plate("time", h.t_obs, dim=-2), state_plate:
                eta = jnp.asarray(numpyro.sample("outlier_eta", dist.StudentT(4.0, 0.0, 1.0)))
            outlier = jnp.asarray(numpyro.deterministic("outlier", kappa * eta))  # (t_obs, states)
        else:
            outlier = jnp.zeros_like(y)

        if stochastic_volatility:
            num_terms = max(1, (h.t_obs + num_lags) // 36)  # the paper counts T from 1990m2
            basis = low_frequency_basis(h.t_obs, num_terms)  # (t_obs, q)
            xi_loc = jnp.asarray(
                numpyro.sample("xi_loc", dist.Normal(0.0, 0.01).expand([num_terms]).to_event(1))
            )
            log_v_xi = numpyro.sample("log_v_xi", dist.Normal(jnp.log(0.01**2), 0.5))
            with numpyro.plate("term", num_terms, dim=-2), state_plate:
                xi_z = jnp.asarray(numpyro.sample("xi_z", dist.Normal(0.0, 1.0)))
            xi = xi_loc[:, None] + jnp.exp(0.5 * log_v_xi) * xi_z  # (q, states)
            log_var = jnp.asarray(numpyro.deterministic("log_var", log_sigma2 + basis @ xi))
            scale = jnp.exp(0.5 * log_var)  # (t_obs, states)
            if h.future > 0:
                with numpyro.plate("time_future", h.future, dim=-2), state_plate:
                    log_var_z = jnp.asarray(
                        numpyro.sample("log_var_z_future", dist.Normal(0.0, 1.0))
                    )
                # The log variance continues as a random walk with step variance v_xi.
                log_var_future = log_var[-1] + jnp.exp(0.5 * log_v_xi) * jnp.cumsum(
                    log_var_z, axis=0
                )
                scale_future = jnp.exp(0.5 * log_var_future)  # (future, states)
            else:
                scale_future = jnp.exp(0.5 * log_sigma2)
        else:
            scale = jnp.exp(0.5 * log_sigma2)  # (states,)
            scale_future = scale

        # AR(12) around the mean: y_t - mu = sum_l phi_l (y_{t-l} - mu) + error, written
        # with the intercept mu (1 - sum_l phi_l) on the lag window of y itself.
        intercept = mu * (1.0 - phi.sum(axis=0))  # (states,)

        def mean(window: Array, _: object) -> Array:
            """Conditional mean from the lag window ``(12, states)``, most recent row last."""
            return intercept + jnp.einsum("ls,ls->s", phi, window[::-1])

        def update(window: Array, y_t: Array, eps_t: Array, _: object) -> Array:
            """Shift the lag window by one row."""
            return jnp.concatenate([window[1:], y_t[None]], axis=0)

        noise = (
            dist.StudentT(nu, 0.0, scale_future) if student_t else dist.Normal(0.0, scale_future)
        )
        # The recursion runs on the series without its additive outliers, so an outlier
        # never enters the lag window.
        r = ssoe(h, "eps", y - outlier, y_init, mean, update, noise)
        mu_t = jnp.asarray(numpyro.deterministic("mu_t", r.mu))  # (t_obs, states)
        obs_dist = (
            dist.StudentT(nu, mu_t + outlier, scale)
            if student_t
            else dist.Normal(mu_t + outlier, scale)
        )
        with numpyro.plate("time", h.t_obs, dim=-2), state_plate:
            numpyro.sample("obs", obs_dist, obs=h.data)
        if h.future > 0:
            y_future = r.y_future
            loc_future = r.mu_future  # (future, states): the one-step means over the horizon
            if outliers:
                with numpyro.plate("time_future", h.future, dim=-2), state_plate:
                    eta_future = jnp.asarray(
                        numpyro.sample("outlier_eta_future", dist.StudentT(4.0, 0.0, 1.0))
                    )
                y_future = y_future + kappa * eta_future
                loc_future = loc_future + kappa * eta_future
            numpyro.deterministic("forecast", y_future)
            # The location and scale of the one-step predictive given the draw, for the
            # leave-future-out log score of the notebook.
            numpyro.deterministic("forecast_loc", loc_future)
            numpyro.deterministic("forecast_scale", jnp.broadcast_to(scale_future, y_future.shape))

    return rts_model


def _level_channel(h: Horizon, name: str, values: Array, gate: Array) -> tuple[SSOEResult, Array]:
    """Gated simple exponential smoothing level channel on `ssoe()`.

    Samples the channel priors (``smoothing``, ``init``, ``noise``) and runs the
    where-gated level recursion; the level updates only where ``gate`` is true
    and is frozen over the forecast horizon (``pad_future`` zeroes the gate
    there), so the forecast is the last level plus iid errors. Meant to be
    called under `numpyro.handlers.scope()`, which prefixes the parameter
    sites and the block's ``f"{name}_future"`` site per channel.

    Parameters
    ----------
    h
        The train/forecast horizon for the current model call.
    name
        Error-site name handed to `ssoe()` (``"eps"`` under a scope).
    values
        Driving series ``(t_obs, 1)``; read only where ``gate`` is true.
    gate
        Boolean update gate ``(t_obs, 1)``.

    Returns
    -------
    tuple[SSOEResult, Array]
        The block result (in-sample means, forecast means, sampled future
        values) and the observation noise scale.
    """
    smoothing = jnp.asarray(numpyro.sample("smoothing", dist.Beta(2.0, 20.0)))
    init = jnp.asarray(numpyro.sample("init", dist.Normal(0.0, 1.0)))
    noise = jnp.asarray(numpyro.sample("noise", dist.HalfNormal(1.0)))

    def mean(level: Array, _: Array | None) -> Array:
        return level

    def update(level: Array, y_t: Array, _: Array, gate_t: Array | None) -> Array:
        assert gate_t is not None
        return jnp.where(gate_t, smoothing * y_t + (1.0 - smoothing) * level, level)

    result = ssoe(
        h,
        name,
        values,
        init[None],
        mean,
        update,
        dist.Normal(0.0, noise),
        xs=pad_future(gate, h.future),
    )
    return result, noise


def croston_model(covariates: Array, data: Array | None = None) -> None:
    """Croston's method as two scoped level channels composed on `ssoe()`.

    The observed demand series doubles as the covariate: ``covariates`` is the
    series over the full horizon with shape ``(duration, 1)`` and only its first
    ``h.t_obs`` rows are read (the block's ``t_obs`` check is the leak guard).
    The demand-size channel ``z`` smooths the demand at demand events; the
    inverse-interval channel ``p_inv`` smooths ``1 / interval`` at the same
    events. The likelihoods and the ``"forecast"`` product are registered
    outside the scopes, from the channels' results.

    Parameters
    ----------
    covariates
        The demand series over the full horizon, shape ``(duration, 1)``.
    data
        Observed demand with time at axis ``-2`` (``None`` for prior sampling
        and under ``predict_in_sample``).
    """
    h = Horizon.from_data(covariates, data)
    y = covariates[..., : h.t_obs, :]
    is_demand = y > 0
    idx = jnp.arange(h.t_obs)[:, None]
    last_at_or_before = jax.lax.cummax(jnp.where(is_demand, idx, -1), axis=0)
    last_before = jnp.concatenate([jnp.full((1, 1), -1), last_at_or_before[:-1]])
    p_inv_obs = 1.0 / (idx - last_before).astype(y.dtype)

    z, z_noise = scope(_level_channel, "z", divider="_")(h, "eps", y, is_demand)
    p_inv, p_inv_noise = scope(_level_channel, "p_inv", divider="_")(
        h, "eps", p_inv_obs, is_demand
    )

    numpyro.deterministic("rate", z.mu * p_inv.mu)
    numpyro.sample("obs", dist.Normal(z.mu, z_noise).mask(is_demand), obs=h.data)
    numpyro.sample(
        "obs_intervals", dist.Normal(p_inv.mu, p_inv_noise).mask(is_demand), obs=p_inv_obs
    )
    if h.future > 0:
        numpyro.deterministic("rate_future", z.mu_future * p_inv.mu_future)
        numpyro.deterministic("forecast", z.y_future * p_inv.y_future)
