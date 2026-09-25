"""Time-axis reparameterization of in-sample latents (Haar / DCT).

Port of the ``time_reparam`` option of Pyro's forecasting module. Pyro wraps the
model with ``poutine.reparam(model, time_reparam_haar)`` in
[`Forecaster` / `HMCForecaster`](https://github.com/pyro-ppl/pyro/blob/dev/pyro/contrib/forecast/forecaster.py),
and the config callables in
[`pyro.contrib.forecast.util`](https://github.com/pyro-ppl/pyro/blob/dev/pyro/contrib/forecast/util.py)
return a ``HaarReparam`` / ``DiscreteCosineReparam`` for every non-observed site
inside the ``time`` plate. Neighboring steps of a random-walk latent are
strongly coupled, so its posterior is a long thin ellipse that a diagonal guide
cannot represent and a diagonal mass matrix explores slowly. Rotating the time
axis into a wavelet or frequency basis makes that posterior closer to diagonal,
and because the rotation is orthonormal the model's log density is unchanged:
only the coordinates the guide or sampler sees change.

NumPyro 0.22 ships the transforms and reparameterizers
(`numpyro.distributions.transforms.HaarTransform`,
`numpyro.distributions.transforms.DiscreteCosineTransform`,
`numpyro.infer.reparam.UnitJacobianReparam`, added in
[pyro-ppl/numpyro#2208](https://github.com/pyro-ppl/numpyro/pull/2208)) but not
Pyro's ``experimental_allow_batch`` flag: the transformed axis must be an event
dimension, while `~~numpyro_forecast.models.innovations()` samples its
in-sample site as a batch dimension under ``plate("time", t, dim=-2)``. Applied
naively the auxiliary site is re-expanded by that plate to ``(t, 1, t, 1)``.
This module therefore hides the auxiliary site from every enclosing plate
(``infer={"block_plates": ...}``, the mechanism
`numpyro.infer.reparam.NeuTraReparam` uses) and lets ``UnitJacobianReparam``
make the time axis and every plate axis to its right event axes. Plate axes to
the left of time stay batch axes of the auxiliary site, but without a plate
frame: the same split as Pyro's ``experimental_allow_batch`` ("the targeted
batch dimension and all batch dimensions to the right will be converted to
event dimensions").

Two deliberate differences from Pyro:

- Pyro's string mapping is swapped relative to its documentation:
  ``Forecaster(time_reparam="haar")`` routes to ``time_reparam_haar``, whose body
  returns ``DiscreteCosineReparam`` (and ``"dct"`` returns ``HaarReparam``).
  Here ``"haar"`` applies ``HaarTransform`` and ``"dct"`` applies
  ``DiscreteCosineTransform``.
- Discrete sites under the ``time`` plate are skipped instead of failing inside
  ``biject_to``.

The ``smooth`` (DCT) and ``flip`` (Haar) knobs of the underlying transforms are
intentionally not exposed. ``smooth=1`` whitens a Brownian latent, but
`~~numpyro_forecast.models.innovations()` sites are the white increments of the
walk, and on those it manufactures the ill-conditioning the transform is meant
to remove (NUTS hit the tree-depth cap on every iteration with a minimum
effective sample size below ten). Pyro's ``Forecaster`` does not expose them
either.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import DiscreteCosineTransform, HaarTransform, Transform
from numpyro.infer.reparam import Reparam, UnitJacobianReparam
from numpyro.primitives import _PYRO_STACK, Messenger

from numpyro_forecast.models import TIME_PLATE
from numpyro_forecast.typing import Array, ForecastModel

TimeTransform = Literal["haar", "dct"]
"""The time-axis transform applied by `time_reparam()`: ``"haar"`` or ``"dct"``."""

_AUX_INFER_KEY = "numpyro_forecast_time_reparam_aux"
_TRANSFORMS: dict[str, tuple[Callable[[int], Transform], str]] = {
    "haar": (lambda dim: HaarTransform(dim=dim), "haar"),
    "dct": (lambda dim: DiscreteCosineTransform(dim=dim), "dct"),
}


class _BlockPlates(Messenger):
    """Hide the sample sites issued inside this context from the named plates.

    Entered around the auxiliary ``numpyro.sample`` of `_TimeReparam`, it lands on
    top of NumPyro's handler stack, so it runs before the plates and can stamp the
    message with ``infer["block_plates"]`` (which `numpyro.plate` honors by
    neither expanding the distribution nor pushing its frame) and with a marker
    that keeps `_TimeReparamConfig` from targeting the auxiliary site again.
    """

    def __init__(self, plate_names: frozenset[str]) -> None:
        super().__init__()
        self.plate_names = plate_names

    def process_message(self, msg: dict[str, Any]) -> None:
        """Stamp ``block_plates`` and the auxiliary marker onto the message.

        The only message issued inside this context is the auxiliary sample.
        """
        infer = msg.setdefault("infer", {})
        infer["block_plates"] = self.plate_names
        infer[_AUX_INFER_KEY] = True


class _TimeReparam(UnitJacobianReparam):
    """`numpyro.infer.reparam.UnitJacobianReparam` whose auxiliary site ignores its plates.

    This plays the role of Pyro's ``experimental_allow_batch=True`` in
    [`pyro.infer.reparam.UnitJacobianReparam`](https://github.com/pyro-ppl/pyro/blob/dev/pyro/infer/reparam/unit_jacobian.py):
    the time axis and every plate axis to its right become event axes of the
    auxiliary site. The upstream ``__call__`` is reused unchanged; the
    `_BlockPlates` messenger entered around it hides the auxiliary sample from
    the plates, and ``UnitJacobianReparam._wrap`` re-expands the base
    distribution to the full plate shape before making the trailing axes an
    event, so no axis is lost.
    """

    def __init__(self, transform: Transform, suffix: str, plate_names: frozenset[str]) -> None:
        super().__init__(transform, suffix=suffix)
        self.plate_names = plate_names

    def __call__(
        self, name: str, fn: dist.Distribution, obs: Array | None
    ) -> tuple[dist.Distribution | None, Array | None]:
        """Sample ``f"{name}_{suffix}"`` outside the plates and return the inverse image."""
        with _BlockPlates(self.plate_names):
            return cast(
                tuple[dist.Distribution | None, Array | None], super().__call__(name, fn, obs)
            )


@dataclass(frozen=True)
class _TimeReparamConfig:
    """Callable ``config`` for `numpyro.handlers.reparam`, after Pyro's ``time_reparam_haar``.

    Pyro's config inspects ``msg["cond_indep_stack"]`` for a frame named
    ``time`` and returns ``HaarReparam(dim=frame.dim - fn.event_dim,
    experimental_allow_batch=True)``
    ([source](https://github.com/pyro-ppl/pyro/blob/dev/pyro/contrib/forecast/util.py)).
    This port reads the live `numpyro.plate` handlers on NumPyro's stack instead
    (the same private walk as ``models._reject_enclosing_plates``), for two
    reasons: `numpyro.handlers.scope` renames the frames after a plate has pushed
    them while the plate blocks by its own bare name, and the auxiliary site must
    be hidden from every enclosing plate, not only ``time``. The auxiliary sample
    carries an ``infer`` marker so it is never targeted a second time.
    """

    transform: TimeTransform

    def __call__(self, msg: Mapping[str, Any]) -> Reparam | None:
        """Return the reparameterizer for a sample site under the ``time`` plate, else ``None``."""
        if msg["type"] != "sample" or msg["is_observed"]:
            return None
        if (msg.get("infer") or {}).get(_AUX_INFER_KEY):
            return None
        fn = msg["fn"]
        if getattr(fn.support, "is_discrete", False):
            return None
        plates = [handler for handler in _PYRO_STACK if isinstance(handler, numpyro.plate)]
        time_plate = next((plate for plate in plates if plate.name == TIME_PLATE), None)
        if time_plate is None:
            return None
        dim = cast(int, time_plate.dim) - fn.event_dim
        make_transform, suffix = _TRANSFORMS[self.transform]
        return _TimeReparam(make_transform(dim), suffix, frozenset(plate.name for plate in plates))


def time_reparam(model: ForecastModel, transform: TimeTransform) -> ForecastModel:
    """Reparameterize every in-sample time latent of ``model`` along the time axis.

    Port of the ``time_reparam`` argument of Pyro's ``Forecaster`` and
    ``HMCForecaster``
    ([forecaster.py](https://github.com/pyro-ppl/pyro/blob/dev/pyro/contrib/forecast/forecaster.py)).
    The returned model is ``numpyro.handlers.reparam(model, config)`` with a
    config that targets every non-observed continuous sample site under the
    ``time`` plate opened by `~~numpyro_forecast.models.innovations()`, exactly
    as Pyro's ``time_reparam_haar`` targets every site inside its ``time``
    plate. Each targeted site ``name`` becomes a ``deterministic`` site of the
    same shape, computed from a new sample site ``f"{name}_haar"`` or
    ``f"{name}_dct"`` whose event shape owns the time axis and every plate axis
    to its right (plate axes to the left of time stay batch axes). The
    transform is orthonormal, so the model's log density is unchanged; only the
    geometry inference sees changes.

    Parameters
    ----------
    model
        A forecasting model ``(covariates, data=None) -> None``.
    transform
        ``"haar"`` for `numpyro.distributions.transforms.HaarTransform` (a
        multi-resolution average/difference basis, suited to blocky or
        multi-scale dynamics) or ``"dct"`` for
        `numpyro.distributions.transforms.DiscreteCosineTransform` (a cosine
        frequency basis, close to the Karhunen-Loeve basis of first-order Markov
        processes). Note that Pyro's own string mapping is swapped: its
        ``"haar"`` runs a DCT and its ``"dct"`` runs a Haar transform.

    Returns
    -------
    ForecastModel
        The wrapped model. It is a `numpyro.handlers.reparam` handler and is
        the single object to hand to the guide, ``SVI`` / ``MCMC`` / blackjax,
        `~~numpyro_forecast.predictive.forecast()`,
        `~~numpyro_forecast.predictive.predict_in_sample()`,
        `~~numpyro_forecast.convert.to_datatree()` and the ``model_fn`` of
        `~~numpyro_forecast.evaluate.backtest()`.

    Raises
    ------
    ValueError
        If ``model`` is already the result of `time_reparam()`. Nesting is not
        supported: the inner handler would turn the site into a deterministic
        before the outer one sees it, so the outer transform would silently be
        a no-op.

    Notes
    -----
    - Create the wrapped model once and reuse it: the drivers jit-compile with
      the model as a static argument, so wrapping again inside a loop recompiles.
    - Apply it innermost. ``scope(time_reparam(model), prefix="a")`` yields
      ``a/drift_haar``; ``time_reparam(scope(model, "a"))`` yields
      ``a/a/drift_haar`` because the auxiliary sample passes through ``scope``
      a second time (generic NumPyro reparam-under-scope behavior).
    - It composes with the per-block ``reparam=`` hook: after
      ``innovations(..., reparam=LocScaleReparam(0))`` the site under the plate
      is ``drift_decentered``, so the auxiliary site is
      ``drift_decentered_haar`` and both ``drift_decentered`` and ``drift``
      become deterministic. This matches Pyro, whose config applies to every
      site in the plate.
    - Untouched sites: the ``_future`` suffix sites (they stay prior-drawn under
      ``time_future``, so `~~numpyro_forecast.predictive.forecast()` is
      unaffected), the scan sites of
      `~~numpyro_forecast.models.markov_series()`, the error sites of
      `~~numpyro_forecast.models.ssoe()`, observed sites and discrete sites.
    - Posterior dictionaries from `~~numpyro_forecast.predictive.draw_posterior()`
      and ``mcmc.get_samples()`` contain both ``drift`` (deterministic) and
      ``drift_haar``; ``Predictive`` substitutes only the latter and recomputes
      the former. ``init_to_value`` must therefore target ``drift_haar``.
    - Measured on random-walk level models: mean-field ``AutoNormal`` reaches a
      better ELBO for both transforms (DCT slightly ahead of Haar), but an
      optimizer schedule tuned for the original coordinates does not transfer
      as is. The rotated posterior is better conditioned, so it tolerates and
      may need a larger learning rate to converge within the same step budget;
      a schedule that is too small stalls with part of the intercept still held
      by the level. NUTS trajectories become cheaper (fewer leapfrog steps per
      iteration) while the effective sample size per draw is model dependent.

    Examples
    --------
    ```python
    model_dct = time_reparam(seasonal_model, "dct")
    guide = AutoNormal(model_dct)
    svi = SVI(model_dct, guide, Adam(0.01), Trace_ELBO())
    svi_result = svi.run(key_fit, 1_500, covariates[:t_obs], data, progress_bar=False)
    posterior = draw_posterior(key_post, guide, svi_result.params, num_samples=100)
    samples = forecast(key_pred, model_dct, posterior, data, covariates)
    ```
    """
    if isinstance(model, numpyro.handlers.reparam) and isinstance(
        model.config, _TimeReparamConfig
    ):
        msg = "time_reparam cannot be nested: `model` is already a time_reparam model"
        raise ValueError(msg)
    wrapped = numpyro.handlers.reparam(model, config=_TimeReparamConfig(transform))
    return cast(ForecastModel, wrapped)
