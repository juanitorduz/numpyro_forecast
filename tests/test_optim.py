"""Tests for optimizer resolution (roadmap §1)."""

import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import optax
import pytest
from conftest import RandomWalkModel, empty_covariates
from jax import random
from numpyro.optim import Adam, _NumPyroOptim

from numpyro_forecast.functional import fit_svi, resolve_optimizer


def test_resolve_none_returns_adam() -> None:
    opt = resolve_optimizer(None)
    assert isinstance(opt, _NumPyroOptim)


def test_resolve_scalar_learning_rate() -> None:
    opt = resolve_optimizer(0.05)
    assert isinstance(opt, _NumPyroOptim)


def test_resolve_int_learning_rate() -> None:
    opt = resolve_optimizer(1)
    assert isinstance(opt, _NumPyroOptim)


def test_resolve_numpyro_optim_is_identity() -> None:
    adam = Adam(0.02)
    assert resolve_optimizer(adam) is adam


def test_resolve_zero_dim_array_accepted() -> None:
    # 0-d arrays are a runtime convenience outside the static union.
    opt = resolve_optimizer(jnp.asarray(0.01))  # ty: ignore[invalid-argument-type]
    assert isinstance(opt, _NumPyroOptim)
    opt_np = resolve_optimizer(np.float64(0.01))
    assert isinstance(opt_np, _NumPyroOptim)


@pytest.mark.parametrize("value", [True, False])
def test_resolve_rejects_bool(value: bool) -> None:
    with pytest.raises(TypeError, match="bool"):
        resolve_optimizer(value)


@pytest.mark.parametrize("value", [jnp.asarray(True), np.asarray(False)])
def test_resolve_rejects_zero_dim_bool_array(value: object) -> None:
    """A 0-d boolean array must not slip past as ``Adam(1.0)`` via the scalar path."""
    with pytest.raises(TypeError, match="bool"):
        resolve_optimizer(value)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_resolve_rejects_non_positive(value: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        resolve_optimizer(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_resolve_rejects_non_finite(value: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        resolve_optimizer(value)


def test_resolve_optax_transformation() -> None:
    opt = resolve_optimizer(optax.adam(0.01))
    assert isinstance(opt, _NumPyroOptim)


def test_resolve_rejects_unknown_type() -> None:
    with pytest.raises(TypeError, match="does not support"):
        resolve_optimizer("adam")  # ty: ignore[invalid-argument-type]


def test_scalar_path_does_not_import_optax() -> None:
    """The scalar/None paths must not import optax (soft-dependency proof)."""
    code = (
        "import sys, numpyro_forecast.functional as f; "
        "f.resolve_optimizer(0.01); f.resolve_optimizer(None); "
        "assert 'optax' not in sys.modules, 'optax was imported'; print('OK')"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, trusted input
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_svifit_carries_training_args() -> None:
    data = jnp.zeros((20, 1))
    covariates = empty_covariates(20)
    fit = fit_svi(random.PRNGKey(0), RandomWalkModel(), data, covariates, num_steps=5)
    assert fit.data is data
    assert fit.covariates is covariates


def test_optax_chain_reaches_adam_loss_magnitude() -> None:
    """optax.chain(clip, adam(schedule)) reaches Adam-baseline loss magnitude."""
    model = RandomWalkModel()
    t = jnp.linspace(0, 4 * jnp.pi, 60)
    data = (jnp.sin(t) + 0.1 * random.normal(random.PRNGKey(0), (60,)))[:, None]
    covariates = empty_covariates(60)

    baseline = fit_svi(random.PRNGKey(1), model, data, covariates, num_steps=400)
    schedule = optax.cosine_decay_schedule(1e-2, decay_steps=400)
    chained = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(schedule))
    tuned = fit_svi(random.PRNGKey(1), model, data, covariates, optim=chained, num_steps=400)

    base_loss = float(baseline.losses[-50:].mean())
    tuned_loss = float(tuned.losses[-50:].mean())
    # Loose statistical agreement: same order of magnitude of the final ELBO.
    assert abs(tuned_loss - base_loss) < 0.5 * abs(base_loss) + 20.0
