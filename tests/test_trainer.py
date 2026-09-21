"""Smoke tests for the top-level DeepPLIV trainer.

These exercise the exact public entry point documented in the README and the
thesis appendix (``DeepPLIV().fit(...)``), since that is the surface most
likely to silently break when the two stages are refactored independently.
"""
import numpy as np
import pytest

from deeppliv import DeepPLIV


def _toy_iv_data(n=200, z_dim=2, x_dim=1, seed=0, binary_y=False):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, z_dim)).astype(np.float32)
    u = rng.standard_normal(n).astype(np.float32)
    v = (z.sum(axis=1) + u).reshape(-1, 1).astype(np.float32)
    x = rng.standard_normal((n, x_dim)).astype(np.float32)
    linear = 0.5 * v[:, 0] + x.sum(axis=1) + u
    if binary_y:
        prob = 1 / (1 + np.exp(-linear))
        y = (rng.random(n) < prob).astype(np.float32).reshape(-1, 1)
    else:
        y = linear.reshape(-1, 1).astype(np.float32)
    return v, z, x, y


def test_fit_end_to_end_does_not_raise():
    """Regression test: fit() previously crashed with
    'fit_first_stage() missing 1 required positional argument: validation_data'
    because it never passed validation_data through, and fit_first_stage had
    no default for it. This must keep working."""
    v, z, x, y = _toy_iv_data()
    model = DeepPLIV()
    first_stage, second_stage = model.fit(
        v_1=v, z_1=z, z_2=z, x=x, y=y,
        first_stage_epochs=2, second_stage_epochs=2,
    )
    assert first_stage is model.first_stage_model
    assert second_stage is model.second_stage_model


def test_predict_after_fit_returns_expected_shape():
    v, z, x, y = _toy_iv_data(n=100)
    model = DeepPLIV()
    model.fit(v_1=v, z_1=z, z_2=z, x=x, y=y, first_stage_epochs=2, second_stage_epochs=2)
    preds = model.predict(z=z, x=x)
    assert preds.shape == y.shape


def test_get_v_predicted_coefficient_returns_float():
    v, z, x, y = _toy_iv_data(n=100)
    model = DeepPLIV()
    model.fit(v_1=v, z_1=z, z_2=z, x=x, y=y, first_stage_epochs=2, second_stage_epochs=2)
    coef = model.get_v_predicted_coefficient()
    assert isinstance(coef, float)


def test_fit_first_stage_auto_splits_validation_when_not_given():
    """fit_first_stage should no longer require validation_data explicitly."""
    v, z, _, _ = _toy_iv_data(n=100)
    model = DeepPLIV()
    fitted = model.fit_first_stage(z, v, epochs_first_stage=2, learning_rate_first_stage=0.01)
    assert fitted is model.first_stage_model
    preds = model.predict_first_stage(z)
    assert preds.shape == v.shape


def test_fit_second_stage_rejects_2sps_for_binary_outcome():
    v, z, x, y = _toy_iv_data(n=100, binary_y=True)
    model = DeepPLIV()
    model.fit_first_stage(z, v, epochs_first_stage=2, learning_rate_first_stage=0.01)
    v_hat = model.predict_first_stage(z)
    with pytest.raises(ValueError, match="2SPS is inconsistent for binary outcomes"):
        model.fit_second_stage(
            v_hat, x, y, epochs_second_stage=2, learning_rate_second_stage=0.01,
            dropout=0.1, method="2sps",
        )
