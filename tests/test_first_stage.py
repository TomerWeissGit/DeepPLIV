"""Smoke tests for NeuralNetworkFirstStage."""
import numpy as np
import torch

from deeppliv.models.first_stage import NeuralNetworkFirstStage


def _toy_data(n=100, d=3, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, d)).astype(np.float32)
    v = (z.sum(axis=1, keepdims=True) + 0.1 * rng.standard_normal((n, 1))).astype(np.float32)
    return z, v


def test_forward_pass_shape():
    z, v = _toy_data()
    model = NeuralNetworkFirstStage(input_dim=z.shape[1], output_dim=1, dropout=0.1)
    out = model(torch.tensor(z, dtype=torch.float32))
    assert out.shape == (z.shape[0], 1)


def test_train_new_data_runs_with_explicit_validation():
    z, v = _toy_data(n=100)
    z_train, v_train = z[:80], v[:80]
    z_val, v_val = z[80:], v[80:]
    model = NeuralNetworkFirstStage(input_dim=z.shape[1], output_dim=1, dropout=0.1)
    model.train_new_data(
        z_train, v_train,
        epochs=2,
        learning_rate=0.01,
        validation_data=(z_val, v_val),
    )
    preds = model.predict(z_val)
    assert preds.shape == v_val.shape


def test_predict_returns_numpy_array():
    z, v = _toy_data(n=50)
    model = NeuralNetworkFirstStage(input_dim=z.shape[1], output_dim=1, dropout=0.0)
    model.train_new_data(z, v, epochs=1, learning_rate=0.01, validation_data=(z, v))
    preds = model.predict(z)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (z.shape[0], 1)
