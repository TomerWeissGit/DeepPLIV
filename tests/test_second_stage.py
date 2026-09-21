"""Smoke tests for NeuralNetworkSecondStage."""
import numpy as np

from deeppliv.models.second_stage import NeuralNetworkSecondStage


def _toy_data(n=100, x_dim=2, seed=0, binary=False):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, x_dim)).astype(np.float32)
    v = rng.standard_normal((n, 1)).astype(np.float32)
    linear = 0.5 * v[:, 0] + x.sum(axis=1)
    if binary:
        prob = 1 / (1 + np.exp(-linear))
        y = (rng.random(n) < prob).astype(np.float32)
    else:
        y = (linear + 0.1 * rng.standard_normal(n)).astype(np.float32)
    return x, v, y


def test_forward_pass_shape():
    x, v, _ = _toy_data()
    model = NeuralNetworkSecondStage(x=x.shape[1], v=v.shape[1], dropout=0.1)
    import torch
    out = model(torch.tensor(x, dtype=torch.float32), torch.tensor(v, dtype=torch.float32))
    assert out.shape == (x.shape[0], 1)


def test_train_new_data_continuous_outcome():
    x, v, y = _toy_data(n=100, binary=False)
    model = NeuralNetworkSecondStage(x=x.shape[1], v=v.shape[1], dropout=0.1)
    model.train_new_data(x_exog=x, v_linear=v, y=y, epochs=2, learning_rate=0.01)
    assert model.binary is False
    preds = model.predict(x_new_exog=x, v_new_endog=v)
    assert preds.shape == (x.shape[0], 1)


def test_train_new_data_binary_outcome_uses_sigmoid():
    x, v, y = _toy_data(n=100, binary=True)
    model = NeuralNetworkSecondStage(x=x.shape[1], v=v.shape[1], dropout=0.1)
    model.train_new_data(x_exog=x, v_linear=v, y=y, epochs=2, learning_rate=0.01)
    assert model.binary is True
    preds = model.predict(x_new_exog=x, v_new_endog=v)
    assert preds.shape == (x.shape[0], 1)
    # Sigmoid output must lie in [0, 1].
    assert (preds >= 0).all() and (preds <= 1).all()


def test_causal_coefficient_is_accessible():
    x, v, y = _toy_data(n=50)
    model = NeuralNetworkSecondStage(x=x.shape[1], v=v.shape[1], dropout=0.0)
    model.train_new_data(x_exog=x, v_linear=v, y=y, epochs=1, learning_rate=0.01)
    coef = model.final_layer.weight[0, 0].item()
    assert isinstance(coef, float)
