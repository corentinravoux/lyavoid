"""Unit tests for :mod:`lyavoid.model` (linear cross-correlation models)."""

import numpy as np

from lyavoid import model, xcorr_objects

from conftest import make_grid


def test_simple_linear_model_scaling():
    r1d, mu1d, R, MU = make_grid()
    xi = np.ones_like(MU)
    m = model.SimpleLinearModel()
    fn = m.model(R, MU, xi)
    # b_v=b_a=2, no RSD -> xi scaled by 4
    out = fn(b_v=2, b_a=2, beta_v=0, beta_a=0)
    assert isinstance(out, xcorr_objects.CrossCorr)
    np.testing.assert_allclose(out.xi_array, 4.0 * xi)


def test_simple_linear_model_rsd_mu_dependence():
    r1d, mu1d, R, MU = make_grid()
    xi = np.ones_like(MU)
    fn = model.SimpleLinearModel().model(R, MU, xi)
    out = fn(b_v=1, b_a=1, beta_v=0.5, beta_a=0.0)
    # (1 + 0.5 mu^2) modulation
    np.testing.assert_allclose(out.xi_array, (1 + 0.5 * MU**2) * xi)


def test_simple_linear_model2_dict_params():
    r1d, mu1d, R, MU = make_grid()
    xi = np.ones_like(MU)
    fn = model.SimpleLinearModel2().model(R, MU, xi)
    out = fn({"b": 2.0, "beta": 0.0})
    np.testing.assert_allclose(out.xi_array, 4.0 * xi)


def test_return_model_dispatch():
    r1d, mu1d, R, MU = make_grid()
    xi = np.ones_like(MU)
    cm = model.CrossCorrModel(model_name="linear")
    fn = cm.return_model(R, MU, xi)
    assert callable(fn)
    assert isinstance(fn(), xcorr_objects.CrossCorr)
