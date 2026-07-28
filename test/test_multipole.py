"""Unit tests for :mod:`lyavoid.multipole` helpers."""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from lyavoid import multipole

from conftest import make_grid


def test_get_poles_isotropic():
    r1d, mu1d, R, MU = make_grid()
    xi = np.ones_like(MU)
    monopole, dipole, quadrupole, hexadecapole = multipole.get_poles(MU, xi, "simps")
    np.testing.assert_allclose(monopole, 1.0, atol=1e-2)
    np.testing.assert_allclose(quadrupole, 0.0, atol=2e-2)
    np.testing.assert_allclose(hexadecapole, 0.0, atol=2e-2)


def test_get_poles_shapes():
    r1d, mu1d, R, MU = make_grid(nr=8)
    xi = np.ones_like(MU)
    poles = multipole.get_poles(MU, xi, "nbody")
    assert len(poles) == 4
    for pole in poles:
        assert pole.shape == (8,)
