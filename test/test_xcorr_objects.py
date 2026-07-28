"""Unit / end-to-end tests for :mod:`lyavoid.xcorr_objects`."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from scipy.special import legendre

from lyavoid import xcorr_objects

from conftest import make_crosscorr, make_grid


# --- CrossCorr FITS round-trip and coordinate switch -----------------------


def test_crosscorr_fits_roundtrip(workdir):
    xi = np.random.default_rng(0).normal(size=(6, 20))
    xcf = make_crosscorr(xi=xi, name=str(workdir / "x.fits"))
    xcf.write()
    back = xcorr_objects.CrossCorr.init_from_fits(str(workdir / "x.fits"))
    np.testing.assert_allclose(back.xi_array, xi)
    np.testing.assert_allclose(back.r_array, xcf.r_array)
    np.testing.assert_allclose(back.mu_array, xcf.mu_array)
    assert back.rmu is True
    assert back.exported is True


def test_crosscorr_switch_roundtrip():
    xcf = make_crosscorr()
    r0, mu0 = xcf.r_array.copy(), xcf.mu_array.copy()
    xcf.switch()               # (r, mu) -> (rt, rp)
    assert xcf.rmu is False
    xcf.switch()               # back to (r, mu)
    assert xcf.rmu is True
    np.testing.assert_allclose(xcf.r_array, r0)
    np.testing.assert_allclose(xcf.mu_array, mu0)


# --- Multipole decomposition (analytic checks) -----------------------------


@pytest.mark.parametrize("method", ["simps", "nbody"])
def test_isotropic_gives_monopole(method):
    r1d, mu1d, R, MU = make_grid()
    xi = np.ones_like(MU)
    mult = xcorr_objects.Multipole.init_from_xcorr([0, 2], MU, xi, method, r_array=r1d)
    np.testing.assert_allclose(mult.poles[0], 1.0, atol=1e-2)
    np.testing.assert_allclose(mult.poles[2], 0.0, atol=2e-2)


def test_p2_input_gives_quadrupole():
    r1d, mu1d, R, MU = make_grid(nmu=60)
    xi = legendre(2)(MU)  # xi(mu) = P2(mu)
    mult = xcorr_objects.Multipole.init_from_xcorr([0, 2], MU, xi, "simps", r_array=r1d)
    np.testing.assert_allclose(mult.poles[0], 0.0, atol=1e-2)
    np.testing.assert_allclose(mult.poles[2], 1.0, atol=1e-2)


def test_unknown_method_raises():
    r1d, mu1d, R, MU = make_grid()
    with pytest.raises(KeyError):
        xcorr_objects.Multipole.init_from_xcorr(
            [0], MU, np.ones_like(MU), "not_a_method", r_array=r1d
        )


def test_trap_method_unavailable_in_recent_scipy():
    """`integrate.trapz` was removed in recent SciPy; the 'trap' method raises.

    Pinned so a SciPy pin (or a code fix to `trapezoid`) is noticed.
    """
    from scipy import integrate

    if hasattr(integrate, "trapz"):
        pytest.skip("this SciPy still provides integrate.trapz")
    r1d, mu1d, R, MU = make_grid()
    with pytest.raises(AttributeError):
        xcorr_objects.Multipole.init_from_xcorr(
            [0], MU, np.ones_like(MU), "trap", r_array=r1d
        )


# --- Multipole FITS round-trip and correlation rebuild ---------------------


def test_multipole_fits_roundtrip(workdir):
    r1d, mu1d, R, MU = make_grid()
    xi = np.ones_like(MU)
    mult = xcorr_objects.Multipole.init_from_xcorr(
        [0, 2], MU, xi, "simps", r_array=r1d, name=str(workdir / "m.fits")
    )
    mult.write_fits()
    back = xcorr_objects.Multipole.init_from_fits(str(workdir / "m.fits"))
    np.testing.assert_allclose(back.r_array, r1d)
    np.testing.assert_allclose(back.poles[0], mult.poles[0])
    np.testing.assert_allclose(back.poles[2], mult.poles[2])


def test_xcorr_from_monopole_rebuild(workdir):
    r1d, mu1d, R, MU = make_grid()
    mult = xcorr_objects.Multipole.init_from_xcorr(
        [0], MU, np.ones_like(MU), "simps", r_array=r1d
    )
    xcf = mult.xcorr_from_monopole(str(workdir / "rebuilt.fits"), nbins_mu=25)
    assert xcf.xi_array.shape == (len(r1d), 25)
    # isotropic monopole (= 1) -> the rebuilt xi is constant in mu at each r
    np.testing.assert_allclose(np.ptp(xcf.xi_array, axis=1), 0.0, atol=1e-9)
    np.testing.assert_allclose(xcf.xi_array, 1.0, atol=1e-9)
