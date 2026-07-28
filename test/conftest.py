"""Shared fixtures and helpers for the lyavoid test suite.

lyavoid depends on `lelantos` (void catalogs / tomographic-map objects) and
`picca`; a few modules import `lelantos` at load time. That package lives in the
sibling ``Packages/lelantos`` checkout rather than being pip-installed, so this
conftest adds it (and the package root) to ``sys.path`` when it can be found.
Tests that need it are skipped gracefully otherwise.

The object-level tests build synthetic :class:`lyavoid.xcorr_objects.CrossCorr`
grids directly and exercise the FITS round-trip, the multipole decomposition and
the linear models, none of which need lelantos or picca.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

TEST_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_DIR.parent
# The lelantos checkout is a sibling package directory.
LELANTOS_ROOT = PACKAGE_ROOT.parent / "lelantos"

for _p in (str(PACKAGE_ROOT), str(TEST_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LELANTOS_AVAILABLE = LELANTOS_ROOT.is_dir()
if LELANTOS_AVAILABLE and str(LELANTOS_ROOT) not in sys.path:
    sys.path.insert(0, str(LELANTOS_ROOT))

requires_lelantos = pytest.mark.skipif(
    not LELANTOS_AVAILABLE, reason="lelantos package not found next to lyavoid"
)


def make_grid(nr=6, nmu=20, r_min=5.0, r_max=50.0, mu_edge=0.95):
    """Return an ``(r, mu)`` meshgrid of shape ``(nr, nmu)`` (indexing='ij')."""
    r1d = np.linspace(r_min, r_max, nr)
    mu1d = np.linspace(-mu_edge, mu_edge, nmu)
    R, MU = np.meshgrid(r1d, mu1d, indexing="ij")
    return r1d, mu1d, R, MU


def make_crosscorr(xi=None, nr=6, nmu=20, name=None, z=2.5):
    """Build a :class:`CrossCorr` on a synthetic ``(r, mu)`` grid.

    Args:
        xi (numpy.ndarray, optional): Correlation values of shape ``(nr, nmu)``;
            defaults to an isotropic ``xi = 1``.
        nr, nmu (int): Grid shape.
        name (str, optional): FITS file path.
        z (float): Constant mean redshift.

    Returns:
        CrossCorr: The synthetic cross-correlation.
    """
    from lyavoid import xcorr_objects

    r1d, mu1d, R, MU = make_grid(nr, nmu)
    if xi is None:
        xi = np.ones((nr, nmu))
    return xcorr_objects.CrossCorr(
        name=name,
        mu_array=MU,
        r_array=R,
        xi_array=xi,
        z_array=np.full((nr, nmu), z),
        exported=True,
        rmu=True,
    )


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run a test inside an isolated temporary working directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
