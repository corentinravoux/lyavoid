"""Unit tests for :mod:`lyavoid.utils`."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lyavoid import utils


def test_return_key():
    assert utils.return_key({"a": 1}, "a", 99) == 1
    assert utils.return_key({"a": 1}, "b", 99) == 99


def test_lambda_ly_constant():
    # Lyman-alpha rest wavelength, in Angstrom
    assert 1215 < utils.lambdaLy < 1216


def test_set_size():
    fig, ax = plt.subplots()
    utils.set_size(4.0, 3.0, ax=ax)
    w, h = fig.get_size_inches()
    # the figure is enlarged so the axes reach the requested size
    assert w >= 4.0
    assert h >= 3.0
    plt.close(fig)
