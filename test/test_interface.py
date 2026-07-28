"""Unit tests for the config-parsing helpers in :mod:`lyavoid.interface`.

Importing ``interface`` pulls in ``xcorr`` (which needs ``lelantos``), so these
tests are skipped when the sibling lelantos checkout is not on the path.
"""

import pytest

from conftest import requires_lelantos

pytestmark = requires_lelantos


def test_parse_int_tuple():
    from lyavoid import interface

    assert interface.parse_int_tuple("2, 2, 300") == (2, 2, 300)
    assert interface.parse_int_tuple("None") is None


def test_parse_float_tuple():
    from lyavoid import interface

    assert interface.parse_float_tuple("7, 50") == (7.0, 50.0)
    assert interface.parse_float_tuple("None") is None


def test_parse_str_tuple():
    from lyavoid import interface

    assert interface.parse_str_tuple("a , b") == ("a", "b")
    assert interface.parse_str_tuple("None") is None


def test_parse_dict():
    from lyavoid import interface

    assert interface.parse_dict("{'n': 8}") == {"n": 8}
    assert interface.parse_dict("None") is None


def test_parse_scalars():
    from lyavoid import interface

    assert interface.parse_float("0.5") == pytest.approx(0.5)
    assert interface.parse_int("42") == 42
    assert interface.parse_string("hello") == "hello"
    assert interface.parse_float("None") is None
    assert interface.parse_int("None") is None
    assert interface.parse_string("None") is None
