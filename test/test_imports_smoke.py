"""Smoke tests: every module byte-compiles and imports (where deps allow).

``xcorr``, ``delta_creation`` and ``interface`` import ``lelantos``; those import
tests are skipped when the sibling lelantos checkout is not available, while
their syntax is still checked by the compile test.
"""

import importlib
import py_compile

import pytest

from conftest import PACKAGE_ROOT, requires_lelantos

CORE_MODULES = ["utils", "model", "autocorr", "xcorr_objects", "multipole"]
LELANTOS_MODULES = ["xcorr", "delta_creation", "interface"]

ALL_PY = sorted((PACKAGE_ROOT / "lyavoid").glob("*.py")) \
    + sorted((PACKAGE_ROOT / "scripts").glob("*.py"))


@pytest.mark.parametrize("path", ALL_PY, ids=[p.name for p in ALL_PY])
def test_module_compiles(path):
    """Every shipped .py file byte-compiles (no syntax errors)."""
    py_compile.compile(str(path), doraise=True)


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_module_imports(module):
    """Core modules import without lelantos/picca."""
    importlib.import_module(f"lyavoid.{module}")


@requires_lelantos
@pytest.mark.parametrize("module", LELANTOS_MODULES)
def test_lelantos_module_imports(module):
    """Modules that need lelantos import when it is on the path."""
    importlib.import_module(f"lyavoid.{module}")
