# lyavoid test suite

Run from the package root:

```bash
pip install -e ".[dev]"   # installs pytest alongside the runtime deps
pytest -q                 # or: pytest test/
```

The object-level tests build synthetic `CrossCorr` grids directly (no data
files needed) and check the FITS round-trip, the multipole decomposition against
analytic cases, and the linear models. Modules that import `lelantos`
(`xcorr`, `delta_creation`, `interface`) are exercised when the sibling
`Packages/lelantos` checkout is found on the path, and skipped otherwise.

| File | Scope | What it covers |
|---|---|---|
| `test_utils.py` | unit | `return_key`, the Lyman-alpha constant, matplotlib axes sizing |
| `test_xcorr_objects.py` | unit + e2e | `CrossCorr` FITS round-trip and `(r,mu)`↔`(rt,rp)` switch; `Multipole` analytic monopole/quadrupole for isotropic and `P₂(µ)` inputs; multipole FITS round-trip; correlation rebuild from the monopole |
| `test_model.py` | unit | linear model bias/RSD scaling of `xi`, dict-parameter model, `return_model` dispatch |
| `test_multipole.py` | unit | `get_poles` monopole/quadrupole/hexadecapole shapes and isotropic values |
| `test_interface.py` | unit | config `parse_*` helpers (skipped without lelantos) |
| `test_imports_smoke.py` | smoke | every shipped module byte-compiles; core modules import; lelantos-dependent modules import when it is on the path |

## Known environment limitations (pinned, not worked around)

- **`integrate.trapz` was removed in recent SciPy**, so the `"trap"` /
  `"rebin_trap"` multipole integration methods raise `AttributeError`. This is
  pinned by `test_trap_method_unavailable_in_recent_scipy` (which `skip`s if the
  installed SciPy still provides `trapz`), so a SciPy pin or a `trapezoid` code
  fix is noticed. The `simps`, `nbody` and `rect` methods are unaffected and are
  used for the analytic checks.
- **`lelantos`** is required by `xcorr` / `delta_creation` / `interface`; those
  tests `skip` when the sibling checkout is absent.
