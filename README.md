# lyavoid

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21646147.svg)](https://doi.org/10.5281/zenodo.21646147)
[![tests](https://github.com/corentinravoux/lyavoid/actions/workflows/tests.yml/badge.svg)](https://github.com/corentinravoux/lyavoid/actions/workflows/tests.yml)

Compute the **cross-correlation between the Lyman-alpha forest and the voids**
detected in 3-D tomographic maps (voids found with the
[lelantos](https://github.com/corentinravoux/lelantos) package), and model its
redshift-space multipoles.

📖 **Full documentation (API reference + user guide):** **<https://corentinravoux.github.io/lyavoid/>**

---

## What it does

```
tomographic map (lelantos) ─▶ void catalog ─┐
                                            ├─▶ xcorr        xi(r, mu)
Lyman-alpha deltas (picca / delta_creation) ┘   ─▶ multipole   monopole / quadrupole / ...
                                                 ─▶ model      linear RSD fit
```

## Install

```bash
pip install .              # core deps: fitsio, numpy, scipy, matplotlib
pip install -e ".[dev]"    # + pytest, to run the test suite
```

lyavoid also uses [picca](https://github.com/igmhub/picca) (cross-correlation /
export back-end), [lelantos](https://github.com/corentinravoux/lelantos) (void
catalog + tomographic-map objects), `numba` (auto-correlation) and `iminuit`
(model fitting).

## Quick start

```python
from lyavoid import xcorr_objects

# load a picca-exported cross-correlation
xcf = xcorr_objects.CrossCorr.init_from_fits("xcorr_exp.fits")
r, mu, xi = xcf.r_array, xcf.mu_array, xcf.xi_array   # r in h^-1 Mpc, mu = r_par/r

# decompose into Legendre multipoles
mult = xcorr_objects.Multipole.init_from_xcorr(
    ell=[0, 2, 4], mu=xcf.mu_array, xi=xcf.xi_array,
    method="rect", r_array=xcf.r_array,
)
monopole, quadrupole = mult.poles[0], mult.poles[2]
```

Run the whole pipeline from a config file:

```bash
lyavoid_interface.py config.ini
```

## Package map

| Module | Role |
|---|---|
| `xcorr` | Void x forest cross-correlation `xi(r, mu)` / `xi(rp, rt)` |
| `xcorr_objects` | `CrossCorr` and `Multipole` data objects (FITS I/O, plots) |
| `multipole` | Legendre multipoles, wedges, 2-D and comparison plots |
| `model` | Linear RSD/bias models + iminuit `Fitter` |
| `autocorr` | Forest delta x delta auto-correlation (numba-accelerated) |
| `delta_creation` | Build picca delta files from a tomographic matter cube |
| `interface` | Config-driven (`.ini`) computation + plotting |
| `utils` | Constants and matplotlib helpers |

`scripts/` holds the CLI entry point (`lyavoid_interface.py`) and edit-then-run
drivers.

## Conventions

- Separations `r` in `h⁻¹ Mpc`; `mu = r_parallel / r`.
- `CrossCorr` stores `xi`, `r_array`, `mu_array`, `z_array` on the same
  `(n_r, n_mu)` grid; the `rmu` flag toggles the `(r, mu)` vs
  `(r_transverse, r_parallel)` layout (`switch()` converts).

## Documentation

📖 **Online: <https://corentinravoux.github.io/lyavoid/>**

Built from the in-code docstrings with
[MkDocs Material](https://squidfunk.github.io/mkdocs-material/) +
[mkdocstrings](https://mkdocstrings.github.io/) and deployed to GitHub Pages on
every push to `main`. Build locally with:

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

See [`test/README.md`](test/README.md) for the layout of the suite.
