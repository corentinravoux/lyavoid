# Getting started

## Installation

```bash
git clone https://github.com/corentinravoux/lyavoid.git
cd lyavoid
pip install .            # or: pip install -e ".[dev]" for tests
```

Core dependencies (`fitsio`, `numpy`, `scipy`, `matplotlib`) install
automatically. lyavoid also uses:

- [picca](https://github.com/igmhub/picca) — the cross-correlation and export
  back-end used by [`xcorr`](api/xcorr.md) / [`multipole`](api/multipole.md);
- [lelantos](https://github.com/corentinravoux/lelantos) — provides the void
  catalog and tomographic-map objects used by [`xcorr`](api/xcorr.md) and
  [`delta_creation`](api/delta_creation.md);
- `numba` — the JIT-compiled pair loops in [`autocorr`](api/autocorr.md);
- `iminuit` — the model fitting in [`model`](api/model.md).

## Working with a cross-correlation

The central object is [`xcorr_objects.CrossCorr`](api/xcorr_objects.md), a 2-D
correlation on an `(r, mu)` grid:

```python
from lyavoid import xcorr_objects

xcf = xcorr_objects.CrossCorr.init_from_fits("xcorr_exp.fits")
r   = xcf.r_array      # separations, h^-1 Mpc
mu  = xcf.mu_array     # mu = r_parallel / r
xi  = xcf.xi_array     # correlation values

xcf.plot_2d(name="xcorr_2d", r_power=2)
```

## Computing multipoles

```python
from lyavoid import xcorr_objects

mult = xcorr_objects.Multipole.init_from_xcorr(
    ell=[0, 2, 4], mu=xcf.mu_array, xi=xcf.xi_array,
    method="rect", r_array=xcf.r_array, name="multipoles.fits",
)
monopole   = mult.poles[0]
quadrupole = mult.poles[2]
mult.write_fits()
```

Available integration `method`s: `rect`, `nbody`, `simps`, `trap`,
`rebin_simps`, `rebin_trap`, `quad`, `romb`.

## Running the full pipeline

The [`interface`](api/interface.md) module drives the computation and plotting
from a single `.ini` file:

```python
from lyavoid import interface

interface.main("config.ini")
```

or from the command line:

```bash
lyavoid_interface.py config.ini
```

The config's `[main]` section points at the void catalog and delta directory,
`[xcorr]` sets the binning and redshift cuts, and `[xcorr plot]` selects the
multipole / wedge / 2-D / comparison plots.
