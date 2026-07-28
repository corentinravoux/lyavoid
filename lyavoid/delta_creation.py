"""Create picca-style delta (flux-contrast) files from tomographic maps.

This module builds artificial lines of sight ("deltas") by sampling a
:class:`lelantos.tomographic_objects.TomographicMap` (or a raw simulated
optical-depth cube) along a grid or random selection of angular/cartesian
positions, and writes them out as picca-compatible FITS delta files. The
resulting deltas are intended to be cross-correlated with void catalogs
(e.g. from `lelantos` or VIDE). Coordinates follow the `lelantos` /
`lyavoid` convention: RA/Dec in radians, comoving distance in h^-1 Mpc.
"""

import os

import fitsio
import numpy as np
from scipy.ndimage import map_coordinates

from lelantos import tomographic_objects, utils


class DeltaGenerator(object):
    """Generate and save picca-style delta files from a tomographic map.

    Loads a :class:`lelantos.tomographic_objects.TomographicMap` and
    samples fake lines of sight from it (through one of several
    coordinate-selection modes), then writes the resulting flux-contrast
    arrays to disk as picca-compatible FITS delta files.

    Attributes:
        los_selection (dict): Line-of-sight selection settings, e.g.
            ``{"method": "equidistant", "rebin": int}``.
        nb_files (int): Number of output delta FITS files to split the
            generated deltas across.
        mode (str): Coordinate/selection mode, one of
            ``"middle"``, ``"cartesian"``, or another value handled by
            :meth:`select_deltas_others` (e.g. ``"distance_redshift"``).
        tomographic_map (lelantos.tomographic_objects.TomographicMap): The
            loaded 3D tomographic map providing ``map_array``, ``shape``,
            ``mpc_per_pixel``, ``boundary_cartesian_coord`` and
            ``Omega_m``.
    """

    def __init__(
        self,
        map_name,
        los_selection,
        nb_files,
        shape=None,
        size=None,
        property_file=None,
        mode="distance_redshift",
    ):
        """Initialize the generator and load the tomographic map.

        Args:
            map_name (str): Path to the tomographic map file to load.
            los_selection (dict): Line-of-sight selection settings (method
                and rebin factor) used by the ``select_deltas_*`` methods.
            nb_files (int): Number of delta FITS files to split the
                output into.
            shape (tuple, optional): Shape of the map grid, forwarded to
                ``TomographicMap.init_classic``. Defaults to None.
            size (tuple, optional): Physical size of the map, forwarded to
                ``TomographicMap.init_classic``. Defaults to None.
            property_file (str, optional): Path to a map property file,
                forwarded to ``TomographicMap.init_classic``. Defaults to
                None.
            mode (str, optional): Coordinate/selection mode used to pick
                which ``select_deltas_*`` method is called. Defaults to
                "distance_redshift".
        """
        self.los_selection = los_selection
        self.nb_files = nb_files
        self.mode = mode

        self.tomographic_map = tomographic_objects.TomographicMap.init_classic(
            name=map_name, shape=shape, size=size, property_file=property_file
        )
        self.tomographic_map.read()

    def select_deltas(self):
        """Dispatch delta selection to the method matching ``self.mode``.

        Returns:
            tuple: ``(deltas_list, deltas_props, redshift_array)`` as
            produced by :meth:`select_deltas_middle`,
            :meth:`select_deltas_cartesian`, or
            :meth:`select_deltas_others` depending on ``self.mode``.
        """
        if self.mode == "middle":
            (deltas_list, deltas_props, redshift_array) = self.select_deltas_middle()
        elif self.mode == "cartesian":
            (deltas_list, deltas_props, redshift_array) = self.select_deltas_cartesian()
        else:
            (deltas_list, deltas_props, redshift_array) = self.select_deltas_others()
        return (deltas_list, deltas_props, redshift_array)

    def select_deltas_middle(self):
        """Select deltas at pixel positions, tagged with the map-middle sky coords.

        Samples the map on a regular pixel-index grid (via
        :meth:`create_index_arrays`), converts the grid coordinates of the
        map's central slice to sky coordinates (via
        :meth:`create_radecz_array_middle`), and builds one delta array
        and property dict per selected line of sight.

        Returns:
            tuple: ``(deltas_list, deltas_props, redshift_array)`` where
            ``deltas_list`` is an ndarray of shape (n_los, n_z) of flux
            contrast values, ``deltas_props`` is a list of per-LOS dicts
            with keys RA, DEC, Z, THING_ID, PLATE, MJD, FIBERID, and
            ``redshift_array`` is the array of redshifts along each LOS.
        """
        map_3d = self.tomographic_map.map_array
        ra_array, dec_array, redshift_array, indices = self.create_radecz_array_middle()

        z_fake_qso = np.max(redshift_array)
        deltas = np.zeros(
            (
                self.tomographic_map.shape[0] // self.los_selection["rebin"],
                self.tomographic_map.shape[1] // self.los_selection["rebin"],
                self.tomographic_map.shape[2],
            )
        )
        deltas[:, :, :] = map_3d[indices[:, :, 0], indices[:, :, 1], :]
        deltas_list = np.array(
            [deltas[i, j, :] for i in range(len(deltas)) for j in range(len(deltas[i]))]
        )
        deltas_props = [
            {
                "RA": ra_array[i],
                "DEC": dec_array[j],
                "Z": z_fake_qso,
                "THING_ID": "1" + "{0:04d}".format(i) + "{0:04d}".format(j),
                "PLATE": "1" + "{0:04d}".format(i) + "{0:04d}".format(j),
                "MJD": "1" + "{0:04d}".format(i) + "{0:04d}".format(j),
                "FIBERID": "1" + "{0:04d}".format(i) + "{0:04d}".format(j),
            }
            for i in range(len(ra_array))
            for j in range(len(dec_array))
        ]

        return (deltas_list, deltas_props, redshift_array)

    def create_radecz_array_middle(self):
        """Build RA/Dec/redshift arrays from the map's central-slice coordinates.

        Converts the pixel-index arrays from :meth:`create_index_arrays`
        into comoving Mpc coordinates, then to sky coordinates (RA, Dec,
        redshift) using the map's cosmology, evaluating the angular
        conversion at the map's mid-redshift.

        Returns:
            tuple: ``(ra_array, dec_array, redshift_array, indices)`` where
            ``ra_array``/``dec_array`` are 1D arrays of sky coordinates
            (radians), ``redshift_array`` is the 1D array of line-of-sight
            redshifts, and ``indices`` is the meshgrid of pixel-index
            pairs (x, y) with shape (len(x_array), len(y_array), 2).
        """
        (x_array, y_array, z_array) = self.create_index_arrays()
        indices = np.moveaxis(
            np.array(np.meshgrid(x_array, y_array, indexing="ij")), 0, -1
        )

        minx, miny, minz = self.tomographic_map.boundary_cartesian_coord[0]
        maxx, maxy, maxz = self.tomographic_map.boundary_cartesian_coord[1]
        x_array_mpc = x_array * self.tomographic_map.mpc_per_pixel[0] + minx
        y_array_mpc = y_array * self.tomographic_map.mpc_per_pixel[1] + miny
        z_array_mpc = z_array * self.tomographic_map.mpc_per_pixel[2] + minz

        (rcomov, distang, inv_rcomov, inv_distang) = utils.get_cosmo_function(
            self.tomographic_map.Omega_m
        )
        conversion_redshift = utils.convert_z_cartesian_to_sky_middle(
            np.array([minz, maxz]), inv_rcomov
        )
        minredshift, maxredshift = conversion_redshift[0], conversion_redshift[1]
        suplementary_parameters = utils.return_suplementary_parameters(
            self.mode, zmin=minredshift, zmax=maxredshift
        )
        ra_array, dec_array, redshift_array = utils.convert_cartesian_to_sky(
            x_array_mpc,
            y_array_mpc,
            z_array_mpc,
            self.mode,
            inv_rcomov=inv_rcomov,
            inv_distang=inv_distang,
            distang=distang,
            suplementary_parameters=suplementary_parameters,
        )
        return (ra_array, dec_array, redshift_array, indices)

    def create_index_arrays(self):
        """Build pixel-index arrays selecting lines of sight on the map grid.

        Uses ``self.los_selection["method"]`` and
        ``self.los_selection["rebin"]`` to pick evenly spaced x and y
        pixel indices across the map, and the full range of z indices.

        Returns:
            tuple: ``(x_array, y_array, z_array)`` of integer pixel-index
            ndarrays for the x, y, and z (redshift) axes of the map.
        """
        shape_map = self.tomographic_map.shape
        if self.los_selection["method"] == "equidistant":
            x_array = np.round(
                np.linspace(
                    0,
                    shape_map[0] - 1,
                    int(shape_map[0] // self.los_selection["rebin"]),
                ),
                0,
            ).astype(int)
            y_array = np.round(
                np.linspace(
                    0,
                    shape_map[1] - 1,
                    int(shape_map[1] // self.los_selection["rebin"]),
                ),
                0,
            ).astype(int)
            z_array = np.arange(0, shape_map[2])
        return (x_array, y_array, z_array)

    def select_deltas_others(self):
        """Select deltas by interpolating the map on a regular sky-coordinate grid.

        Builds RA/Dec/redshift sample points (via
        :meth:`create_radecz_array`), converts them to cartesian pixel
        coordinates, and interpolates the map at those positions with
        :func:`scipy.ndimage.map_coordinates` (linear order). This is the
        fallback used when ``self.mode`` is neither ``"middle"`` nor
        ``"cartesian"``.

        Returns:
            tuple: ``(deltas, deltas_props, redshift_array)`` where
            ``deltas`` is an ndarray of shape (n_los, n_z) of interpolated
            flux contrast values, ``deltas_props`` is a list of per-LOS
            dicts with keys RA, DEC, Z, THING_ID, PLATE, MJD, FIBERID, and
            ``redshift_array`` is the array of redshifts along each LOS.
        """
        map_3d = self.tomographic_map.map_array
        ra_array, dec_array, redshift_array = self.create_radecz_array()

        (rcomov, distang, inv_rcomov, inv_distang) = utils.get_cosmo_function(
            self.tomographic_map.Omega_m
        )
        indice = np.moveaxis(
            np.array(np.meshgrid(ra_array, dec_array, redshift_array, indexing="ij")),
            0,
            -1,
        )
        indice_box = np.zeros(indice.shape)
        (
            indice_box[:, :, :, 0],
            indice_box[:, :, :, 1],
            indice_box[:, :, :, 2],
        ) = utils.convert_sky_to_cartesian(
            ra_array,
            dec_array,
            redshift_array,
            self.mode,
            inv_rcomov=inv_rcomov,
            inv_distang=inv_distang,
            distang=distang,
            suplementary_parameters=None,
        )
        indice_box = indice_box / self.tomographic_map.mpc_per_pixel
        indice_box = indice_box.reshape(
            indice_box.shape[0] * indice_box.shape[1] * indice_box.shape[2],
            indice_box.shape[3],
        )

        z_fake_qso = np.max(redshift_array)
        deltas = map_coordinates(map_3d, np.transpose(indice_box), order=1).reshape(
            (indice.shape[0], indice.shape[1], indice.shape[2])
        )
        deltas = deltas.reshape(deltas.shape[0] * deltas.shape[1], deltas.shape[2])
        deltas_props = [
            {
                "RA": ra_array[i],
                "DEC": dec_array[j],
                "Z": z_fake_qso,
                "THING_ID": "1" + "{0:04d}".format(i) + "{0:04d}".format(j),
                "PLATE": "1" + "{0:04d}".format(i) + "{0:04d}".format(j),
                "MJD": "1" + "{0:04d}".format(i) + "{0:04d}".format(j),
                "FIBERID": "1" + "{0:04d}".format(i) + "{0:04d}".format(j),
            }
            for i in range(len(ra_array))
            for j in range(len(dec_array))
        ]
        return (deltas, deltas_props, redshift_array)

    def create_radecz_array(self):
        """Build RA/Dec/redshift sample arrays spanning the full map volume.

        Converts every map pixel's cartesian coordinates to sky
        coordinates to determine the RA/Dec/redshift ranges covered by
        the map, then (for the ``"equidistant"`` selection method) builds
        evenly spaced RA and Dec arrays (per
        ``self.los_selection["rebin"]``) and a redshift array spanning
        the full z-axis resolution.

        Returns:
            tuple: ``(ra_array, dec_array, redshift_array)`` 1D ndarrays
            of sky coordinates (RA/Dec in radians) and redshifts.
        """
        minx, miny, minz = self.tomographic_map.boundary_cartesian_coord[0]
        maxx, maxy, maxz = self.tomographic_map.boundary_cartesian_coord[1]

        indice = (
            np.transpose(np.indices(self.tomographic_map.shape), axes=(1, 2, 3, 0))
            * self.tomographic_map.mpc_per_pixel
        )
        indice = indice + [minx, miny, minz]
        (rcomov, distang, inv_rcomov, inv_distang) = utils.get_cosmo_function(
            self.tomographic_map.Omega_m
        )
        (
            indice[:, :, :, 0],
            indice[:, :, :, 1],
            indice[:, :, :, 2],
        ) = utils.convert_cartesian_to_sky(
            indice[:, :, :, 0],
            indice[:, :, :, 1],
            indice[:, :, :, 2],
            self.mode,
            inv_rcomov=inv_rcomov,
            inv_distang=inv_distang,
            distang=distang,
            suplementary_parameters=None,
        )

        ramin, ramax = np.min(indice[:, :, :, 0]), np.max(indice[:, :, :, 0])
        decmin, decmax = np.min(indice[:, :, :, 1]), np.max(indice[:, :, :, 1])
        redshiftmin, redshiftmax = np.min(indice[:, :, :, 2]), np.max(
            indice[:, :, :, 2]
        )
        if self.los_selection["method"] == "equidistant":
            ra_array = np.linspace(
                ramin,
                ramax,
                self.tomographic_map.shape[0] // self.los_selection["rebin"],
            )
            dec_array = np.linspace(
                decmin,
                decmax,
                self.tomographic_map.shape[1] // self.los_selection["rebin"],
            )
            redshift_array = np.linspace(
                redshiftmin, redshiftmax, self.tomographic_map.shape[2]
            )
        return (ra_array, dec_array, redshift_array)

    def select_deltas_cartesian(self):
        """Select deltas at pixel positions, tagged with cartesian coordinates.

        Samples the map on a regular pixel-index grid (via
        :meth:`create_index_arrays_cartesian`) and builds one delta array
        and property dict (using cartesian X/Y/Z rather than sky
        coordinates) per selected line of sight.

        Returns:
            tuple: ``(deltas_list, deltas_props, redshift_array)`` where
            ``deltas_list`` is an ndarray of shape (n_los, n_z) of flux
            contrast values, ``deltas_props`` is a list of per-LOS dicts
            with keys X, Y, Z, ZQSO, THING_ID, PLATE, MJD, FIBERID, and
            ``redshift_array`` is the array of redshifts along each LOS.
        """
        map_3d = self.tomographic_map.map_array
        (
            x_array,
            y_array,
            z_array,
            redshift_array,
            indices,
        ) = self.create_index_arrays_cartesian()

        z_fake_qso = np.max(redshift_array)
        deltas = np.zeros(
            (
                self.tomographic_map.shape[0] // self.los_selection["rebin"],
                self.tomographic_map.shape[1] // self.los_selection["rebin"],
                self.tomographic_map.shape[2],
            )
        )
        deltas[:, :, :] = map_3d[indices[:, :, 0], indices[:, :, 1], :]
        deltas_list = np.array(
            [deltas[i, j, :] for i in range(len(deltas)) for j in range(len(deltas[i]))]
        )
        deltas_props = [
            {
                "X": x_array[i],
                "Y": y_array[j],
                "Z": z_array,
                "ZQSO": z_fake_qso,
                "THING_ID": "1" + "{0:04d}".format(i) + "{0:04d}".format(j),
                "PLATE": "1" + "{0:04d}".format(i) + "{0:04d}".format(j),
                "MJD": "1" + "{0:04d}".format(i) + "{0:04d}".format(j),
                "FIBERID": "1" + "{0:04d}".format(i) + "{0:04d}".format(j),
            }
            for i in range(len(x_array))
            for j in range(len(y_array))
        ]

        return (deltas_list, deltas_props, redshift_array)

    def create_index_arrays_cartesian(self, method="mean"):
        """Build cartesian-Mpc coordinate arrays selecting lines of sight on the map grid.

        Uses ``self.los_selection["method"]`` and
        ``self.los_selection["rebin"]`` to pick evenly spaced x and y
        pixel indices across the map, converts them (plus the full z
        pixel range) to comoving Mpc offsets from the map's boundary, and
        also derives the corresponding redshift along the line of sight.

        Args:
            method (str, optional): Unused placeholder parameter. Defaults
                to "mean".

        Returns:
            tuple: ``(x_array_mpc, y_array_mpc, z_array_mpc,
            redshift_array, indices)`` where the first three are 1D
            ndarrays of comoving Mpc coordinates, ``redshift_array`` is
            the redshift along the z axis, and ``indices`` is the
            meshgrid of pixel-index pairs (x, y).
        """
        if self.los_selection["method"] == "equidistant":
            x_array = np.round(
                np.linspace(
                    0,
                    self.tomographic_map.shape[0] - 1,
                    int(self.tomographic_map.shape[0] // self.los_selection["rebin"]),
                ),
                0,
            ).astype(int)
            y_array = np.round(
                np.linspace(
                    0,
                    self.tomographic_map.shape[1] - 1,
                    int(self.tomographic_map.shape[1] // self.los_selection["rebin"]),
                ),
                0,
            ).astype(int)
            z_array = np.arange(0, self.tomographic_map.shape[2])
        indices = np.moveaxis(
            np.array(np.meshgrid(x_array, y_array, indexing="ij")), 0, -1
        )

        minx, miny, minz = self.tomographic_map.boundary_cartesian_coord[0]
        mpc_per_pixel = self.tomographic_map.mpc_per_pixel
        x_array_mpc, y_array_mpc, z_array_mpc = (
            x_array * mpc_per_pixel[0] + minx,
            y_array * mpc_per_pixel[1] + miny,
            z_array * mpc_per_pixel[2] + minz,
        )
        (rcomov, distang, inv_rcomov, inv_distang) = utils.get_cosmo_function(
            self.tomographic_map.Omega_m
        )
        redshift_array = utils.convert_z_cartesian_to_sky_middle(
            z_array_mpc, inv_rcomov
        )
        return (x_array_mpc, y_array_mpc, z_array_mpc, redshift_array, indices)

    # CR - change delta saving with connection to lelantos

    def save_deltas(self, deltas_list, deltas_props, redshift_array, path_out):
        """Split deltas into ``self.nb_files`` chunks and write each to a FITS file.

        Distributes the lines of sight in ``deltas_list`` evenly across
        ``self.nb_files`` output files named ``delta-{i}.fits`` under
        ``path_out``, delegating each chunk's writing to
        :meth:`create_a_delta_file_cartesian` (if ``self.mode ==
        "cartesian"``) or :meth:`create_a_delta_file` otherwise. Any
        remainder lines of sight are appended to the last file.

        Args:
            deltas_list (numpy.ndarray): Array of shape (n_los, n_z) of
                flux contrast values for every line of sight.
            deltas_props (list): List of per-LOS property dicts, one per
                row of ``deltas_list``.
            redshift_array (numpy.ndarray): Redshifts along each line of
                sight, shared by all deltas.
            path_out (str): Output directory where delta FITS files are
                written.
        """
        nb_deltas = len(deltas_list)
        nb_deltas_per_files = nb_deltas // (self.nb_files - 1)
        name_delta = os.path.join(path_out, "delta-{}.fits")
        for i in range(self.nb_files - 1):
            delta = deltas_list[
                i * nb_deltas_per_files : (i + 1) * nb_deltas_per_files, :
            ]
            delta_props = deltas_props[
                i * nb_deltas_per_files : (i + 1) * nb_deltas_per_files
            ]
            if self.mode == "cartesian":
                self.create_a_delta_file_cartesian(
                    delta, delta_props, name_delta.format(i), redshift_array
                )
            else:
                self.create_a_delta_file(
                    delta, delta_props, name_delta.format(i), redshift_array
                )
        delta = deltas_list[(i + 1) * nb_deltas_per_files : :, :]
        delta_props = deltas_props[(i + 1) * nb_deltas_per_files : :]
        if self.mode == "cartesian":
            self.create_a_delta_file_cartesian(
                delta, delta_props, name_delta.format(i + 1), redshift_array
            )
        else:
            self.create_a_delta_file(
                delta, delta_props, name_delta.format(i + 1), redshift_array
            )

    def create_a_delta_file(
        self, delta, delta_props, name_out, redshift_array, weight=1.0, cont=1.0
    ):
        """Write a picca-style delta FITS file with sky-coordinate headers.

        Each row of ``delta`` becomes one FITS HDU containing LOGLAM,
        DELTA, WEIGHT and CONT columns, with a header carrying THING_ID,
        RA (wrapped to [0, 2*pi)), DEC, Z, PLATE, MJD and FIBERID from
        the matching entry of ``delta_props``.

        Args:
            delta (numpy.ndarray): Array of shape (n_los, n_z) of flux
                contrast values to write.
            delta_props (list): List of per-LOS property dicts (RA, DEC,
                Z, THING_ID, PLATE, MJD, FIBERID), one per row of
                ``delta``.
            name_out (str): Output FITS file path.
            redshift_array (numpy.ndarray): Redshifts along the line of
                sight, used to compute LOGLAM via the Lyman-alpha
                wavelength.
            weight (float, optional): Constant weight assigned to every
                pixel. Defaults to 1.0.
            cont (float, optional): Constant continuum value assigned to
                every pixel. Defaults to 1.0.
        """
        new_delta = fitsio.FITS(name_out, "rw", clobber=True)
        loglambda = np.log10((1 + redshift_array) * utils.lambdaLy)
        for j in range(len(delta)):
            nrows = len(delta[j])
            h = np.zeros(
                nrows,
                dtype=[
                    ("LOGLAM", "f8"),
                    ("DELTA", "f8"),
                    ("WEIGHT", "f8"),
                    ("CONT", "f8"),
                ],
            )
            h["DELTA"] = delta[j][:]
            h["LOGLAM"] = loglambda
            h["WEIGHT"] = np.array([weight for i in range(nrows)])
            h["CONT"] = np.array([cont for i in range(nrows)])
            head = {}
            head["THING_ID"] = delta_props[j]["THING_ID"]
            if delta_props[j]["RA"] >= 0:
                head["RA"] = delta_props[j]["RA"]
            else:
                head["RA"] = delta_props[j]["RA"] + 2 * np.pi
            head["DEC"] = delta_props[j]["DEC"]
            head["Z"] = delta_props[j]["Z"]
            head["PLATE"] = delta_props[j]["PLATE"]
            head["MJD"] = delta_props[j]["MJD"]
            head["FIBERID"] = delta_props[j]["FIBERID"]
            new_delta.write(h, extname=delta_props[j]["THING_ID"], header=head)
        new_delta.close()

    def create_a_delta_file_cartesian(
        self, delta, delta_props, name_out, redshift_array, weight=1.0, cont=1.0
    ):
        """Write a picca-style delta FITS file with cartesian-coordinate headers.

        Each row of ``delta`` becomes one FITS HDU containing LOGLAM,
        DELTA, WEIGHT and Z columns, with a header carrying X, Y, ZQSO,
        PLATE, MJD, FIBERID and THING_ID from the matching entry of
        ``delta_props``.

        Args:
            delta (numpy.ndarray): Array of shape (n_los, n_z) of flux
                contrast values to write.
            delta_props (list): List of per-LOS property dicts (X, Y, Z,
                ZQSO, THING_ID, PLATE, MJD, FIBERID), one per row of
                ``delta``.
            name_out (str): Output FITS file path.
            redshift_array (numpy.ndarray): Redshifts along the line of
                sight, used to compute LOGLAM via the Lyman-alpha
                wavelength.
            weight (float, optional): Constant weight assigned to every
                pixel. Defaults to 1.0.
            cont (float, optional): Unused; kept for signature symmetry
                with :meth:`create_a_delta_file`. Defaults to 1.0.
        """
        new_delta = fitsio.FITS(name_out, "rw", clobber=True)
        loglambda = np.log10((1 + redshift_array) * utils.lambdaLy)
        for j in range(len(delta)):
            nrows = len(delta[j])
            h = np.zeros(
                nrows,
                dtype=[
                    ("LOGLAM", "f8"),
                    ("DELTA", "f8"),
                    ("WEIGHT", "f8"),
                    ("Z", "f8"),
                ],
            )
            h["DELTA"] = delta[j][:]
            h["LOGLAM"] = loglambda
            h["WEIGHT"] = np.array([weight for i in range(nrows)])
            h["Z"] = delta_props[j]["Z"]
            head = {}
            head["X"] = delta_props[j]["X"]
            head["Y"] = delta_props[j]["Y"]
            head["ZQSO"] = delta_props[j]["ZQSO"]
            head["PLATE"] = delta_props[j]["PLATE"]
            head["MJD"] = delta_props[j]["MJD"]
            head["FIBERID"] = delta_props[j]["FIBERID"]
            head["THING_ID"] = delta_props[j]["THING_ID"]
            new_delta.write(h, extname=delta_props[j]["THING_ID"], header=head)
        new_delta.close()

    def create_deltas_from_cube(self, path_out):
        """Select deltas from the tomographic map and save them to disk.

        Convenience wrapper that calls :meth:`select_deltas`, creates
        ``path_out`` if needed, and writes the resulting deltas via
        :meth:`save_deltas`.

        Args:
            path_out (str): Output directory for the generated delta FITS
                files (created if it does not exist).
        """
        (deltas_list, deltas_props, redshift_array) = self.select_deltas()
        os.makedirs(path_out, exist_ok=True)
        self.save_deltas(deltas_list, deltas_props, redshift_array, path_out)


def create_index_arrays_cartesian(
    shape,
    size,
    method="equidistant",
    rebin=1,
    n_los=1000,
):
    """Build cartesian-Mpc coordinate arrays selecting lines of sight on a raw cube.

    Standalone counterpart of
    ``DeltaGenerator.create_index_arrays_cartesian`` for use with a raw
    (x, y, z)-shaped array (e.g. a simulated optical-depth cube) rather
    than a `TomographicMap`. Supports either an evenly spaced
    ("equidistant") grid of x/y pixel indices, or a ("random") uniform
    draw of ``n_los`` sight lines, and converts the selected indices to
    comoving Mpc offsets.

    Args:
        shape (tuple): Shape (nx, ny, nz) of the cube.
        size (tuple): Physical size of the cube, used with ``shape`` to
            compute the Mpc-per-pixel scale via ``utils.mpc_per_pixel``.
        method (str, optional): Selection method, "equidistant" or
            "random". Defaults to "equidistant".
        rebin (int, optional): Rebinning factor for the "equidistant"
            method (selects roughly ``shape // rebin`` sight lines per
            axis). Defaults to 1.
        n_los (int, optional): Number of sight lines to draw for the
            "random" method. Defaults to 1000.

    Returns:
        tuple: ``(x_array_mpc, y_array_mpc, z_array_mpc, indices)`` where
        the first three are ndarrays of comoving Mpc coordinates (x and y
        per selected line of sight, z along the full cube depth), and
        ``indices`` is an ndarray of shape (n_los, 2) of the selected
        (x, y) pixel indices.
    """
    if method == "equidistant":
        x_array = np.round(
            np.linspace(
                0,
                shape[0] - 1,
                int(shape[0] // rebin),
            ),
            0,
        ).astype(int)
        y_array = np.round(
            np.linspace(
                0,
                shape[1] - 1,
                int(shape[1] // rebin),
            ),
            0,
        ).astype(int)

        indices = np.moveaxis(
            np.array(np.meshgrid(x_array, y_array, indexing="ij")), 0, -1
        )
        indices = indices.reshape(indices.shape[0] * indices.shape[1], indices.shape[2])

    elif method == "random":
        x_array = np.random.choice(
            np.linspace(0, shape[0] - 1, shape[0]),
            size=n_los,
        ).astype(int)
        y_array = np.random.choice(
            np.linspace(0, shape[1] - 1, shape[1]),
            size=n_los,
        ).astype(int)
        indices = np.transpose(np.stack([x_array, y_array]))

    z_array = np.arange(0, shape[2])
    mpc_per_pixel = utils.mpc_per_pixel(size, shape)

    x_array_mpc, y_array_mpc, z_array_mpc = (
        indices[:, 0] * mpc_per_pixel[0],
        indices[:, 1] * mpc_per_pixel[1],
        z_array * mpc_per_pixel[2],
    )

    return (x_array_mpc, y_array_mpc, z_array_mpc, indices)


def create_deltas_from_simulation(
    tau_red,
    shape,
    size,
    method="equidistant",
    rebin=1,
    n_los=1000,
    rebin_z=None,
    method_rebin_z="mean",
):
    """Build fake delta lines of sight directly from a simulated optical-depth cube.

    Selects lines of sight from ``tau_red`` using
    :func:`create_index_arrays_cartesian`, converts optical depth to flux
    (``F = exp(-tau_red)``) and then to flux contrast
    (``delta = F / mean(F) - 1``) along each line of sight, optionally
    rebinning along the redshift axis, and packages the results as a
    list of per-LOS dictionaries (cartesian coordinates, not sky
    coordinates or FITS files).

    Args:
        tau_red (numpy.ndarray): Simulated redshift-space optical depth
            cube of shape matching ``shape``.
        shape (tuple): Shape (nx, ny, nz) of ``tau_red``.
        size (tuple): Physical size of the cube, used to compute the
            Mpc-per-pixel scale.
        method (str, optional): Line-of-sight selection method,
            "equidistant" or "random", forwarded to
            :func:`create_index_arrays_cartesian`. Defaults to
            "equidistant".
        rebin (int, optional): Rebinning factor for the "equidistant"
            method. Defaults to 1.
        n_los (int, optional): Number of sight lines for the "random"
            method. Defaults to 1000.
        rebin_z (int, optional): If given, rebin each line of sight (and
            the z coordinate array) along the redshift axis by this
            factor using ``method_rebin_z``. Defaults to None (no
            rebinning).
        method_rebin_z (str, optional): Reduction operation passed to
            ``utils.bin_ndarray`` when rebinning along z (e.g. "mean").
            Defaults to "mean".

    Returns:
        list: List of dicts, one per line of sight, each with keys
        "index" (int), "x", "y" (float, comoving Mpc), "z" (ndarray of
        comoving Mpc positions along the LOS), "delta" (ndarray of flux
        contrast values), and "weights" (ndarray of ones, same shape as
        "delta").
    """
    (x_array_mpc, y_array_mpc, z_array_mpc, indices) = create_index_arrays_cartesian(
        shape, size, method=method, rebin=rebin, n_los=n_los
    )

    if rebin_z is not None:
        deltas = np.zeros(
            (
                indices.shape[0],
                shape[2] // rebin_z,
            )
        )
    else:
        deltas = np.zeros(
            (
                indices.shape[0],
                shape[2],
            )
        )

    for i in range(len(deltas)):
        F = np.exp(-tau_red[indices[i, 0], indices[i, 1], :])
        delta = F / np.mean(F) - 1
        if rebin_z is not None:
            deltas[i, :] = utils.bin_ndarray(
                delta, (F.size // rebin_z,), operation=method_rebin_z
            )
        else:
            deltas[i, :] = delta

    if rebin_z is not None:
        z_array_mpc = utils.bin_ndarray(
            z_array_mpc, (z_array_mpc.size // rebin_z,), operation=method_rebin_z
        )

    delta_dicts = []
    for i in range(len(deltas)):
        delta_dict = {}
        delta_dict["index"] = i
        delta_dict["x"] = x_array_mpc[i]
        delta_dict["y"] = y_array_mpc[i]
        delta_dict["z"] = z_array_mpc
        delta_dict["delta"] = deltas[i]
        delta_dict["weights"] = np.full(deltas[i].shape, 1)
        delta_dicts.append(delta_dict)

    return delta_dicts
