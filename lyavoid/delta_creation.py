import os

import fitsio
import numpy as np
from scipy.ndimage import map_coordinates

from lelantos import tomographic_objects, utils


class DeltaGenerator(object):
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
        self.los_selection = los_selection
        self.nb_files = nb_files
        self.mode = mode

        self.tomographic_map = tomographic_objects.TomographicMap.init_classic(
            name=map_name, shape=shape, size=size, property_file=property_file
        )
        self.tomographic_map.read()

    def select_deltas(self):
        if self.mode == "middle":
            (deltas_list, deltas_props, redshift_array) = self.select_deltas_middle()
        elif self.mode == "cartesian":
            (deltas_list, deltas_props, redshift_array) = self.select_deltas_cartesian()
        else:
            (deltas_list, deltas_props, redshift_array) = self.select_deltas_others()
        return (deltas_list, deltas_props, redshift_array)

    def select_deltas_middle(self):
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
        (deltas_list, deltas_props, redshift_array) = self.select_deltas()
        os.makedirs(path_out, exist_ok=True)
        self.save_deltas(deltas_list, deltas_props, redshift_array, path_out)
