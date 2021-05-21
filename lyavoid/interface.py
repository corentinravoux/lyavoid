import os
import configparser
import ast
from lyavoid import xcorr,multipole


def parse_int_tuple(input):
    if(input == "None"):
        return(None)
    else:
        return tuple(int(k.strip()) for k in input.strip().split(','))

def parse_float_tuple(input):
    if(input == "None"):
        return(None)
    else:
        return tuple(float(k.strip()) for k in input.strip().split(','))

def parse_str_tuple(input):
    if(input == "None"):
        return(None)
    else:
        return tuple(str(k.strip()) for k in input.strip().split(','))

def parse_dict(input):
    if(input == "None"):
        return(None)
    else:
        acceptable_string = input.replace("'", "\"")
        return(ast.literal_eval(acceptable_string))

def parse_float(input):
    if(input == "None"):
        return(None)
    else:
        return(float(input))

def parse_int(input):
    if(input == "None"):
        return(None)
    else:
        return(int(input))

def parse_string(input):
    if(input == "None"):
        return(None)
    else:
        return(str(input))


def main(input_file):
    # CR - stack can be added
    config = configparser.ConfigParser(allow_no_value=True,
                                       converters={"str": parse_string,
                                                   "int": parse_int,
                                                   "float": parse_float,
                                                   "tupleint": parse_int_tuple,
                                                   "tuplefloat": parse_float_tuple,
                                                   "tuplestr": parse_str_tuple,
                                                   "dict":parse_dict})
    config.optionxform = lambda option: option
    config.read(input_file)

    main_config = config["main"]
    main_path = os.path.abspath(main_config["path"])
    os.makedirs(main_path,exist_ok=True)

    xcorr_config = config["xcorr"]
    xcorr_plot_config = config["xcorr plot"]



    xcorr_path = os.path.join(main_path,xcorr_config.getstr("xcorr_path"))
    if(main_config.getboolean("execute_xcorr")):
        execute_xcorr(xcorr_path,xcorr_config,main_config)

    plot_xcorr_path = os.path.join(main_path,xcorr_plot_config.getstr("plot_xcorr_path"))
    if(main_config.getboolean("plot_xcorr")):
        plot_xcorr(plot_xcorr_path,xcorr_path,xcorr_plot_config,xcorr_config,main_config)


def execute_xcorr(xcorr_path,xcorr_config,main_config):
    os.makedirs(xcorr_path,exist_ok=True)

    xcorr_name = os.path.join(xcorr_path,main_config["name"])
    dict_picca_rmu = xcorr_config.getdict("dict_picca_rmu")
    dict_picca_rprt = xcorr_config.getdict("dict_picca_rprt")

    xcorr_dict = {}
    xcorr_dict["drq"] = main_config.getstr("void_catalog")
    xcorr_dict["in-dir"] = main_config.getstr("delta_path")
    xcorr_dict["z-cut-min"] = xcorr_config.getfloat("zmin")
    xcorr_dict["z-cut-max"] = xcorr_config.getfloat("zmax")
    xcorr_dict["z-min-obj"] = xcorr_config.getfloat("zmin")
    xcorr_dict["z-max-obj"] = xcorr_config.getfloat("zmax")

    dict_picca_rmu.update(xcorr_dict)
    dict_picca_rmu["out"] = f"{xcorr_name}.fits"
    dict_picca_rprt.update(xcorr_dict)
    dict_picca_rprt["out"] = f"{xcorr_name}_rprt.fits"

    xcorr.run_cross_corr_picca(dict_picca_rmu,rmu=True)
    xcorr.run_export_picca(f"{xcorr_name}.fits",f"{xcorr_name}_exp.fits",smooth=False)

    xcorr.run_cross_corr_picca(dict_picca_rprt,rmu=False)
    xcorr.run_export_picca(f"{xcorr_name}_rprt.fits",f"{xcorr_name}_rprt_exp.fits",smooth=True)




def plot_xcorr(plot_xcorr_path,xcorr_path,xcorr_plot_config,xcorr_config,main_config):
    os.makedirs(plot_xcorr_path,exist_ok=True)
    xcorr_name = os.path.join(xcorr_path,main_config["name"])
    plot_xcorr_name = os.path.join(plot_xcorr_path,main_config["name"])
    if(xcorr_plot_config.getboolean("plot_multipole")):
        multipole.compute_and_plot_multipole(f"{xcorr_name}_exp.fits",
                                             f"{plot_xcorr_name}_multipoles",
                                             supress_first_pixels=xcorr_plot_config.getint("pixel_supress"),
                                             error_bar=f"{xcorr_name}.fits",
                                             multipole_method=xcorr_plot_config.getstr("multipole_method"),
                                             monopole_division=xcorr_plot_config.getboolean("monopole_division"))
    if(xcorr_plot_config.getboolean("plot_wedge")):
        multipole.compute_and_plot_wedge(f"{xcorr_name}_rprt_exp.fits",
                                         f"{plot_xcorr_name}_wedge_plot")

    if(xcorr_plot_config.getboolean("plot_2d")):
        multipole.plot_2d(f"{xcorr_name}_exp.fits",
                          f"{plot_xcorr_name}_2D_plot",
                          supress_first_pixels=xcorr_plot_config.getint("pixel_supress"),
                          **xcorr_plot_config.getdict("dict_2D"))

    if(xcorr_plot_config.getboolean("plot_comparison")):
        xcorr_comparison_name = xcorr_plot_config.getstr("comparison_name")
        multipole.compute_and_plot_multipole_comparison(f"{xcorr_name}_exp.fits",
                                                        f"{xcorr_comparison_name}_exp.fits",
                                                        f"{plot_xcorr_name}_comparison_multipole",
                                                        supress_first_pixels=xcorr_plot_config.getint("pixel_supress"),
                                                        error_bar=f"{xcorr_name}.fits",
                                                        error_bar2=f"{xcorr_comparison_name}.fits",
                                                        legend=list(xcorr_plot_config.gettuplestr("comparison_legend")),
                                                        multipole_method=xcorr_plot_config.getstr("multipole_method"),
                                                        monopole_division=xcorr_plot_config.getboolean("monopole_division"))
        if(xcorr_plot_config.getboolean("plot_wedge")):
            multipole.compute_and_plot_wedge_comparison(f"{xcorr_name}_rprt_exp.fits",
                                                        f"{xcorr_comparison_name}_rprt_exp.fits",
                                                        f"{plot_xcorr_name}_comparison_wedges",
                                                        legend=list(xcorr_plot_config.gettuplestr("comparison_legend")))
