import xcorr_utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--deltas', default='example_data/deltas/cartesian',
                    help='Folder containing deltas')
parser.add_argument('--tracers', default='example_data/voids/catalog_voids_box_z0_cartesian.fits',
                    help='Path to tracer catalog')
parser.add_argument('--output-corr', default=None,
                    help='Path to output correlation function')
parser.add_argument('--output-mult', default=None,
                    help='Path to output multipoles')
parser.add_argument('--output-mult-plot', default=None,
                    help='Path to output multipoles plot')
parser.add_argument('--z-min-obj', type=float, default = None,
                    help='Mininum z for tracers')
parser.add_argument('--z-max-obj', type=float, default = None,
                    help='Maximum z for tracers')
parser.add_argument('--z-evol-obj', type=float, default = 1.,
                    help='Power for evolution of tracer weight with redshift')
parser.add_argument('--z-evol-del', type=float, default = 2.9,
                    help='Power for evolution of delta weight with redshift')
parser.add_argument('--z-ref', type=float, default=2.25,
                    help='Pivot redshift for evolution of weight with redshift')
parser.add_argument('--mu-min', type=float, default=-1.,
                    help='Min mu')
parser.add_argument('--mu-max', type=float, default=1.,
                    help='Max mu')
parser.add_argument('--nbins-mu', type=int, default=10,
                    help='Number of bins in mu')
parser.add_argument('--r-min', type=float, default=0.,
                    help='Minimum separation in Mpc/h')
parser.add_argument('--r-max', type=float, default=50.,
                    help='Maximum separation in Mpc/h')
parser.add_argument('--nbins-r', type=int, default=50,
                    help='Number of bins in r')
parser.add_argument('--multipole-method', type=str, default='trap',
                    help='Method to compute multipoles')
args = parser.parse_args()


options = vars(args)
for key in options:
    print(key, ':', options[key])

xcf = xcorr_utils.create_xcf(options)

(r, mu, xi) = xcorr_utils.xcorr_cartesian(xcf,save_corr=options["output_corr"])

xcorr_utils.compute_and_plot_multipole_cartesian(mu, r, xi,save_plot=options["output_mult_plot"],
                                                savetxt=options["output_mult"],multipole_method = options["multipole_method"])
