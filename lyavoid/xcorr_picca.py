import numpy as np
lambdaLy = 1215.673123130217

from picca import io
from picca import constants
from picca import xcf
from picca import utils




def picca_create_xcf(delta_path,void_catalog,Omega_m):
    cosmo = constants.cosmo(Om=Omega_m)
    lambda_abs = constants.absorber_IGM["LYA"]
    nside = 16
    z_evol_del = 2.9
    z_ref = 2.25
    nspec = None
    no_project = False
    from_image = None
    z_min_obj = None
    z_max_obj = None
    z_evol_obj = 1.
    rt_max = 50.
    rt_min = 0.
    rp_min = -50.
    rp_max = 50.
    npb = 30
    ntb = 15

    mu_max = 1.
    mu_min = -1.
    r_min =0.
    r_max = 50.
    nmu = 30
    nr = 15


    z_cut_max = 10.
    z_cut_min = 0.
    rmu = True

    dels, ndels, zmin_pix, zmax_pix = io.read_deltas(delta_path, nside, lambda_abs, z_evol_del, z_ref, cosmo=cosmo,nspec=nspec,no_project=no_project,from_image=from_image)
    objs,zmin_obj = io.read_objects(void_catalog, nside, z_min_obj, z_max_obj,z_evol_obj, z_ref,cosmo)


    pix = dels.keys()
    if(rmu):
        xcf.rt_max = mu_max
        xcf.rt_min = mu_min
        xcf.rp_min = r_min
        xcf.rp_max = r_max
        xcf.npb = nr
        xcf.ntb = nmu
    else:
        xcf.rt_max = rt_max
        xcf.rt_min = rt_min
        xcf.rp_min = rp_min
        xcf.rp_max = rp_max
        xcf.npb = npb
        xcf.ntb = ntb
    xcf.rmu = rmu
    xcf.z_cut_max = z_cut_max
    xcf.z_cut_min = z_cut_min
    xcf.nside = nside
    xcf.dels = dels
    xcf.npix = len(dels)
    xcf.ndels = ndels
    xcf.objs = objs
    xcf.angmax = utils.compute_ang_max(cosmo, rt_max, zmin_pix, zmin_obj)
    xcf.fill_neighs(pix)
    return(xcf,cosmo)


def picca_xcorr_cartesian(xcf,cosmo,supress_first_pixels=0):
    npb = xcf.npb
    ntb = xcf.ntb
    xi = np.zeros(npb*ntb)
    we = np.zeros(npb*ntb)
    rp = np.zeros(npb*ntb)
    rt = np.zeros(npb*ntb)
    z = np.zeros(npb*ntb)
    nb = np.zeros(npb*ntb,dtype=np.int64)
    dels = xcf.dels
    for pix in dels.keys():
        for d in dels[pix]:
            if (d.qneighs.size != 0):
                ang = d^d.qneighs
                zqso = [q.zqso for q in d.qneighs]
                weights_qso = [q.we for q in d.qneighs]
                r_comov_qso = [q.r_comov for q in d.qneighs]
                dist_m_qso = [q.rdm_comov for q in d.qneighs]
                (rebin_weight, rebin_xi, rebin_r_par, rebin_r_trans,rebin_z, rebin_num_pairs) =  picca_fast_xcf(xcf,d.z,d.r_comov,d.rdm_comov,d.we,d.de,zqso,r_comov_qso,dist_m_qso,weights_qso,ang)
                xi[:len(rebin_xi)]+=rebin_xi
                we[:len(rebin_weight)]+=rebin_weight
                rp[:len(rebin_r_par)]+=rebin_r_par
                rt[:len(rebin_r_trans)]+=rebin_r_trans
                z[:len(rebin_z)]+=rebin_z
                nb[:len(rebin_num_pairs)]+=rebin_num_pairs.astype(int)
    w = we>0
    xi[w]/=we[w]
    rp[w]/=we[w]
    rt[w]/=we[w]
    z[w]/=we[w]

    nr = xcf.npb
    nmu = xcf.ntb
    da = xi
    r = rp
    mu = rt
    r = r.reshape(nr, nmu)[supress_first_pixels:,:]
    mu = mu.reshape(nr, nmu)[supress_first_pixels:,:]
    da = da.reshape(nr, nmu)[supress_first_pixels:,:]
    return(r,mu,da)


def picca_fast_xcf(xcf,z1,r1,rdm1,w1,d1,z2,r2,rdm2,w2,ang):

    if xcf.ang_correlation:
        rp = r1[:,None]/r2
        rt = ang*np.ones_like(rp)
    else:
        rp = (r1[:,None]-r2)*np.cos(ang/2)
        rt = (rdm1[:,None]+rdm2)*np.sin(ang/2)

    if xcf.rmu:
        r = np.sqrt(rp**2+rt**2)
        rt = rp/r
        rp = r

    z = (z1[:,None]+z2)/2

    we = w1[:,None]*w2
    wde = (w1*d1)[:,None]*w2

    w = (rp>xcf.rp_min) & (rp<xcf.rp_max) & (rt>xcf.rt_min) & (rt<xcf.rt_max)
    rp = rp[w]
    rt = rt[w]
    z  = z[w]
    we = we[w]
    wde = wde[w]

    bp = ((rp-xcf.rp_min)/(xcf.rp_max-xcf.rp_min)*xcf.npb).astype(int)
    bt = ((rt-xcf.rt_min)/(xcf.rt_max-xcf.rt_min)*xcf.ntb).astype(int)
    bins = bt + xcf.ntb*bp

    rebin_xi = np.bincount(bins,weights=wde)
    rebin_weight = np.bincount(bins,weights=we)
    rebin_r_par = np.bincount(bins,weights=rp*we)
    rebin_r_trans = np.bincount(bins,weights=rt*we)
    rebin_z = np.bincount(bins,weights=z*we)
    rebin_num_pairs = np.bincount(bins,weights=(we>0.))

    return (rebin_weight, rebin_xi, rebin_r_par, rebin_r_trans,rebin_z, rebin_num_pairs)
