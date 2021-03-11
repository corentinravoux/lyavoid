import fitsio
import numpy as np
import glob,os
from lyavoid import xcorr_objects
from lslyatomo import tomographic_objects
lambdaLy = 1215.673123130217
import subprocess


def run_cross_corr_picca(dict_picca,rmu=True):
    cmd = " picca_xcf.py"
    for key,value in dict_picca.items():
        cmd += f" --{key} {value}"
    if(rmu):
        cmd += " --rmu"
    subprocess.call(cmd, shell=True)

def run_export_picca(input,output,smooth=False):
    cmd = " picca_export.py"
    cmd += f" --data {input}"
    cmd += f" --out {output}"
    if(not(smooth)):
        cmd += " --do-not-smooth-cov"
    subprocess.call(cmd, shell=True)



def create_xcf(delta_path,void_catalog,xcorr_options):
    xcf = xcorr_options
    read_voids(xcf,void_catalog)
    read_deltas(xcf,delta_path)
    fill_neighs(xcf)
    return(xcf)


def read_voids(xcf,void_catalog):
    void = tomographic_objects.VoidCatalog.init_from_fits(void_catalog)
    if(xcf["z_max_obj"] is not None):
        void.cut_catalog(coord_min=(-np.inf,-np.inf,xcf["z_min_obj"]))
    if(xcf["z_max_obj"] is not None):
        void.cut_catalog(coord_max=(np.inf,np.inf,xcf["z_max_obj"]))
    void_coords = void.compute_cross_corr_parameters()
    void_coords[:,3] = void_coords[:,3] * (((1 + void_coords[:,4])/(1+xcf["z_ref"]))**(xcf["z_evol_obj"]-1))
    xcf["voids"]=void_coords
    print('..done. Number of voids:', void_coords.shape[0])



def read_voids_old(xcf,void_catalog):
    """obsolete"""
    voids = fitsio.FITS(void_catalog)[1]
    if xcf["weights_voids"] is None :
        void_coords = np.transpose(np.stack([voids["X"][:],voids["Y"][:],voids["Z"][:],voids["WEIGHT"][:],voids["REDSHIFT"][:]]))
    else:
        void_coords = np.transpose(np.stack([voids["X"][:],voids["Y"][:],voids["Z"][:],np.full(len(voids["Z"][:]),xcf["weights_voids"]),voids["REDSHIFT"][:]]))
    void_coords[:,3] = void_coords[:,3] * (((1 + void_coords[:,4])/(1+xcf["z_ref"]))**(xcf["z_evol_obj"]-1))
    mask = np.full(len(void_coords),True)
    if(xcf["z_max_obj"] is not None):
        mask &= void_coords[:,4] <= xcf["z_max_obj"]
    if(xcf["z_min_obj"] is not None):
        mask &= void_coords[:,4] >= xcf["z_min_obj"]
    xcf["voids"]=void_coords[mask]
    print('..done. Number of voids:', len(xcf['voids']))


def read_deltas(xcf,delta_path):
    # CR - To replace when lslyatomo is used
    print('Reading deltas from ', delta_path)
    deltas = []
    delta_names = glob.glob(os.path.join(delta_path,"delta-*.fits"))
    for delta_name in delta_names:
        delta = fitsio.FITS(delta_name)
        for i in range(1,len(delta)):
            delta_dict = {}
            header = delta[i].read_header()
            delta_dict["x"] = header["X"]
            delta_dict["y"] = header["Y"]
            delta_dict["z"] = delta[i]["Z"][:]
            delta_dict["delta"] = delta[i]["DELTA"][:]
            redshift = ((10**delta[i]["LOGLAM"][:] / lambdaLy)-1)
            delta_dict["weights"] = delta[i]["WEIGHT"][:] *(((1+redshift)/(1+xcf["z_ref"]))**(xcf["z_evol_del"]-1))
            deltas.append(delta_dict)
    xcf["deltas"] = deltas
    print('..done. Number of deltas:', len(deltas), 'Number of pixels:', len(deltas)*len(redshift))



def fill_neighs(xcf):
    print('Filling neighbors..')
    deltas = xcf["deltas"]
    voids = xcf["voids"]
    r_max = xcf["r_max"]
    for d in deltas:
        x,y,z = d["x"],d["y"],d["z"]
        mask = np.sqrt((voids[:,0]-x)**2 + (voids[:,1]-y)**2) <= r_max
        mask &= voids[:,2]  <= np.max(z) - r_max
        mask &= voids[:,2]  >= np.min(z) + r_max
        d["neighbors"] = voids[mask]
    print('..done')




def fast_xcf(xcf,x1,y1,z1,w1,d1,x2,y2,z2,w2):
    x1_array,y1_array = np.full(z1.shape,x1),np.full(z1.shape,y1)
    rt = np.sqrt((x1_array[:,None] - x2)**2 + (y1_array[:,None] - y2)**2)
    rp = (z2 - z1[:,None])
    r = np.sqrt(rp**2+rt**2)
#    mask = r == 0
#    mu = np.zeros(r.shape)
#    mu[mask] = np.nan
#    mu[~mask] = rp[~mask]/r[~mask]
    mu = rp/r

    z = (z1[:,None]+z2)/2

    we = w1[:,None]*w2
    wde = (w1*d1)[:,None]*w2
    w = (r>xcf["r_min"]) & (r<xcf["r_max"]) & (mu>xcf["mu_min"]) & (mu<xcf["mu_max"])
    r = r[w]
    mu = mu[w]
    z  = z[w]
    we = we[w]
    wde = wde[w]


    br = ((r-xcf["r_min"])/(xcf["r_max"]-xcf["r_min"])*xcf["nbins_r"]).astype(int)
    bmu = ((mu-xcf["mu_min"])/(xcf["mu_max"]-xcf["mu_min"])*xcf["nbins_mu"]).astype(int)

    bins = bmu + xcf["nbins_mu"]*br

    rebin_xi = np.bincount(bins,weights=wde)
    rebin_weight = np.bincount(bins,weights=we)
    rebin_r = np.bincount(bins,weights=r*we)
    rebin_mu = np.bincount(bins,weights=mu*we)
    rebin_z = np.bincount(bins,weights=z*we)
    rebin_num_pairs = np.bincount(bins,weights=(we>0.))

    return (rebin_weight, rebin_xi, rebin_r, rebin_mu, rebin_z, rebin_num_pairs)




def xcorr_cartesian(xcf,save_corr=None):
    nr = xcf["nbins_r"]
    nmu = xcf["nbins_mu"]
    xi = np.zeros(nr*nmu)
    weights = np.zeros(nr*nmu)
    r = np.zeros(nr*nmu)
    mu = np.zeros(nr*nmu)
    z = np.zeros(nr*nmu)
    num_pairs = np.zeros(nr*nmu, dtype=np.int64)
    deltas = xcf["deltas"]
    for d in deltas:
        voids = d["neighbors"]
        if (len(voids) != 0):
            delta_values = d["delta"]
            x_del,y_del,z_del = d["x"],d["y"],d["z"]
            weights_deltas =  d["weights"]
            x_voids = voids[:,0]
            y_voids = voids[:,1]
            z_voids = voids[:,2]
            weights_voids = voids[:,3]
            (rebin_weight, rebin_xi, rebin_r, rebin_mu, rebin_z, rebin_num_pairs) =  fast_xcf(xcf,x_del,y_del,z_del,weights_deltas,delta_values,x_voids,y_voids,z_voids,weights_voids)
            xi[:len(rebin_xi)]+=rebin_xi
            weights[:len(rebin_weight)]+=rebin_weight
            r[:len(rebin_r)]+=rebin_r
            mu[:len(rebin_mu)]+=rebin_mu
            z[:len(rebin_z)]+=rebin_z
            num_pairs[:len(rebin_num_pairs)]+=rebin_num_pairs.astype(int)

    w = weights>0
    xi[w]/=weights[w]
    r[w]/=weights[w]
    mu[w]/=weights[w]
    z[w]/=weights[w]

    if(save_corr is not None):
        xcorr = xcorr_objects.CrossCorr(name=save_corr,mu_array=mu,r_array=r,xi_array=xi,z_array=z,exported=True)
        xcorr.write(xcf,weights)

    r = r.reshape(nr, nmu)
    mu = mu.reshape(nr, nmu)
    xi = xi.reshape(nr, nmu)
    return(r,mu,xi,z)


# def xcorr_cartesian_error(xcf,nb_bins=5,save_corr=None):
#     (r,mu,xi,z) = xcorr_cartesian(xcf,save_corr=save_corr)
#     deltas = xcf["deltas"]
#     sub_deltas = int(round(len(deltas)/(nb_bins-1),0))
#
#     xcorr = xcorr_objects.CrossCorr.init_from_fits(file_xi_no_export,exported=False,supress_first_pixels=supress_first_pixels)
#     mu,da =  xcorr.mu_array,xcorr.xi_array
#     monopole,dipole,quadrupole,hexadecapole = [],[],[],[]
#     for i in range(len(da)):
#         (mono,di,quad,hexa) = get_poles(mu,da[i])
#         monopole.append(mono)
#         dipole.append(di)
#         quadrupole.append(quad)
#         hexadecapole.append(hexa)
#
#     xcorr = xcorr_objects.CrossCorr(name=save_corr,mu_array=mu,r_array=r,xi_array=xi,z_array=z,exported=False)
#     xcorr.write(xcf,weights)
#     error_monopole = sem(np.array(monopole))
#     error_dipole = sem(np.array(dipole))
#     error_quadrupole = sem(np.array(quadrupole))
#     error_hexadecapole = sem(np.array(hexadecapole))



def xcorr(delta_path,void_catalog,xcorr_options,corr_name,error_calculation=None):
    xcf = create_xcf(delta_path,void_catalog,xcorr_options)
    if(error_calculation is not None):
        return()
    else:
        xcorr_cartesian(xcf,save_corr=corr_name)











    ###
