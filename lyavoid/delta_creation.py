import fitsio
import numpy as np
from scipy import interpolate
from scipy.optimize import fsolve
import os
try:
   import picca.constants as constants_picca
except:
    import lsstomo.picca.constants as constants_picca
    raise Warning("Picca might be updated, we suggest to install picca independently")
import lsstomo.tomography as tomography
import pickle
from scipy.ndimage import map_coordinates



# from lslyatomo import tomographic_objects


class DeltaGenerator(object):

    def __init__(self,pwd,shape_map,box_DM_name,los_selection,nb_files,mode="distance_redshift"):

        self.pwd = pwd
        self.shape_map = shape_map
        self.box_DM_name = box_DM_name
        self.los_selection = los_selection
        self.nb_files = nb_files
        self.mode = mode



    def compute_redshift_array(self,minredshift,z,rcomov):
        redshift = np.zeros(z.shape)
        for i in range(len(z)):
            redshift[i] = fsolve(self.f,minredshift,args=(z[i],rcomov))
        return(redshift)

    def compute_redshift(self,minredshift,z,rcomov):
        return(fsolve(self.f,minredshift,args=(z,rcomov)))

    def f(self,redshift,z,rcomov):
        return(rcomov(redshift) - (z))


    def create_my_cosmo_function(self,Om):
        Cosmo = constants_picca.cosmo(Om)
        rcomoving = Cosmo.r_comoving
        rcomov = rcomoving
        distang = Cosmo.dm
        return(distang,rcomov,Cosmo)



    def select_deltas(self,size_map ,max_list_name,Omega_m):
        if(self.mode == "distance_redshift"):
            (deltas_list,deltas_props,redshift_array) = self.select_deltas_distance_redshift(size_map ,max_list_name,Omega_m)
        elif(self.mode == "full"):
            (deltas_list,deltas_props,redshift_array) = self.select_deltas_full(size_map ,max_list_name,Omega_m)
        elif(self.mode == "full_angle"):
            (deltas_list,deltas_props,redshift_array) = self.select_deltas_full_angle(size_map ,max_list_name,Omega_m)
        elif(self.mode == "cartesian"):
            (deltas_list,deltas_props,redshift_array) = self.select_deltas_cartesian(size_map ,max_list_name,Omega_m)
        return(deltas_list,deltas_props,redshift_array)






    def select_deltas_distance_redshift(self,size_map ,max_list_name,Omega_m):
        tomo = tomography.TreatClamato(self.pwd,MapName=self.box_DM_name,shapeMap=self.shape_map,PixelName=None)
        map_3d = tomo.readClamatoMapFile()
        (x_array,y_array,z_array) = self.create_index_arrays(self.shape_map)
        (ra_array,dec_array,redshift_array) =self.compute_celest_coordinates_from_cartesians(x_array,y_array,z_array,Omega_m,max_list_name,size_map,self.shape_map)
        z_fake_qso = np.max(redshift_array)
        deltas_props = [{"RA": ra_array[i], "DEC": dec_array[j], "Z" : z_fake_qso, "THING_ID":'1' + '{0:04d}'.format(i) + '{0:04d}'.format(j), "PLATE":'1' + '{0:04d}'.format(i) + '{0:04d}'.format(j), "MJD":'1' + '{0:04d}'.format(i) + '{0:04d}'.format(j), "FIBERID":'1' + '{0:04d}'.format(i) + '{0:04d}'.format(j)} for i in range(len(ra_array)) for j in range(len(dec_array))]
        indices =np.moveaxis(np.array(np.meshgrid(x_array,y_array,indexing='ij')),0,-1)
        deltas = np.zeros((self.shape_map[0]//self.los_selection["rebin"],self.shape_map[1]//self.los_selection["rebin"],self.shape_map[2]))
        deltas[:,:,:] = map_3d[indices[:,:,0],indices[:,:,1],:]
        deltas_list = np.array([deltas[i,j,:] for i in range(len(deltas)) for j in range(len(deltas[i]))])
        return(deltas_list,deltas_props,redshift_array)


    def create_index_arrays(self ,shape_map):
        if(self.los_selection["method"] == "equidistant"):
            x_array = np.round(np.linspace(0,shape_map[0]-1,int(shape_map[0]//self.los_selection["rebin"])),0).astype(int)
            y_array = np.round(np.linspace(0,shape_map[1]-1,int(shape_map[1]//self.los_selection["rebin"])),0).astype(int)
            z_array = np.arange(0,shape_map[2])
        return(x_array,y_array,z_array)

    def compute_celest_coordinates_from_cartesians(self,x_array,y_array,z_array,Omega_m,max_list_name,size_map,shape_map,method ="mean"):
        max_list = pickle.load(open(max_list_name,'rb'))
        minx, miny,minz ,minredshift,maxredshift= max_list[0],max_list[2],max_list[4],max_list[6],max_list[7]
        if(method == "mean"):
            angular_distance_redshift =(minredshift +  maxredshift)/2
# shape/size
        mpc_per_pixels = np.array(size_map)/(np.array(shape_map)-1)
        mpc_per_pixels = np.array(size_map)/(np.array(shape_map))
        x_array_mpc, y_array_mpc,z_array_mpc = x_array * mpc_per_pixels[0] + minx, y_array* mpc_per_pixels[1] + miny, z_array * mpc_per_pixels[2] + minz
        (distang,rcomov,Cosmo) = self.create_my_cosmo_function(Omega_m)
        redshift_array = self.compute_redshift_array(minredshift,z_array_mpc,rcomov)
        dist_angular = distang(angular_distance_redshift)
        ra_array = (x_array_mpc / dist_angular)
        dec_array = (y_array_mpc / dist_angular)
        return(ra_array,dec_array,redshift_array)


    def select_deltas_cartesian(self,size_map ,max_list_name,Omega_m):
        tomo = tomography.TreatClamato(self.pwd,MapName=self.box_DM_name,shapeMap=self.shape_map,PixelName=None)
        map_3d = tomo.readClamatoMapFile()
        (x_array,y_array,z_array,redshift_array,indices) = self.create_index_arrays_cartesian(Omega_m,max_list_name,size_map,self.shape_map)
        z_fake_qso = np.max(redshift_array)
        deltas_props = [{"X": x_array[i], "Y": y_array[j], "Z" : z_array, "ZQSO" : z_fake_qso, "THING_ID":'1' + '{0:04d}'.format(i) + '{0:04d}'.format(j), "PLATE":'1' + '{0:04d}'.format(i) + '{0:04d}'.format(j), "MJD":'1' + '{0:04d}'.format(i) + '{0:04d}'.format(j), "FIBERID":'1' + '{0:04d}'.format(i) + '{0:04d}'.format(j)} for i in range(len(x_array)) for j in range(len(y_array))]
        deltas = np.zeros((self.shape_map[0]//self.los_selection["rebin"],self.shape_map[1]//self.los_selection["rebin"],self.shape_map[2]))
        deltas[:,:,:] = map_3d[indices[:,:,0],indices[:,:,1],:]
        deltas_list = np.array([deltas[i,j,:] for i in range(len(deltas)) for j in range(len(deltas[i]))])
        return(deltas_list,deltas_props,redshift_array)


    def create_index_arrays_cartesian(self ,Omega_m,max_list_name,size_map,shape_map,method ="mean"):
        if(self.los_selection["method"] == "equidistant"):
            x_array = np.round(np.linspace(0,shape_map[0]-1,int(shape_map[0]//self.los_selection["rebin"])),0).astype(int)
            y_array = np.round(np.linspace(0,shape_map[1]-1,int(shape_map[1]//self.los_selection["rebin"])),0).astype(int)
            z_array = np.arange(0,shape_map[2])
        indices =np.moveaxis(np.array(np.meshgrid(x_array,y_array,indexing='ij')),0,-1)
        max_list = pickle.load(open(max_list_name,'rb'))
        minx, miny,minz ,minredshift= max_list[0],max_list[2],max_list[4],max_list[6]
# shape/size
        mpc_per_pixels = np.array(size_map)/(np.array(shape_map)-1)
        mpc_per_pixels = np.array(size_map)/(np.array(shape_map))
        x_array_mpc, y_array_mpc,z_array_mpc = x_array * mpc_per_pixels[0] + minx, y_array* mpc_per_pixels[1] + miny, z_array * mpc_per_pixels[2] + minz
        (distang,rcomov,Cosmo) = self.create_my_cosmo_function(Omega_m)
        redshift_array = self.compute_redshift_array(minredshift,z_array_mpc,rcomov)
        return(x_array_mpc, y_array_mpc,z_array_mpc,redshift_array,indices)


    def select_deltas_full(self,size_map ,max_list_name,Omega_m):
        tomo = tomography.TreatClamato(self.pwd,MapName=self.box_DM_name,shapeMap=self.shape_map,PixelName=None)
        map_3d = tomo.readClamatoMapFile()
        (distang,rcomov,Cosmo) = self.create_my_cosmo_function(Omega_m)
        max_list = pickle.load(open(max_list_name,'rb'))
        minx, miny,minz,minredshift= max_list[0],max_list[2],max_list[4],max_list[6]
        ra_array,dec_array,redshift_array= self.create_radecz_array(distang,rcomov,minredshift,minx, miny,minz,self.shape_map,size_map )
        indice =np.moveaxis(np.array(np.meshgrid(ra_array,dec_array,redshift_array,indexing='ij')),0,-1)
        indice_mpc=self.convert_indice_matrix_my_cosmo(indice,distang,rcomov,minx, miny, minz)
# shape/size
        nb_mpc_per_pixel = np.array(size_map)/(np.array(self.shape_map) - 1)
        nb_mpc_per_pixel = np.array(size_map)/(np.array(self.shape_map))
        indice_pixels = indice_mpc/nb_mpc_per_pixel
        list_indice = indice_pixels.reshape(indice_pixels.shape[0]*indice_pixels.shape[1]*indice_pixels.shape[2],indice_pixels.shape[3])
        z_fake_qso = np.max(redshift_array)
        deltas = map_coordinates(map_3d, np.transpose(list_indice), order=1).reshape((indice_pixels.shape[0],indice_pixels.shape[1],indice_pixels.shape[2]))
        deltas = deltas.reshape(deltas.shape[0]*deltas.shape[1],deltas.shape[2])
        deltas_props = [{"RA": ra_array[i], "DEC": dec_array[j], "Z" : z_fake_qso, "THING_ID":'1' + '{0:04d}'.format(i) + '{0:04d}'.format(j), "PLATE":'1' + '{0:04d}'.format(i) + '{0:04d}'.format(j), "MJD":'1' + '{0:04d}'.format(i) + '{0:04d}'.format(j), "FIBERID":'1' + '{0:04d}'.format(i) + '{0:04d}'.format(j)} for i in range(len(ra_array)) for j in range(len(dec_array))]
        return(deltas,deltas_props,redshift_array)






    def create_radecz_array(self,distang,rcomov,minredshift,minx, miny,minz,shape_map,size_map ):
# shape/size
        mpc_per_pixel = np.array(size_map)/(np.array(shape_map)-1)
        mpc_per_pixel = np.array(size_map)/(np.array(shape_map))
        z_array = np.arange(self.shape_map[2]) * mpc_per_pixel[2] + minz
        redshift_array = np.zeros(self.shape_map[2])
        for i in range(self.shape_map[2]):
            redshift_array[i] = self.compute_redshift(minredshift,z_array[i],rcomov)
        dist_angular_max = distang(redshift_array[-1])
        ramin,ramax = (minx)/dist_angular_max, (size_map[0]+minx)/dist_angular_max
        decmin,decmax = (miny)/dist_angular_max, (size_map[1]+miny)/dist_angular_max
        ra_array = np.linspace(ramin,ramax,shape_map[0]//self.los_selection["rebin"])
        dec_array = np.linspace(decmin,decmax,shape_map[1]//self.los_selection["rebin"])
        return(ra_array,dec_array,redshift_array)


    def convert_indice_matrix_my_cosmo(self,indice,distang,rcomov,minx, miny,minz):
        indice_mpc = np.zeros(indice.shape)
        for i in range(indice.shape[2]):
            redshift = indice[0,0,i,2]
            Z = rcomov(redshift) - minz
            dist_angular = distang(redshift)
            X = (indice[:,:,i,0])*dist_angular - minx
            Y = (indice[:,:,i,1])*dist_angular - miny
            indice_mpc[:,:,i,0],indice_mpc[:,:,i,1],indice_mpc[:,:,i,2] = X,Y,Z
        return(indice_mpc)




    def select_deltas_full_angle(self,size_map ,max_list_name,Omega_m):
        tomo = tomography.TreatClamato(self.pwd,MapName=self.box_DM_name,shapeMap=self.shape_map,PixelName=None)
        map_3d = tomo.readClamatoMapFile()
        (distang,rcomov,Cosmo) = self.create_my_cosmo_function(Omega_m)
        max_list = pickle.load(open(max_list_name,'rb'))
        minx, miny,minz= max_list[0],max_list[2],max_list[4]
        ra_array,dec_array,redshift_array= self.create_radecz_array_angle(rcomov,minx, miny,minz,self.shape_map,size_map )
        indice =np.moveaxis(np.array(np.meshgrid(ra_array,dec_array,redshift_array,indexing='ij')),0,-1)
        indice_mpc=self.convert_indice_matrix_my_cosmo_angle(indice,distang,rcomov,minx, miny, minz)
# shape/size
        nb_mpc_per_pixel = np.array(size_map)/(np.array(self.shape_map) - 1)
        nb_mpc_per_pixel = np.array(size_map)/(np.array(self.shape_map))
        indice_pixels = indice_mpc/nb_mpc_per_pixel
        list_indice = indice_pixels.reshape(indice_pixels.shape[0]*indice_pixels.shape[1]*indice_pixels.shape[2],indice_pixels.shape[3])
        z_fake_qso = np.max(redshift_array)
        deltas = map_coordinates(map_3d, np.transpose(list_indice), order=1).reshape((indice_pixels.shape[0],indice_pixels.shape[1],indice_pixels.shape[2]))
        deltas = deltas.reshape(deltas.shape[0]*deltas.shape[1],deltas.shape[2])
        deltas_props = [{"RA": ra_array[i], "DEC": dec_array[j], "Z" : z_fake_qso, "THING_ID":'1' + '{0:04d}'.format(i) + '{0:04d}'.format(j), "PLATE":'1' + '{0:04d}'.format(i) + '{0:04d}'.format(j), "MJD":'1' + '{0:04d}'.format(i) + '{0:04d}'.format(j), "FIBERID":'1' + '{0:04d}'.format(i) + '{0:04d}'.format(j)} for i in range(len(ra_array)) for j in range(len(dec_array))]
        return(deltas,deltas_props,redshift_array)




    def create_radecz_array_angle(self,rcomov,minx, miny,minz,shape_map,size_map ):
# shape/size
        mpc_per_pixel = np.array(size_map)/(np.array(shape_map)-1)
        mpc_per_pixel = np.array(size_map)/(np.array(shape_map))
        indice = np.transpose(np.indices(shape_map),axes=(1,2,3,0))*mpc_per_pixel
        indice = indice + [minx,miny,minz]
        indice_radecz = self.full_XYZ_to_RADECz(indice[:,:,:,0],indice[:,:,:,1],indice[:,:,:,2],rcomov)
        ramin,ramax = np.min(indice_radecz[:,:,:,0]),np.max(indice_radecz[:,:,:,0])
        decmin,decmax = np.min(indice_radecz[:,:,:,1]),np.max(indice_radecz[:,:,:,1])
        redmin,redmax = np.min(indice_radecz[:,:,:,2]),np.max(indice_radecz[:,:,:,2])
        ra_array = np.linspace(ramin,ramax,shape_map[0]//self.los_selection["rebin"])
        dec_array = np.linspace(decmin,decmax,shape_map[1]//self.los_selection["rebin"])
        redshift_array = np.linspace(redmin,redmax,shape_map[2])
        return(ra_array,dec_array,redshift_array)


    def convert_indice_matrix_my_cosmo_angle(self,indice,rcomov,minx, miny,minz):
        indice_mpc = np.zeros(indice.shape)
        indice_mpc = self.full_RADECz_to_XYZ(indice[:,:,:,0],indice[:,:,:,1],indice[:,:,:,2],rcomov)
        indice_mpc = indice_mpc - [minx,miny,minz]
        return(indice_mpc)

    def full_XYZ_to_RADECz(self,X,Y,Z,rcomov):
        RA = np.arctan2(X,Z)
        DEC = np.arcsin(Y/np.sqrt(X**2 + Y**2 + Z**2))
        z = self.inverse_dm(np.sqrt(X**2 + Y**2 + Z**2),rcomov)
        return(RA,DEC,z)

    def full_RADECz_to_XYZ(self,RA,DEC,z,rcomov):
        X = rcomov(z)*np.sin(RA)*np.cos(DEC)
        Y = rcomov(z)*np.sin(DEC)
        Z = rcomov(z)*np.cos(RA)*np.cos(DEC)
        return(X,Y,Z)

    def inverse_dm(self,R,rcomov):
        redshift_array = np.linspace(0,5,10000)
        R_array = rcomov(redshift_array)
        inv_dm = interpolate.interp1d(R_array,redshift_array)
        return(inv_dm(R))




    def create_a_delta_file(self,delta,delta_props,name_out,redshift_array,weight=1.0,cont=1.0):
        new_delta = fitsio.FITS(os.path.join(self.pwd,name_out),'rw',clobber=True)
        lambdaLy = 1215.673123130217
        loglambda = np.log10((1 + redshift_array)* lambdaLy)
        for j in range(len(delta)):
            nrows = len(delta[j])
            h = np.zeros(nrows, dtype=[('LOGLAM','f8'),('DELTA','f8'),('WEIGHT','f8'),('CONT','f8')])
            h['DELTA'] = delta[j][:]
            h['LOGLAM'] = loglambda
            h['WEIGHT'] = np.array([weight for i in range(nrows)])
            h['CONT'] = np.array([cont for i in range(nrows)])
            head = {}
            head['THING_ID'] = delta_props[j]["THING_ID"]
            if delta_props[j]["RA"]>=0 :
                head['RA'] = delta_props[j]["RA"]
            else:
                head['RA'] = delta_props[j]["RA"] + 2*np.pi
            head['DEC'] = delta_props[j]["DEC"]
            head['Z']  = delta_props[j]["Z"]
            head['PLATE'] = delta_props[j]["PLATE"]
            head['MJD'] = delta_props[j]["MJD"]
            head['FIBERID'] = delta_props[j]["FIBERID"]
            new_delta.write(h,extname=delta_props[j]["THING_ID"],header=head)
        new_delta.close()


    def create_a_delta_file_cartesian(self,delta,delta_props,name_out,redshift_array,weight=1.0,cont=1.0):
        new_delta = fitsio.FITS(os.path.join(self.pwd,name_out),'rw',clobber=True)
        lambdaLy = 1215.673123130217
        loglambda = np.log10((1 + redshift_array)* lambdaLy)
        for j in range(len(delta)):
            nrows = len(delta[j])
            h = np.zeros(nrows, dtype=[('LOGLAM','f8'),('DELTA','f8'),('WEIGHT','f8'),('Z','f8')])
            h['DELTA'] = delta[j][:]
            h['LOGLAM'] = loglambda
            h['WEIGHT'] = np.array([weight for i in range(nrows)])
            h['Z'] = delta_props[j]["Z"]
            head = {}
            head['X'] = delta_props[j]["X"]
            head['Y'] = delta_props[j]["Y"]
            head['ZQSO']  = delta_props[j]["ZQSO"]
            head['PLATE'] = delta_props[j]["PLATE"]
            head['MJD'] = delta_props[j]["MJD"]
            head['FIBERID'] = delta_props[j]["FIBERID"]
            head['THING_ID'] = delta_props[j]["THING_ID"]
            new_delta.write(h,extname=delta_props[j]["THING_ID"],header=head)
        new_delta.close()


    def save_deltas(self,deltas_list,deltas_props,redshift_array):
        nb_deltas = len(deltas_list)
        nb_deltas_per_files = nb_deltas//(self.nb_files-1)
        for i in range(self.nb_files-1):
            delta = deltas_list[i*nb_deltas_per_files:(i+1)*nb_deltas_per_files,:]
            delta_props = deltas_props[i*nb_deltas_per_files:(i+1)*nb_deltas_per_files]
            if(self.mode == "cartesian"):
                self.create_a_delta_file_cartesian(delta,delta_props,'delta-{}.fits'.format(i),redshift_array)
            else:
                self.create_a_delta_file(delta,delta_props,'delta-{}.fits'.format(i),redshift_array)
        delta = deltas_list[(i+1)*nb_deltas_per_files::,:]
        delta_props = deltas_props[(i+1)*nb_deltas_per_files::]
        if(self.mode == "cartesian"):
            self.create_a_delta_file_cartesian(delta,delta_props,'delta-{}.fits'.format(i+1),redshift_array)
        else:
            self.create_a_delta_file(delta,delta_props,'delta-{}.fits'.format(i+1),redshift_array)




    def create_deltas_from_cube(self,size_map ,max_list_name,Omega_m):
        (deltas_list,deltas_props,redshift_array) = self.select_deltas(size_map ,max_list_name,Omega_m)
        self.save_deltas(deltas_list,deltas_props,redshift_array)
