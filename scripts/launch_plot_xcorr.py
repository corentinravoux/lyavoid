
import os
import sys
src1 ="/ccc/cont003/home/drf/ravouxco/2_Software/Python/Classes"
if(os.path.isdir(src1)):
    sys.path.append(src1)
    src = src1
src2="/local/home/cravoux/Documents/Python/Classes"
if(os.path.isdir(src2)):
    sys.path.append(src2)
    src = src2
pwd = os.getcwd()
import xcorr_utils


file_xi = "./xcf-void-lya-exp_voids_box.fits"
export_xi ="./xcf-void-lya_voids_box.fits"
nameout_box = "mono_di_quadru_poles_box.pdf"

supress_first_pixels = 0

if __name__ == "__main__":
    xcorr_utils.compute_and_plot_multipole(file_xi,nameout_box,supress_first_pixels=supress_first_pixels,error_bar=export_xi)


