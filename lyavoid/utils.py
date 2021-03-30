import matplotlib.pyplot as plt
# from scipy import integrate

lambdaLy = 1215.673123130217


def return_key(dictionary,string,default_value):
    return(dictionary[string] if string in dictionary.keys() else default_value)


def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

#
# def numerical_integration(method,integrand,variable,axis=None):
#     if(method=="simps"):
#         pole_l = integrate.simps(integrand,variable,axis=axis)
#     return(pole_l)
