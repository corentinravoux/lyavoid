"""Small shared helpers for lyavoid (Lyman-alpha rest wavelength, mpl sizing)."""

import matplotlib.pyplot as plt

# from scipy import integrate

lambdaLy = 1215.673123130217


def return_key(dictionary, string, default_value):
    """Return ``dictionary[string]`` if present, else ``default_value``.

    Args:
        dictionary (dict): Source mapping.
        string: Key to look up.
        default_value: Value returned when the key is absent.

    Returns:
        The value at ``string`` or ``default_value``.
    """
    return dictionary[string] if string in dictionary.keys() else default_value


def set_size(w, h, ax=None):
    """Resize a matplotlib figure so its axes have an exact physical size.

    Args:
        w (float): Target axes width in inches.
        h (float): Target axes height in inches.
        ax (matplotlib.axes.Axes, optional): Axis to size (current axis if None).
    """
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


#
# def numerical_integration(method,integrand,variable,axis=None):
#     if(method=="simps"):
#         pole_l = integrate.simpson(integrand,variable,axis=axis)
#     return(pole_l)
