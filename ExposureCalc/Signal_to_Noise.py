import numpy as np
from astropy import units as u

def SNR(S, exp_times, A = 1 * u.arcsec**2, B = 2 * u.arcsec**-2 * u.s**-1, N_pix = 5, sig = 10, spec = False):
    """
    Compute the Signal-to-Noise Ratio (SNR).

    Parameters
    ----------
    S : float or array-like
        Signal value.
    exp_times : float or array-like
        Exposure times.
    A : `~astropy.units.Quantity`, optional
        Area term with units of arcsec^2. Default is 1 arcsec^2.
    B : `~astropy.units.Quantity`, optional
        Background term with units of arcsec^-2 s^-1. Default is 2 arcsec^-2 s^-1.
    N_pix : int, optional
        Number of pixels. Default is 5.
    sig : float, optional
        Sigma value. Default is 10.
    spec : bool, optional
        A flag to specify a particular mode. If False, default mode is applied. Default is False.

    Returns
    -------
    SNR : float or array-like
        Signal-to-Noise Ratio.
    """
    if not spec:
        SNR = (S*exp_times) / np.sqrt(S*exp_times + A*B*exp_times + N_pix * sig**2)

    return SNR

def ExpTime(S, desired_SNR, A = 1 * u.arcsec**2, B = 2 * u.arcsec**-2 * u.s**-1, N_pix = 5, sig = 10, spec = False):
    """
    Compute the required exposure time to achieve a desired SNR.

    Parameters
    ----------
    S : float or array-like
        Signal value.
    desired_SNR : float
        Desired Signal-to-Noise Ratio value.
    A : `~astropy.units.Quantity`, optional
        Area term with units of arcsec^2. Default is 1 arcsec^2.
    B : `~astropy.units.Quantity`, optional
        Background term with units of arcsec^-2 s^-1. Default is 2 arcsec^-2 s^-1.
    N_pix : int, optional
        Number of pixels. Default is 5.
    sig : float, optional
        Sigma value. Default is 10.
    spec : bool, optional
        A flag to specify a particular mode. If False, default mode is applied. Default is False.

    Returns
    -------
    exp_time : float
        Exposure time required to achieve the desired SNR.

    """
    if not spec:
        a = -S**2
        b = (desired_SNR)**2 * (S+A*B)
        c = (desired_SNR)**2 * sig**2 * N_pix
        D = b**2 - 4*a*c

        print((-b - np.sqrt(D)) / (2*a))
        print((-b + np.sqrt(D)) / (2*a))

    return (-b - np.sqrt(D)) / (2*a)