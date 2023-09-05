import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from astropy import units as u

def telescopeArea(diameter):
    """
    Calculate the collecting area of a telescope based on its diameter.

    Parameters
    ----------
    diameter : float
        Diameter of the telescope in centimeters.

    Returns
    -------
    Quantity
        Collecting area of the telescope in square centimeters.
    """

    # Compute the radius of the telescope in cm
    radius = (diameter / 2) * u.cm

    return np.pi * (radius) ** 2 

def telescopeTransmission(wavelength, constant = None, path = None):
    """
    Calculate the telescope's transmission either as a constant value or from a given file.

    Parameters
    ----------
    wavelength : array_like
        Array of wavelengths at which the transmission is to be evaluated.

    constant : float, optional
        Constant value for the telescope's transmission. If provided, the function will return 
        an array with this constant value at each given wavelength. Default is None.

    path : str, optional
        Path to a CSV file containing the telescope's transmission as a function of wavelength. 
        This file should have two columns: 1. Wavelengths 2. Transmission data. If `constant` is not provided, 
        this argument must be given. Default is None.

    Returns
    -------
    array
        Array of the telescope's transmission values at the given wavelengths.

    Raises
    ------
    ValueError
        If neither `constant` nor `path` is provided.

    Notes
    -----
    The function either uses a constant value for the transmission or reads it from a file. 
    If reading from a file, it uses cubic spline interpolation to evaluate the transmission 
    at the desired wavelengths.

    Example
    -------
    >>> wavelengths = [500, 600, 700]
    >>> telescopeTransmission(wavelengths, constant=0.9)
    array([0.9, 0.9, 0.9])
    """

    # If the transmission is a constant value, turn into an array with corresponding entries for each wavelength
    if constant is not None:
        return np.full(wavelength.shape, constant)

    # If not, read the transmission data from a file
    df = pd.read_csv(path)  # In ths case, source has to contain the absolute path to the file

    # Read the wavelengths and fluxes from the file. 
    # Requires the flie to have two columns, 1. Wavelengths 2. Transmission Data
    wave = df.iloc[0]
    telescope_transmission = df.iloc[1]

    # Create a cubic spline interpolation
    cs = CubicSpline(wave, telescope_transmission)

    # Using the spline, evaluate the flux at the preferred wavelengths
    telescope_transmission = cs(wavelength)

    return telescope_transmission