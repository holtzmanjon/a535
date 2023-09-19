import numpy as np
import pandas as pd
from astropy.modeling.physical_models import BlackBody
from scipy.interpolate import CubicSpline
from astropy import constants as const
from astropy import units as u

def normalisedFlux(wavelength, T = None, source = 'constant', const_Flux = None):
    """
    Compute the normalized photon flux for a given source. The flux can be computed for a blackbody radiation based on a given temperature or loaded from a file.

    Parameters
    ----------
    wavelength : array_like
        Wavelength values for which to compute the flux, in Angstroms.
    
    T : float
        Effective temperature in Kelvin. Only used if source is set to 'blackbody'.
    
    source : str, optional
        Specifies the type of the source. Default is 'blackbody'.
        If set to 'blackbody', uses the blackbody radiation formula.
        Otherwise, expects the absolute path to a CSV file containing wavelength and flux data.

    Returns
    -------
    photon_flux : astropy.units.quantity.Quantity
        Photon flux normalized by the photon flux of Vega, in the given wavelength range.

    Notes
    -----
    The CSV file, if used, is expected to have two columns:
    1. Wavelengths
    2. Flux
    The flux is normalized using the photon flux of Vega at 5500 Angstroms.

    Example
    -------
    >>> flux = normalisedFlux([5000, 6000, 7000], 6000)
    >>> print(flux)

    Raises
    ------
    FileNotFoundError
        If the CSV file specified in `source` cannot be found.

    ValueError
        If the CSV file format does not match the expected format.
    """

    if source == 'constant':
        flux = np.full(wavelength.shape, const_Flux) * u.erg / u.cm**2 / u.s / u.Angstrom
        photon_flux = flux * (wavelength * 1e-8 * u.cm) / (const.h.to(u.erg * u.s) * const.c.to(u.cm / u.s))

        return photon_flux

    if source == 'blackbody':
        # Define wavelengths in Angstroms
        wavelength = wavelength * u.Angstrom

        # Define the effective temperature in Kelvin
        T = T * u.K

        # Compute the blackbody flux for the given effective temperature across the given wavelength range
        blackbody = BlackBody(temperature=T)
        flux = blackbody(wavelength)

    else:
        df = pd.read_csv(source)  # In ths case, source has to contain the absolute path to the file

        # Read the wavelengths and fluxes from the file. 
        # Requires the flie to have two columns, 1. Wavelengths 2. Flux
        wave = df.iloc[0]
        F = df.iloc[1]

        # Create a cubic spline interpolation
        cs = CubicSpline(wave, F)

        # Using the spline, evaluate the flux at the preferred wavelengths
        flux = cs(wavelength)

        # Define wavelengths in Angstroms
        wavelength = wavelength * u.Angstrom

        # Define the flux in CGS units
        flux = flux * u.erg / u.cm**2 / u.s / u.Angstrom


    # Convert this into a photon flux
    photon_flux = flux * wavelength.to(u.cm) / (const.h.to(u.erg * u.s) * const.c.to(u.cm / u.s))

    ####   Normalising this flux by the photon flux of Vega   ####

    # Flux of Vega at 5500AA
    F_Vega = 3.63e-9 * u.erg / u.cm**2 / u.s / u.Angstrom

    # Convert to a photon flux
    F_Vega_photon = F_Vega * (5500e-8 * u.cm) / (const.h.to(u.erg * u.s) * const.c.to(u.cm / u.s))

    # Normalise the photon flux of the object by this value
    photon_flux = photon_flux / F_Vega_photon

    return photon_flux

    