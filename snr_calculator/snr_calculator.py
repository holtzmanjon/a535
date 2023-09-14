# This file contains functions for predicting the SNR of an observation
# given a spectral energy distribution and information about the
# telescope
import sys
import numpy as np
import astropy
from astropy import units as u
from astropy.modeling import models
from astropy import constants as const
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy.integrate import quad


def generate_sed(wavelengths, mag, spec_type='bb5500'):
    """Generates the spectral energy distribution of a given type at the given wavelengths

        Uses spectral models to create an array of photon fluxes at the given wavelengths.
        Currently only supports blackbody spectra. This is normalized so that a star with
        magnitude 0 returns Vega's flux at 5500 Angstroms.

        Parameters:
            wavelengths (np.array): numpy array of the wavelengths (with astropy units) at
                                    which the SED should be calculated
            mag (float): magnitude of the object for which the SED should be calculated; 0
                         is normalized to the flux of Vega
            spec_type (string): the type of SED to be generated; defaults to a blackbody with temperature
                                5500K


        Returns:
            np.array: numpy array of the same length as wavelengths with photon flux distributed
                      based on spec_type

    """
    sed = None
    if 'bb' in spec_type:
        temp = float(spec_type[2:]) * u.K
        bb = models.BlackBody(temperature=temp, scale=1*u.J*u.m**-2/u.s/u.micron/u.sr)
        norm_factor = (3.63e-9*u.erg/u.cm**2/u.s/u.Angstrom) / (bb(5500 * u.Angstrom) * u.sr) * 10**(-0.4 * mag)
        energy = const.c * const.h / wavelengths
        sed = bb(wavelengths) * norm_factor * u.sr / energy
    else:
        print('Spectral type currently unsupported. Null value returned.')

    return sed


def atmos_transmission(wavelengths, model='flat0.8'):
    """Generates the atmospheric transmission curve

        Calculates the transmission curve of the atmosphere based on the model entered.
        Currently only supports a flat model.

        Parameters:
            wavelengths (np.array): numpy array of the wavelengths (with astropy units) at
                                    which the transmission curve should be calculated
            model (string): model to use for the atmospheric transmission; currently only
                            supports a flat transmission curve


        Returns:
            np.array: atmospheric transmission at the given wavelengths

    """
    transmission = np.ones(wavelengths.shape)
    
    #trans=scipy.interpolate.interp1d(wavelengths,airmass,fill_value="extrapolate")
    #trans*=np.ones(data.shape)
    

    if 'flat' in model:
        trans_value = float(model[4:])
        transmission *= trans_value

    return transmission


def instrument_transmission(wavelengths, model='flat0.5'):
    """Generates the instrumental transmission curve

        Calculates the transmission curve of the instrument based on the model entered.
        Currently only supports a flat model.

        Parameters:
            wavelengths (np.array): numpy array of the wavelengths (with astropy units) at
                                    which the transmission curve should be calculated
            model (string): model to use for the instrumental transmission; currently only
                            supports a flat transmission curve


        Returns:
            np.array: instrumental transmission at the given wavelengths

    """
    transmission = np.ones(wavelengths.shape)

    if 'flat' in model:
        trans_value = float(model[4:])
        transmission *= trans_value

    return transmission


def filter_transmission(wavelengths, filter='flat0.8'):
    """Generates the filter transmission curve

        Calculates the transmission curve of the filter based on the model entered.
        Currently only supports a flat model.

        Parameters:
            wavelengths (np.array): numpy array of the wavelengths (with astropy units) at
                                    which the transmission curve should be calculated
            filter (string): type of filter; currently only supports a flat transmission curve


        Returns:
            np.array: filter transmission at the given wavelengths

    """
    transmission = np.ones(wavelengths.shape)

    if 'flat' in filter:
        trans_value = float(filter[4:])
        transmission *= trans_value

    return transmission


def mirror_transmission(wavelengths, filter='flat0.8'):
    """Generates the mirror transmission curve

        Calculates the transmission curve of the mirror based on the model entered.
        Currently only supports a flat model.

        Parameters:
            wavelengths (np.array): numpy array of the wavelengths (with astropy units) at
                                    which the transmission curve should be calculated
            filter (string): type of filter; currently only supports a flat transmission curve


        Returns:
            np.array: mirror transmission at the given wavelengths

    """
    transmission = np.ones(wavelengths.shape)

    if 'flat' in filter:
        trans_value = float(filter[4:])
        transmission *= trans_value

    return transmission

def detector_response(wavelengths, filter='flat0.8'):
    """Generates the detector response for given pixel location

        Calculates the response curve of the detector based on the model entered.
        Currently only supports a flat model.

        Parameters:
            wavelengths (np.array): numpy array of the wavelengths (with astropy units) at
                                    which the transmission curve should be calculated
            filter (string): type of filter; currently only supports a flat transmission curve


        Returns:
            np.array: detector response at the given wavelengths

    """
    response = np.ones(wavelengths.shape)

    if 'flat' in filter:
        trans_value = float(filter[4:])
        response *= trans_value

    return response


def tele_area(name='APO'):
    """Gives telescope area based on telescope name

            Calculates the telescope's area based on its name (or alternately it's radius)

            Parameters:
                name (string): name of the telescope; can also be 'radxx' to use a telescope
                               with radius xx


            Returns:
                float: area of the telescope

    """
    area = None
    if 'rad' in name:
        radius = float(name[3:]) * u.m
        area = np.pi * radius**2
    elif name == 'APO':
        radius = 3.5 / 2 * u.m
        area = np.pi * radius ** 2
    else:
        print('Telescope name not recognized. Null value returned.')

    return area


def calc_monochrome_signal(sed, transmission, area=(np.pi*(3.5/2*u.m)**2), exp_time=(100*u.s)):
    """Calculates the monochromatic flux of the given SED

        Uses the signal equation, along with the given transmission curve, to calculate
        the monochromatic signal at the given wavelengths. Wavelengths and sed should
        have units using astropy.units.

        Parameters:
            sed (np.array): numpy array of the same size as wavelengths which contains
                            the flux at the given wavelengths
            transmission (np.array): transmission spectrum (combined to include any relevant factors
                                     including atmosphere, detector effeciency, etc.)
            area (Quantity): telescope area; defaults to the area of a telescope with diameter
                                  3.5 m; can use 1 (with no units) to receive signal per area
            exp_time (Quantity): exposure time for the observation; defaults to 100 s; can use 1 (with
                                 no units) to receive photons per second rather than total photons


        Returns:
            np.array: numpy array of the same length as wavelengths of predicted photon counts
                      at each wavelength

    """
    monochrome_signal = (area * exp_time * sed * transmission).to(1/u.Angstrom)

    return monochrome_signal

def snr_(mag,wavelengths,time):
    """Calculates the snr of a given object and exposure time
        !!fix later

        Uses the signal equation, along with the given transmission curve, to calculate
        the monochromatic signal at the given wavelengths. Wavelengths and sed should
        have units using astropy.units.

        Parameters:
            sed (np.array): numpy array of the same size as wavelengths which contains
                            the flux at the given wavelengths
            transmission (np.array): transmission spectrum (combined to include any relevant factors
                                     including atmosphere, detector effeciency, etc.)
            area (Quantity): telescope area; defaults to the area of a telescope with diameter
                                  3.5 m; can use 1 (with no units) to receive signal per area
            exp_time (Quantity): exposure time for the observation; defaults to 100 s; can use 1 (with
                                 no units) to receive photons per second rather than total photons


        Returns:
            np.array: numpy array of the same length as wavelengths of predicted photon counts
                      at each wavelength

    """
    F0=3.63e-9 #ergs/cm2/s/ang STMAG at 5500
    h=6.626176e-27 #ergs s
    c=2.99792458e10 #cm/s
    atmos_trans=atmos_transmission(wavelengths)
    mirror_trans=mirror_transmission(wavelengths)
    instrument_trans=instrument_transmission(wavelengths)
    filter_trans = filter_transmission(wavelengths)
    detect_respon = detector_response(wavelengths)
    F=F0*10**(-0.4*mag)
    integral=quad(F,wavelengths[0],wavelengths[-1])
    signal=tele_area*time
    aperature=3
    background_area=aperature
    surf_bright=1
    rn_sigma=3
    platescale=3
    Npix=platescale*background_area
    snr=signal/np.sqrt(signal+background_area*surf_bright+rn_sigma*Npix)
    
    return snr
    


def main():
    wavelengths = 5500 * u.Angstrom
    V_mag = 20
    atmos_trans = atmos_transmission(wavelengths)
    filter_trans = filter_transmission(wavelengths)
    instrument_trans = instrument_transmission(wavelengths)
    transmission = atmos_trans * filter_trans * instrument_trans
    area = tele_area(name='rad0.3')
    photon_sed = generate_sed(wavelengths, V_mag)
    signal = calc_monochrome_signal(photon_sed, transmission, area=area, exp_time=(1*u.s))
    print(signal)
    print(signal*850*u.Angstrom)


if __name__ == "__main__":
    main()
