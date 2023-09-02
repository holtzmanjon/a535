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


def generate_sed(wavelengths, spec_type='bb5500'):
    """Generates the spectral energy distribution of a given type at the given wavelengths

        Uses spectral models to create an array of photon fluxes at the given wavelengths.
        Currently only supports blackbody spectra. Also, it currently returns per sr, but
        we will want flux, so will need to think about how to change that.

        Parameters:
            wavelengths (np.array): numpy array of the wavelengths (with astropy units) at
                                    which the SED should be calculated
            spec_type (string): the type of SED to be generated; defaults to a blackbody with temperature
                                5500K


        Returns:
            np.array: numpy array of the same length as wavelengths with flux distributed
                      based on spec_type

    """
    sed = None
    if 'bb' in spec_type:
        temp = float(spec_type[2:]) * u.K
        bb = models.BlackBody(temperature=temp, scale=1*u.J*u.m**-2/u.s/u.micron/u.sr)
        sed = bb(wavelengths)
    else:
        print('Spectral type currently unsupported. Null value returned.')

    return sed


def calc_monochrome_signal(wavelengths, sed, transmission, tele_area=(np.pi*(3.5/2*u.m)**2), exp_time=(100*u.s)):
    """Calculates the monochromatic flux of the given SED

        Uses the signal equation, along with the given transmission curve, to calculate
        the monochromatic signal at the given wavelengths. Wavelengths and sed should
        have units using astropy.units.

        Parameters:
            wavelengths (np.array): numpy array of the wavelengths (with astropy units) at
                                    which the SED should be calculated
            sed (np.array): numpy array of the same size as wavelengths which contains
                            the flux at the given wavelengths
            transmission (np.array): transmission spectrum (combined to include any relevant factors
                                     including atmosphere, detector effeciency, etc.)
            tele_area (Quantity): telescope area; defaults to the area of a telescope with diameter
                                  3.5 m; can use 1 (with no units) to receive signal per area
            exp_time (Quantity): exposure time for the observation; defaults to 100 s; can use 1 (with
                                 no units) to receive photons per second rather than total photons


        Returns:
            np.array: numpy array of the same length as wavelengths of predicted photon counts
                      at each wavelength

    """
    energy = const.c * const.h / wavelengths
    monochrome_signal = (tele_area * exp_time * sed * transmission / energy).decompose()

    return monochrome_signal


def main():
    pass


if __name__ == "__main__":
    main()
