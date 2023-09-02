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
        Currently only supports blackbody spectra.

        Parameters:
            wavelengths (np.array): numpy array of the wavelengths (with astropy units) at
                                    which the SED should be calculated
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
        sed = bb(wavelengths)
    else:
        print('Spectral type currently unsupported. Null value returned.')

    return sed


def main():
    pass


if __name__ == "__main__":
    main()
