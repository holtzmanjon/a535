import numpy as np
from astropy import units as u
from astropy import constants as const
from scipy.integrate import simps

from Telescope import *
from Instrument import *
from Filter import *
from Atmosphere import *
from Flux import *

wavelength = np.arange(5075, 5925, 25) 

# Define a constant flux
photon_flux  = normalisedFlux(wavelength, const_Flux = 3.63e-17)

# Compute the telescope area
area = telescopeArea(60)

# Compute the transmission (same as in Problem Set 2, Q1.2)
atm_transmission = atmosphericTransmission(wavelength, 0.8)
telescope_transmission = instrumentTransmission(wavelength, 0.5)
filter_response = filterTransmission(wavelength, 0.8)

# Define the wavelength in Angstroms
wavelength = wavelength * u.AA

# Perform the integration
integral = simps(photon_flux * atm_transmission * telescope_transmission * filter_response, wavelength) * u.cm**(-2) * u.s**(-1)


print(f"The signal per unit time is {integral * area}")


