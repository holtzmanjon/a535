

import astropy.units as u
import astropy.constants as c
from astropy.modeling.physical_models import BlackBody
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps,trapz
import warnings

class SED:
    '''
    PURPOSE:
        Create an SED class to generate model spectrum
        
    INPUTS:
        mag: magnitude of source
        temperature: temperature of source
        temp_units: units of temperature, 'Kelvin' is supported
        wave_start: starting wavelength of spectrum
        wave_end: ending wavelength of spectrum
        wave_units: units of wavelength "Angstrom" is supported
    '''
    # Initialize the class
    def __init__(self, mag, temperature, temp_units, wave_start, wave_stop, wave_units):
  
        # Define variables
        self.mag = mag
        self.temperature = temperature

        # Define the units of temperature
        if temp_units == 'Kelvin':
            self.temperature = self.temperature*u.K

        # Define the wavelength range
        self.wave_start = wave_start
        self.wave_stop = wave_stop

        # Define the units of wavelength
        if wave_units == "Angstrom":
            self.wave_start = self.wave_start*u.AA
            self.wave_stop = self.wave_stop*u.AA

    def blackbody_spectrum(self):

        '''
        PURPOSE:
                This function creates a blackbody spectrum based on the temperature and magnitude of the star.

        INPUTS:
                        [mag; float]: The magnitude of the star
                [temperature; float]: The temperature of the star
                 [wave_start; float]: The starting wavelength of the spectrum
                  [wave_stop; float]: The ending wavelength of the spectrum

        OUTPUTS:
                          [wavelength; np.array, float]: The wavelength array of the blackbody spectrum
                [normalized_spectrum,; np.array, float]: The normalized blackbody spectrum

        AUTHOR:
                Tim M. Sept. 1 2023.
        '''

        # Create the blackbody spectrum
        blackbody_model = BlackBody(self.temperature, scale = 1*u.erg/u.s/u.cm**2/u.sr/u.AA)

        # Normalize the spectrum
        self.wavelength = np.linspace(self.wave_start, self.wave_stop, 100_000)

        # Define the reference spectrum
        self.model_spectrum = blackbody_model(self.wavelength)
        
        # Normalize the spectrum
        ref_spectrum = BlackBody(temperature = 9_700*u.K, scale = 1*u.erg/u.s/u.cm**2/u.sr/u.AA)

        # Normalize the spectrum
        normalize = (3.63e-9*u.erg/u.cm**2/u.s/u.AA)/(ref_spectrum(5500*u.AA)*u.sr) 

        # Define the normalized spectrum
        self.normalized_spectrum = self.model_spectrum*u.sr*normalize*10.**(-0.4*self.mag)
        
        return self.wavelength, self.normalized_spectrum
    
    def photon_spectrum(self):
        '''
        PURPOSE:
            Create a photon spectrum
        INPUTS:
            Self
        OUTPUTS:
            photon_spectrum, an array in units of photons per second per cm^2 per Angstrom
        '''
        with warnings.catch_warnings():            
            warnings.filterwarnings("ignore", category=RuntimeWarning)          
            wavelength,normalized_spectrum = SED.blackbody_spectrum(self)       
            light_speed = c.c.to(u.cm/u.s)
            h = c.h.to(u.erg*u.s)
            photon_spectrum = normalized_spectrum*(5500*u.AA.to(u.cm)*u.cm)/(h*light_speed)

        return wavelength,photon_spectrum
    
    def plot(self, plot):

        '''
        PURPOSE:
                This function plots the blackbody spectrum.

        INPUTS:
                [plot; string:  Type of plot

        OUTPUTS:
                Displays a plot of the blackbody spectrum.

        AUTHOR:
                Tim M. Sept. 1 2023.
        '''

        # Create normalized spectrum 
        if plot=='Flux Density':
        
            wavelength, spectrum = SED.blackbody_spectrum(self)
            fig,ax = plt.subplots(1, 1, figsize = (10, 8))
            ax.set_xlabel('Wavelength ($\AA$)')
            ax.set_ylabel('Flux Density (ergs $s^{-1} cm^{-2} \AA^{-1}$)')
            ax.plot(wavelength, spectrum)
            plt.show()
        
        if plot=='Photon Flux':
            wavelength, spectrum = SED.photon_spectrum(self)
            fig,ax = plt.subplots(1, 1, figsize = (10, 8))
            ax.set_xlabel('Wavelength ($\AA$)')
            ax.set_ylabel('Flux Density (photons $s^{-1} cm^{-2} \AA^{1})$')
            ax.plot(wavelength, spectrum)
            plt.show()
        
        return 
        
class BandPass:
    '''
    PURPOSE: Create a class for bandpass
    
    INPUTS:
        filter: filter used for observation, 'V' is supported and creates a square filter centered on 5500 Angstroms
    '''
    # Initialize the bandpass class
    def __init__(self, filter):


        # For the Johnson V filter
        if filter == 'V':

            # Define the wavelength range
            self.filter_wavelength = np.arange(0, 10_000)

            # Make the filter array of zeros
            self.filter = np.zeros(len(self.filter_wavelength))

            # Make a boxy filter response
            index = np.where((self.filter_wavelength > 5075) & (self.filter_wavelength < 5925))
            self.filter[index] = 0.8
            
    def convolve_SED(self, wavelength, normalized_spectrum):

        ''' 
        PURPOSE:
                Function that convolves the normalized spectrum with a filter response function.

        INPUTS:
                wavelength: array wavelength of spectrum 
                normalized_spectrum: array for normalized spectrum 

        OUTPUTS:
                [convolved_spectrum; np.array, float]:  Normalized spectrum convolved with the filter response.

        AUTHOR:
                Tim M. Sept. 1 2023.
        '''

        # Interpolate the filter response
        interpolated_filter_values = np.interp(wavelength.value, self.filter_wavelength, self.filter)
        
        # Multiply the normalized spectrum by the interpolated filter values
        convolved_spectrum = normalized_spectrum*interpolated_filter_values
        
        return convolved_spectrum
    
class Telescope:
    '''
    PURPOSE:
        Define telescope parameters to determine collecting area
    INPUTS:
        diameter: diameter of primary mirror
        units: units of length for diameter
        
            
    '''
    def __init__(self,diameter,units):
        if units == 'cm':
            self.diameter = diameter*u.cm
        if units == 'm':
            self.diameter = diameter*u.m.to(u.cm)*u.cm
        if units == 'mm':
            self.diameter = diameter*u.mm.to(u.cm)*u.cm
    
    def calc_area(self):    
        area = np.pi*(self.diameter/2)**2
        return area
   
    def mirror(self):
            mirror_efficiency = 1.0
            return mirror_efficiency
    
class atmosphere:
    '''
    PURPOSE:
        Define absorption due to atmospheric conditions
        
    INPUTS:
        quality: Either 'good' or 'bad' observing conditions
    '''
    
    def __init__(self,quality):
        if quality=='good':
            self.atm = 0.8
        if quality=='bad':
            self.atm = 0.4
    def get_atm(self):
        atm = self.atm
        return atm
            
class detector:
    def __init__(self,camera_cost):
        self.camera_cost = camera_cost

    def detector_efficiency(self):   
    
        if self.camera_cost == '$$$':
            QE = 0.5
        if self.camera_cost == '$':
            QE = 0.08
        return QE
    
class Counts_Equation:
    '''
    PURPOSE:
        Creates a class representing observing conditions
        
    INPUTS:
        model_SED: SED class with given temperature and magnitude
        filter: filter class with bandpass argument
        telescope: telescope class with diameter argument
        atmosphere: class atmosphere with quality argument
        detector: class detector with camera_cost argument
        
    '''
    def __init__(self,SED_model,filter,telescope,atmosphere,detector):
        
        self.wavelength,self.photon_spectrum = SED_model.photon_spectrum()
        self.bandpass = BandPass(filter)
        self.convolved_spectrum = self.bandpass.convolve_SED(self.wavelength, self.photon_spectrum)
        
        
        self.area = telescope.calc_area()
        self.mirror = telescope.mirror()

        self.atm = atmosphere.get_atm()
        self.QE = detector.detector_efficiency()
        
    def get_counts(self):
        '''
        PURPOSE:
            calculate the expected counts per second given the class's attributes
        
        INPUTS:
            none
        
        RETURNS:
            counts, number in photons per second
        '''
        
        adjusted_spectrum = self.convolved_spectrum*self.area*self.atm*self.mirror*self.QE
        counts = simps(adjusted_spectrum, wavelength)
        return counts



# Initialize the SED class
model = SED(mag = 20, temperature = 9700, temp_units = 'Kelvin', wave_start = 1, wave_stop = 10000, wave_units = 'Angstrom')

# Create the spectrum
wavelength,spectrum = model.photon_spectrum()

# Initialize the BandPass class
Vband = BandPass('V')

# Convolve the spectrum with the filter response
convolved_spectrum = Vband.convolve_SED(wavelength, spectrum)






model_SED = SED(mag = 20, temperature = 9700, temp_units = 'Kelvin', wave_start = 1, wave_stop = 10000, wave_units = 'Angstrom')


timstelescope = Telescope(60,'cm') #create telescope


LasCruces = atmosphere('good') #create atmosphere conditions

camera = detector('$$$') #create detector



observation = Counts_Equation(model_SED,                              #initialize observation
                              filter='V',telescope=timstelescope,
                              atmosphere=LasCruces,detector=camera)

counts = observation.get_counts()

print(np.round(counts,2),'photons per second')

# Plot the convolved spectrum
plt.plot(wavelength,spectrum,label='photon spectrum')
plt.plot(Vband.filter_wavelength,Vband.filter*.00001,label='filter bandpass')
plt.plot(wavelength,convolved_spectrum,label='convolved spectrum')

plt.legend()
plt.show()




