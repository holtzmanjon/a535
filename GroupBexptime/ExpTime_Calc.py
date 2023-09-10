import astropy.units as u
import astropy.constants as c
from astropy.modeling.physical_models import BlackBody
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps,trapz
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import warnings
import pandas as pd
import glob
import sys
import os
notebook_directory = os.getcwd()
sys.path.append(notebook_directory)

class SED:
    '''
    PURPOSE:
        Create an SED class to generate model spectrum
        
    INPUTS:
        mag: magnitude of source
        temperature: temperature of source
    '''
    # Initialize the class
    def __init__(self, mag, temperature):
  
        # Define variables
        self.mag = mag
        self.temperature = temperature*u.K

    def blackbody_spectrum(self):

        '''
        PURPOSE:
                This function creates a blackbody spectrum based on the temperature and magnitude of the star.

        INPUTS:
                        [mag; float]: The magnitude of the star
                [temperature; float]: The temperature of the star

        OUTPUTS:
                          [wavelength; np.array, float]: The wavelength array of the blackbody spectrum
                [normalized_spectrum,; np.array, float]: The normalized blackbody spectrum

        AUTHOR:
                Tim M. Sept. 1 2023.
        '''

        # Create the blackbody spectrum
        blackbody_model = BlackBody(self.temperature, scale = 1*u.erg/u.s/u.cm**2/u.sr/u.AA)

        # Normalize the spectrum
        self.wavelength = np.linspace(1, 15000, 100000)

        # Define the reference spectrum
        self.model_spectrum = blackbody_model(self.wavelength)
        
        # Normalize the spectrum
        ref_spectrum = BlackBody(temperature = 9_700*u.K, scale = 1*u.erg/u.s/u.cm**2/u.sr/u.AA)

        # Normalize the spectrum
        normalize = (3.63e-9*u.erg/u.cm**2/u.s/u.AA)/(ref_spectrum(5500*u.AA)*u.sr) 

        # Define the normalized spectrum
        self.normalized_spectrum = self.model_spectrum*u.sr*normalize*10.**(-0.4*self.mag)
        
        return self.wavelength, self.normalized_spectrum
    
    def photon_spectrum(self,plot=False):
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
            photon_spectrum = normalized_spectrum*(wavelength*u.AA.to(u.cm)*u.cm)/(h*light_speed)
        
        # Create spectrum plot
        if plot==True:
            fig,ax = plt.subplots(1, 1, figsize = (10, 8))
            ax.set_xlabel('Wavelength ($\AA$)')
            ax.set_ylabel('Flux Density (photons $s^{-1} cm^{-2} \AA^{1})$')
            ax.plot(wavelength, photon_spectrum)
            plt.show()
        
        return wavelength,photon_spectrum
    
class BandPass:
    '''
    PURPOSE: Create a class for bandpass
    
    INPUTS:
        filter: filter used for observation, supported filter inputs are 'B','V','R','I'
    '''
    # Initialize the bandpass class
    def __init__(self, filter):

        # For the Johnson filter system:
        filter_table = pd.read_csv(notebook_directory+'/filter_data/'+filter+'.dat',delimiter='\t')

        self.filter_wavelength = filter_table['wavelength']

        self.filter_transmission = filter_table['transmission']
            
    def filter_SED(self, wavelength, normalized_spectrum,plot=False):

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
        interpolate = interp1d(self.filter_wavelength, self.filter_transmission,kind='cubic',fill_value=0.0,bounds_error=False)
        self.interpolated_filter_values = interpolate(wavelength)/100 
        # Multiply the normalized spectrum by the interpolated filter values
        filtered_spectrum = normalized_spectrum*self.interpolated_filter_values
        
        
        if plot==True:
            fig,ax = plt.subplots(1, 1, figsize = (8, 6))
            max = normalized_spectrum.max()
            
            ax.set_xlabel('Wavelength ($\AA$)')
            ax.set_ylabel('Flux Density (photons $s^{-1} cm^{-2} \AA^{1})$')
            ax.plot(wavelength, normalized_spectrum,label='Photon Spectrum')
            ax.plot(wavelength, self.interpolated_filter_values*max,label='Filter')
            ax.plot(wavelength, filtered_spectrum, label= 'Filtered Spectrum')
            plt.legend()
            plt.show()
             
        return filtered_spectrum
    
class Telescope:
    '''
    PURPOSE:
        Define telescope parameters to determine collecting area
    INPUTS:
        diameter: diameter of primary mirror
        units: units of length for diameter
        
            
    '''
    def __init__(self,diameter,diameter_units,focal_length,foc_len_units):
        if diameter_units == 'cm':
            self.diameter = diameter*u.cm
        if foc_len_units == 'cm':
            self.focal_length = focal_length*u.cm
        if diameter_units == 'mm':
            self.diameter = diameter*u.mm
        if foc_len_units == 'mm':
            self.focal_length = focal_length*u.mm
        if diameter_units == 'm':
            self.diameter = diameter*u.m
        if foc_len_units == 'm':
            self.focal_length = focal_length*u.m
            
    def calc_area(self):    
        area = np.pi*(self.diameter/2)**2
        return area
   
    def mirror(self):
            mirror_efficiency = 1.0
            return mirror_efficiency
    
class atmosphere:
    '''
    PURPOSE:
        A class which defines absorption due to atmospheric conditions
            and skyglow
        
    INPUTS:
        quality: 'default' reads in sky.csv and uses those parameters
        seeing: value in arcsec
    '''
    
    def __init__(self,quality,seeing):
        if quality=='default':
            sky_table = pd.read_csv(notebook_directory+'/sky_data/sky.csv')
            self.lam = sky_table['lam']*u.nm.to(u.AA)
            self.flux = (np.array(sky_table['flux'])*u.photon/u.m**2/u.s/u.micron/
                         u.arcsec**2).to(u.photon/u.cm**2/u.s/u.AA/u.arcsec**2)
            self.trans = sky_table['trans']
            self.seeing = seeing*u.arcsec
            
    def absorption(self,wavelength,spectrum):
        '''
        multiply spectrum times absorption of atmosphere
    
        INPUTS:
        wavelength: wavelength array of spectrum
        spectrum: spectrum of object
        RETURNS: 
        absorbed_spectrum
        '''
        # Interpolate the filter response
        interpolate = interp1d(self.lam, self.trans,kind='cubic',fill_value=0.0,bounds_error=False)
        self.interpolated_absorption_values = interpolate(wavelength) 
        # Multiply the normalized spectrum by the interpolated filter values
        absorbed_spectrum = spectrum*self.interpolated_absorption_values
        return absorbed_spectrum
    
        
        
    
class detector:
    def __init__(self,camera_cost):
        self.camera_cost = camera_cost  
        if self.camera_cost == '$$$':
            self.QE = 0.5
            self.resolution = np.array([1000,1000]) #pixels
            self.px_size = 9*u.micron
            self.read_noise = 5 #e/px
        if self.camera_cost == '$':
            self.QE = 0.08
            self.resolution = np.array([500,500])
            self.px_size = 15*u.micron
            self.read_noise = 20 #e/px
            
class Observation:
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
        self.atmosphere = atmosphere
        self.area = telescope.calc_area()
        self.mirror = telescope.mirror()
        self.detector = detector
        self.QE = detector.QE
        self.seeing = atmosphere.seeing
        self.filtered_spectrum = self.bandpass.filter_SED(self.wavelength, self.photon_spectrum)
        self.absorbed_spectrum = self.atmosphere.absorption(self.wavelength,self.filtered_spectrum)
        
        self.px_scale = np.arctan(detector.px_size.to(u.mm)
                               /telescope.focal_length.to(u.mm)).to(u.arcsec)**2 #arcsecond^2 per pixel

        self.N_px = (np.pi*self.seeing**2)/self.px_scale

        
        self.filtered_skyglow = self.bandpass.filter_SED(atmosphere.lam, atmosphere.flux)*u.arcsec**2/self.px_scale*self.N_px*self.area
        self.total_skyglow = simps(self.filtered_skyglow, atmosphere.lam) #photons per second
        
    def get_counts(self):
        '''
        PURPOSE:
            calculate the expected counts per second given the class's attributes
        
        INPUTS:
            none
        
        RETURNS:
            counts, number in photons per second
        '''
        
        adjusted_spectrum = self.absorbed_spectrum*self.area*self.mirror*self.QE
        counts = simps(adjusted_spectrum, self.wavelength)
        print(np.round(counts,2),'photons per second')
        return counts
    
    
class Signal_to_Noise:
    '''
    Class for calculating needed exposure time for observation
    
    INPUTS: 
        Counts_Equation, class with all observing factors used to calculate signal
        SNR: desired SNR
    '''
    def __init__(self,Counts_Equation,SNR):
        self.counts = Counts_Equation.get_counts()
        self.SNR = SNR
        self.skyglow = Counts_Equation.total_skyglow
        self.N_px = Counts_Equation.N_px
        self.sigma_rn = Counts_Equation.detector.read_noise
        
    def calc_exptime(self,plot=False):
        '''
        Calculates needed exposure time for desired SNR
        INPUTS:
            plot: to plot or not to plot SNR as function of time
        RETURNS:
            time: time in seconds to reach desired SNR
        NOTES:
            currently only factors in counts uncertainty
        
        '''
        def snr_eq(t):
            return (self.counts*t)/np.sqrt(self.counts*t+self.skyglow*t+self.N_px*self.sigma_rn**2)-self.SNR 
        roots = fsolve(snr_eq, 0.1)
        time = roots[0]
        print(' ')
        print(str(time),'seconds for SNR='+str(self.SNR))
        
        
        if plot==True:  
            t = np.linspace(0,time*2,1000)
            with warnings.catch_warnings():            
                warnings.filterwarnings("ignore", category=RuntimeWarning)  
                snr_val = snr_eq(t)+self.SNR
            plt.plot(t,snr_val)
            plt.axvline(time,color='red',linestyle='dashed',label=str(np.round(time,2))+' seconds')
            plt.axhline(self.SNR,color='black',linestyle='dashed',label='SNR='+str(self.SNR))
            plt.xlabel('time (s)')
            plt.ylabel('SNR')
            plt.legend()
            plt.show()
        
        return time