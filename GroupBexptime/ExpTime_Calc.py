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
    def __init__(self, mag=None, temperature=9700,test_spectrum=None):
  
        # Define variables
        self.mag = mag
        self.temperature = temperature*u.K
        if (test_spectrum==None) or (test_spectrum==False):
            self.test_spectrum=False
        if test_spectrum==True:   
            self.test_spectrum = True
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
        ref_spectrum = BlackBody(temperature = 9700*u.K, scale = 1*u.erg/u.s/u.cm**2/u.sr/u.AA)

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
        if self.test_spectrum==False:        
            with warnings.catch_warnings():            
                warnings.filterwarnings("ignore", category=RuntimeWarning)          
                wavelength,normalized_spectrum = SED.blackbody_spectrum(self)       
                light_speed = c.c.to(u.cm/u.s)
                h = c.h.to(u.erg*u.s)
                photon_spectrum = normalized_spectrum*(wavelength*u.AA.to(u.cm)*u.cm)/(h*light_speed)
        
        if self.test_spectrum==True:
                wavelength = np.linspace(1, 15000, 100000)
                photon_spectrum = np.full(100000, 0.1)
        
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
        filter: filter used for observation, supported filter inputs are 'B','V','R','I','test
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
        interpolate = interp1d(self.filter_wavelength, self.filter_transmission,fill_value=0.0,bounds_error=False)
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
        mirror: name of mirror file. Options are '1.0_mirror' and '0.8_mirror'
            
    '''
    def __init__(self,diameter,diameter_units,focal_length,foc_len_units,mirror):
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
           
        self.mirror = pd.read_csv(notebook_directory+'/mirror_data/'+str(mirror)+'_mirror.csv')
        
    def calc_area(self):    
        area = (np.pi*(self.diameter/2)**2).to(u.cm**2)
        return area
   

    
class atmosphere:
    '''
    PURPOSE:
        A class which defines absorption due to atmospheric conditions
            and skyglow
        
    INPUTS:
        moon: Reads in sky table with corresponding moon illumination percent in 0.1 increments. 'test' has sky flux of 
                  100 photons/m^2/s/micron/arcsec^2 in all wavelengths and transmission of 1
        seeing: value in arcsec
    '''
    
    def __init__(self,moon,seeing):
        
        if moon=='test':
            sky_table = pd.read_csv(notebook_directory+'/sky_data/test.csv')
        else:
            sky_table = pd.read_csv(notebook_directory+'/sky_data/sky_'+str(moon)+'.csv')
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
    '''
    PURPOSE:
        creates a class representing detector and associated properties
        
    INPUTS:
        QE: quantum efficiency of detector. Either 0.8 or 1.0
        px_size: pixel size in microns
        read_noise: read noise in e-
        instrument_type: type of instrument. either "photometry" or "spectroscopy"
        dispersion: dispersion of spectrograph in angstroms per pixel
    '''
    def __init__(self,QE,px_size,read_noise,instrument_type='photometry',dispersion=None):

        self.QE = pd.read_csv(notebook_directory+'/detector_data/'+str(QE)+'_QE.csv')
        
        self.px_size = px_size*u.micron
        self.read_noise = read_noise #e/px
        if instrument_type == 'photometry':
            self.instrument_type = 'phot'
        if instrument_type == 'spectroscopy':
            self.instrument_type = 'spec'
            self.dispersion = dispersion
class Observation:
    '''
    PURPOSE:
        Creates a class representing observing conditions
        
    INPUTS:
        model_SED
        filter
        telescope
        atmosphere
        detector  
    '''
    def __init__(self,SED_model,filter,telescope,atmosphere,detector):
        
        self.wavelength,self.photon_spectrum = SED_model.photon_spectrum()
        self.bandpass = BandPass(filter)
        self.atmosphere = atmosphere
        self.area = telescope.calc_area()
        
        self.mirror_int = telescope.mirror
        interpolate = interp1d(self.mirror_int['Wavelength'], self.mirror_int['Reflection'],kind='cubic',fill_value=0.0,bounds_error=False)
        self.mirror = interpolate(self.wavelength) 
        
        self.detector = detector
        self.QE_int = detector.QE
        interpolate = interp1d(self.QE_int['Wavelength'], self.QE_int['Response'],kind='cubic',fill_value=0.0,bounds_error=False)
        self.QE = interpolate(self.wavelength) 
        
        
        self.seeing = atmosphere.seeing
        self.filtered_spectrum = self.bandpass.filter_SED(self.wavelength, self.photon_spectrum)
        self.absorbed_spectrum = self.atmosphere.absorption(self.wavelength,self.filtered_spectrum)
        self.adjusted_spectrum = self.absorbed_spectrum*self.area*self.mirror*self.QE

        
        self.px_scale = np.arctan(detector.px_size.to(u.mm)/telescope.focal_length.to(u.mm)).to(u.arcsec) #arcsecond per pixel
        
        
        self.filtered_skyglow = self.bandpass.filter_SED(atmosphere.lam, atmosphere.flux)

        if detector.instrument_type=='phot':
            self.N_px = (np.pi*self.seeing**2)/self.px_scale**2 #number of pixels due to scale and seeing
            self.skyglow_area = self.filtered_skyglow*(self.px_scale**2*self.N_px*self.area) 
            self.total_skyglow = simps(self.skyglow_area, atmosphere.lam) #photons per second
            
        if detector.instrument_type=='spec':
            self.skyglow_area = self.filtered_skyglow*(self.px_scale**2*self.area)
        
    def get_counts(self):
        '''
        PURPOSE:
            calculate the expected signal counts per second given the class's attributes
        
        INPUTS:
            none
        
        RETURNS:
            counts, number in photons per second
        '''
        
        counts = simps(self.adjusted_spectrum, self.wavelength)
        print(np.round(counts,2),'photons per second')
        return counts
    
    
class Signal_to_Noise:
    '''
    Class for calculating needed exposure time for observation
    
    INPUTS: 
        Counts_Equation, class with all observing factors used to calculate signal
        SNR: desired SNR
        wavelength: wavelength for spectroscopic observation in Angstroms
    '''
    def __init__(self,Counts_Equation,SNR,wavelength=None):
        self.SNR = SNR  
        self.sigma_rn = Counts_Equation.detector.read_noise
        
        if Counts_Equation.detector.instrument_type == 'phot': 
            self.counts = Counts_Equation.get_counts()
            self.N_px = Counts_Equation.N_px
            self.skyglow = Counts_Equation.total_skyglow
            
        if Counts_Equation.detector.instrument_type == 'spec':       
            self.wavelength = wavelength
            self.wavelength_array = Counts_Equation.wavelength
            self.dispersion = Counts_Equation.detector.dispersion
            self.spectrum = Counts_Equation.adjusted_spectrum
            
            index = np.where((self.wavelength_array>=self.wavelength-self.dispersion/2)&(self.wavelength_array<=self.wavelength+self.dispersion/2))
            self.counts = simps(self.spectrum[index],self.wavelength_array[index])
            self.N_px = 1
            self.skyglow_area = Counts_Equation.skyglow_area
            
            index2 = np.where((Counts_Equation.atmosphere.lam>=self.wavelength-self.dispersion/2)&(Counts_Equation.atmosphere.lam<=self.wavelength+self.dispersion/2))
            
            self.index2 = index2
            
            self.skyglow = simps(self.skyglow_area[index2], np.array(Counts_Equation.atmosphere.lam)[index2]) #photons per second
            

        
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
        print(str(np.round(time,2)),'seconds for SNR='+str(self.SNR))
        
        
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
    
    def calc_SNR(self,time,plot=False):
        '''
        PURPOSE:
            calculates SNR at given time 
            
        INPUTS:
            time: time in seconds
            plot: True or False
        RETURNS:
            snr: signal to noise at given exposure time
        '''
        def snr_t_eq(t):
                return (self.counts*t)/np.sqrt(self.counts*t+self.skyglow*t+self.N_px*self.sigma_rn**2)
            
        snr = snr_t_eq(time)
        
        if plot==True:  
            t = np.linspace(0,time,1000)
            with warnings.catch_warnings():            
                warnings.filterwarnings("ignore", category=RuntimeWarning)  
                snr_val = snr_t_eq(t)
            plt.plot(t,snr_val)
            plt.axvline(time,color='red',linestyle='dashed',label=str(np.round(time,2))+' seconds')
            plt.axhline(snr,color='black',linestyle='dashed',label='SNR='+str(snr))
            plt.xlabel('time (s)')
            plt.ylabel('SNR')
            plt.legend()
            plt.show()
        return snr