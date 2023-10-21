import astropy.units as u
import astropy.constants as c
from astropy.modeling.physical_models import BlackBody
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import warnings
import pandas as pd
import sys
import os
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from photutils.centroids import centroid_sources
from photutils.centroids import centroid_com
from scipy.interpolate import UnivariateSpline
notebook_directory = os.getcwd()
sys.path.append(notebook_directory)
import photometry_functions as pf

class SED:

    '''
    PURPOSE:
            Create an SED class to generate model spectrum.
        
    INPUTS:
            [wavelength; np.array, float]:  Wavelength array of the blackbody spectrum

    OPT.
    INPUTS:
                    [mag; float]:  Magnitude of source
            [temperature; float]:  Temperature of source (K)
    '''

    # Initialize the class
    def __init__(self, wavelength, mag=None, temperature=9_700, test_spectrum=None):
  
        # Define variables
        self.wavelength = wavelength
        self.mag = mag
        self.temperature = temperature*u.K

        # Whether or not to use a test spectrum
        if (test_spectrum == None) or (test_spectrum == False):
            self.test_spectrum = False
        if test_spectrum == True:   
            self.test_spectrum = True

    def blackbody_spectrum(self):

        '''
        PURPOSE:
                This function creates a blackbody spectrum based on the temperature and magnitude of the star.

        INPUTS:
                        [mag; float]:  The magnitude of the star
                [temperature; float]:  The temperature of the star

        OUTPUTS:
                          [wavelength; np.array, float]:  The wavelength array of the blackbody spectrum
                [normalized_spectrum,; np.array, float]:  The normalized blackbody spectrum

        AUTHOR:
                Tim M. Sept. 1 2023.
        '''

        # Create the blackbody spectrum
        blackbody_model = BlackBody(self.temperature, scale = 1*u.erg/u.s/u.cm**2/u.sr/u.AA)

        # Define the reference spectrum
        self.model_spectrum = blackbody_model(self.wavelength)
        
        # Normalize the spectrum
        ref_spectrum = BlackBody(temperature = 9_700*u.K, scale = 1*u.erg/u.s/u.cm**2/u.sr/u.AA)

        # Normalize the spectrum
        normalize = (3.63e-9*u.erg/u.cm**2/u.s/u.AA)/(ref_spectrum(5_500*u.AA)*u.sr) 

        # Define the normalized spectrum
        self.normalized_spectrum = self.model_spectrum*u.sr*normalize*10.**(-0.4*self.mag)
        
        return self.wavelength, self.normalized_spectrum
    
    def photon_spectrum(self, plot=False):

        '''
        PURPOSE:
                Create a photon spectrum.

        INPUTS:
                        [mag; float]: The magnitude of the star
                [temperature; float]: The temperature of the star

        OUTPUTS:
                     [wavelegnth; np.array, float]: Wavelength array of the photon spectrum
                [photon_spectrum; np.array, float]: Photons per second per cm^2 per Angstrom
        '''

        # Check if the test spectrum is being used
        if self.test_spectrum == False:

            # Warning filter   
            with warnings.catch_warnings():            
                warnings.filterwarnings("ignore", category = RuntimeWarning)

                # Create the blackbody spectrum    
                wavelength, normalized_spectrum = SED.blackbody_spectrum(self)

                # Convert the spectrum to photons per second per cm^2 per Angstrom   
                light_speed = c.c.to(u.cm/u.s)
                h = c.h.to(u.erg*u.s)
                photon_spectrum = normalized_spectrum*(wavelength*u.AA.to(u.cm)*u.cm)/(h*light_speed)
        
        # If the test spectrum is being used
        if self.test_spectrum == True:
                
                # Define the photon spectrum
                photon_spectrum = np.full(100_000, 0.1)
        
        # Create spectrum plot
        if plot == True:
            fig, ax = plt.subplots(1, 1, figsize = (10, 8))
            ax.set_xlabel('Wavelength ($\AA$)')
            ax.set_ylabel('Flux Density (photons $s^{-1} cm^{-2} \AA^{1})$')
            ax.plot(wavelength, photon_spectrum)
            fig.show()
        
        return wavelength, photon_spectrum
    
class BandPass:

    '''
    PURPOSE: 
            Create a class for bandpass for a given filter.
    
    INPUTS:
            [filter; string]:  Filter used for observation, supported filter inputs are 'B','V','R','I','test
    '''

    # Initialize the bandpass class
    def __init__(self, filter):

        # For the Johnson filter system:
        filter_table = pd.read_csv(notebook_directory + '/filter_data/' + filter + '.dat', delimiter = '\t')

        # Define the filter wavelength and transmission
        self.filter_wavelength = filter_table['wavelength']
        self.filter_transmission = filter_table['transmission']
            
    def filter_SED(self, wavelength, normalized_spectrum, plot=False):

        ''' 
        PURPOSE:
                Function that mutliplies the normalized spectrum with a filter response function.

        INPUTS:
                         [wavelength; np.array, float]:  Wavelength of spectrum 
                [normalized_spectrum; np.array, float]:  Normalized spectrum 

        OUTPUTS:
                  [filtered_spectrum; np.array, float]:  Normalized spectrum multiplied with the filter response.

        AUTHOR:
                Tim M. Sept. 1 2023.
        '''

        # Interpolate the filter response
        interpolate = interp1d(self.filter_wavelength, self.filter_transmission, fill_value = 0.0, bounds_error = False)
        self.interpolated_filter_values = interpolate(wavelength)/100

        # Multiply the normalized spectrum by the interpolated filter values
        filtered_spectrum = normalized_spectrum*self.interpolated_filter_values
        
        # Plot the spectrum
        if (plot == True):

            # Calculate the maximum value of the spectrum
            max = normalized_spectrum.max()

            # Make the plot
            fig, ax = plt.subplots(1, 1, figsize = (8, 6))
            ax.set_xlabel('Wavelength ($\AA$)')
            ax.set_ylabel('Flux Density (photons $s^{-1} cm^{-2} \AA^{1})$')
            ax.plot(wavelength, normalized_spectrum, label = 'Photon Spectrum')
            ax.plot(wavelength, self.interpolated_filter_values*max, label ='Filter')
            ax.plot(wavelength, filtered_spectrum, label = 'Filtered Spectrum')
            ax.legend()
            fig.show()
             
        return filtered_spectrum
    
class Telescope:
    
    '''
    PURPOSE:
            Define telescope parameters to determine collecting area

    INPUTS:
                   [diameter; float]:  Diameter of primary mirror
            [diameter_units; string]:  Units of length for diameter
               [focal_length; float]:  Focal length of telescope
             [foc_len_units; string]:  Units of length for focal length
                    [mirror; string]:  Name of mirror file. Options are '1.0_mirror' and '0.8_mirror'
    '''

    # Initialize the telescope class
    def __init__(self, diameter, diameter_units, focal_length, foc_len_units, mirror):
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
        
        # Read in mirror data
        self.mirror = pd.read_csv(notebook_directory + '/mirror_data/' + str(mirror) + '_mirror.csv')
        
    def calc_area(self):   

        '''
        PURPOSE:
                Calculate the area of the telescope
        
        INPUTS:
                [diameter; float]:  Diameter of primary mirror

        OUTPUTS:
                [area; float]:  Area of telescope in cm^2
        '''

        # Calculate the area of the telescope
        area = (np.pi*(self.diameter/2)**2).to(u.cm**2)

        return area
    
class Atmosphere:

    '''
    PURPOSE:
            A class which defines absorption due to atmospheric conditions and skyglow
        
    INPUTS:
             [moon; string]:  Reads in sky table with corresponding moon illumination percent in 0.1 increments. 'test' has sky flux of 100 photons/m^2/s/micron/arcsec^2 in all wavelengths and transmission of 1
            [seeing; float]:  Value in arcsec
    '''
    
    # Initialize the atmosphere class
    def __init__(self, moon):
        
        # Read in sky data
        if moon == 'test':
            sky_table = pd.read_csv(notebook_directory + '/sky_data/test.csv')
        else:
            sky_table = pd.read_csv(notebook_directory + '/sky_data/sky_' + str(moon) + '.csv')

        # Define the wavelength, flux, and transmission
        self.lam = sky_table['lam']*u.nm.to(u.AA)
        self.flux = (np.array(sky_table['flux'])*u.photon/u.m**2/u.s/u.micron/u.arcsec**2).to(u.photon/u.cm**2/u.s/u.AA/u.arcsec**2)
        self.trans = sky_table['trans']
            
    def absorption(self, wavelength, spectrum):

        '''
        PURPOSE:
                Multiply spectrum times absorption of atmosphere.
    
        INPUTS:
                        [wavelength; np.array, float]:  Wavelength of spectrum
               [normalized_spectrum; np.array, float]:  Normalized spectrum

        RETURNS:
                 [absorbed_spectrum; np.array, float]:  Normalized spectrum multiploed by atmospheric transmission.
        '''

        # Interpolate the filter response
        interpolate = interp1d(self.lam, self.trans, kind = 'cubic', fill_value = 0.0, bounds_error = False)

        # Multiply the normalized spectrum by the interpolated filter values
        self.interpolated_absorption_values = interpolate(wavelength)

        # Multiply the normalized spectrum by the interpolated filter values
        self.absorbed_spectrum = spectrum*self.interpolated_absorption_values

        return self.absorbed_spectrum
    
    class Seeing:

        # Initialize the class
        def __init__(self, data, sources, header, filter):

            # Define the atrributes
            self.data = data
            self.sources = sources
            self.header = header
            self.filter = filter

        def GetFWHM(self, makeplot=False):

            '''
            PURPOSE:
                    This function calculates the FWHM of the data.

            INPUTS:
                    [data; np.array, float]:  The data to be bias subtracted.
                    [sources; np.array, float]:  The sources found in the data.
                            [makeplot; bool]:  Whether or not to make a plot.

            OUTPUTS:
                    [median_FWHM; float]: The median FWHM of the data.
            '''

            # Get the background
            mean, median, std = sigma_clipped_stats(self.data, sigma = 3.0)

            # Set the background
            background = median
            
            # Get the x and y coordinates
            xlist = (np.array(self.sources['xcentroid']))
            ylist = (np.array(self.sources['ycentroid']))
            
            # Get the centroids
            xlist, ylist = centroid_sources(self.data, xlist, ylist, box_size = 25, centroid_func = centroid_com)
            
            # Make empty array to store FWHMs
            FWHM = []

            # Loop over sources
            for ID in range(len(xlist)):

                # Get the x and y centers
                xcenter, ycenter = xlist[ID], ylist[ID]
                
                try:

                    # Get the horizontal and vertical profiles
                    horizontal, vertical = Atmosphere.Seeing.profiles(self.data, xcenter, ycenter)

                    # Interpolate the width     
                    r1h, r2h, fwhm_x = Atmosphere.Seeing.interpolate_width(horizontal, background)
                    r1v, r2v, fwhm_y = Atmosphere.Seeing.interpolate_width(vertical, background)

                    # Get the FWHM         
                    fwhm_inst = np.mean([fwhm_x, fwhm_y])

                    # Append the FWHM
                    FWHM = np.append(FWHM, fwhm_inst)
                    

                    # Check if plot should be made
                    if makeplot == True:

                        # Make range for x axis
                        xx = np.arange(0, len(vertical), 1)

                        # Get the half max
                        hor_half = ((horizontal - background).max())/2
                        ver_half = ((vertical - background).max())/2                    

                        # Make plot
                        fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize = (10, 4))
                        
                        ax1.imshow(self.data[int(ycenter) - 25:int(ycenter) + 25, int(xcenter) - 25:int(xcenter) + 25], norm = LogNorm(), origin = 'lower')
                        ax1.axvline(25, color = 'red', linestyle = 'dashed', alpha = 0.8)
                        ax1.axhline(25, color = 'red', linestyle = 'dashed', alpha = 0.8)
                        
                        ax2.plot(xx,vertical - background)
                        ax2.hlines(ver_half, color = 'red', xmin = r1v, xmax = r2v)
                        ax2.set_title('Vertical Cross Section')
                        ax2.set_xlabel('FWHM= ' + str(np.round(fwhm_y, 2)))
                        
                        ax3.plot(xx,horizontal - background)
                        ax3.set_title('Horizontal Cross Section')
                        ax3.hlines(hor_half, color = 'red', xmin = r1h, xmax = r2h)
                        ax3.set_title('Horizontal Cross Section')
                        ax3.set_xlabel('FWHM= ' + str(np.round(fwhm_x, 2)))                
                        plt.show()    
                        #plt.savefig('FWHM example.pdf',bbox_inches='tight',facecolor='white')
                except:
                    pass

            # Get the median FWHM 
            median_FWHM = np.median(FWHM)

            # Make histogram and save
            plt.hist(FWHM, bins = 'auto')
            plt.axvline(median_FWHM, color = 'red', linestyle = 'dashed')
            plt.xlabel(str(len(FWHM)) + ' FWHMs calculated \n Median FWHM: ' + str(np.round(median_FWHM, 2)), size = 13)
            plt.show()
            print('Median FWHM: ' + str(np.round(median_FWHM, 2)))
            
            return median_FWHM
        
        def profiles(data, xcenter, ycenter):

            '''
            PURPOSE:
                    This function returns the horizontal and vertical profiles of the data.

            INPUTS:
                    [data; np.array, float]:  The data to be bias subtracted.
                        [xcenter; float]: The x coordinate of the star.
                        [ycenter; float]: The y coordinate of the star.

            OUTPUTS:
                    [x; np.array, float]: The horizontal profile.
                    [y; np.array, float]: The vertical profile.
            '''

            # Get the horizontal profile
            ypix, xpix = ycenter, xcenter

            # Get the vertical profile
            x = np.take(data, ypix, axis = 0)[int(xcenter) - 25:int(xcenter) + 25]
            y = np.take(data, xpix, axis = 1)[int(ycenter) - 25:int(ycenter) + 25]

            return x, y

        def interpolate_width(axis, background):

            '''
            PURPOSE:
                    This function interpolates the width of the data.

            INPUTS:
                    [axis; np.array, float]:  The horizontal or vertical profile.
                        [background; float]:  The background level of the data.

            OUTPUTS:
                        [r1; float]: The left root.
                        [r2; float]: The right root.
                    [r2 - r1; float]: The FWHM.
            '''

            # Get the peak
            peak = axis.max()

            # Get the half max
            half_max = (peak - background)/2

            # Get the x values
            x = np.linspace(0, len(axis), len(axis))

            # Do the interpolation
            spline = UnivariateSpline(x, axis - background - half_max, s = 0)

            # Get the roots
            r1, r2 = spline.roots()

            return r1, r2, r2 - r1
        
        def seeing(self):

            '''
            PURPOSE:
                    Estimate seeing from FWHM.

            INPUTS:
                [data; np.array, float]:  Data to be used for seeing estimate.

            OUTPUTS:
                    [seeing; float]:  Seeing estimate in arcsec.
            '''

            # Caluluate the median FWHM
            self.median_fwhm = Atmosphere.Seeing.GetFWHM(self)

            # Get the pixel scale of the image
            wcs = WCS(self.header)

            # Get pixel scale in arcsec
            px_scale = wcs.proj_plane_pixel_scales()[0].to(u.arcsec)

            # Get the seeing
            self.seeing = self.median_fwhm*px_scale
            print('FWHM:', self.median_fwhm,'pixels')
            print('Seeing:', self.seeing)
            
            return self.seeing
    
class Detector:

    '''
    PURPOSE:
            Creates a class representing detector and associated properties.
        
    INPUTS:
                      [QE; float]:  Quantum efficiency of detector. Either 0.8 or 1.0
                 [px_size; float]:  Pixel size in microns
              [read_noise; float]:  Read noise in e-
        [instrument_type; string]:  Type of instrument. Either "photometry" or "spectroscopy"
              [dispersion; float]:  Dispersion of spectrograph in angstroms per pixel
    '''

    # Initialize the detector class
    def __init__(self, QE, px_size, read_noise, instrument_type='photometry', dispersion=None):

        # Read in QE data
        self.QE = pd.read_csv(notebook_directory + '/detector_data/' + str(QE) + '_QE.csv')
        
        # Define the pixel size and read noise
        self.px_size = px_size*u.micron
        self.read_noise = read_noise #e/px

        # Define the instrument type
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
            [SED_model; class]:  SED class
                 [filter; str]:  Filter used for observation, supported filter inputs are 'B','V','R','I','test
            [telescope; class]:  Telescope class
           [atmosphere; class]:  Atmosphere class
             [detector; class]:  Detector class
    '''

    # Initialize the observation class
    def __init__(self, SED_model, filter, telescope, atmosphere, seeing, detector):
        
        # Define the attributes
        self.wavelength,self.photon_spectrum = SED_model.photon_spectrum()
        self.bandpass = BandPass(filter)
        self.atmosphere = atmosphere
        self.area = telescope.calc_area()
        self.mirror_int = telescope.mirror

        # Interpolate the mirror
        interpolate = interp1d(self.mirror_int['Wavelength'], self.mirror_int['Reflection'], kind = 'cubic', fill_value = 0.0, bounds_error = False)
        self.mirror = interpolate(self.wavelength) 
        self.detector = detector
        self.QE_int = detector.QE

        # Interpolate the QE
        interpolate = interp1d(self.QE_int['Wavelength'], self.QE_int['Response'], kind = 'cubic', fill_value = 0.0, bounds_error = False)

        # Multiply the normalized spectrum by the interpolated filter values
        self.QE = interpolate(self.wavelength) 
        self.seeing = seeing
        self.filtered_spectrum = self.bandpass.filter_SED(self.wavelength, self.photon_spectrum)
        self.absorbed_spectrum = self.atmosphere.absorption(self.wavelength, self.filtered_spectrum)
        self.adjusted_spectrum = self.absorbed_spectrum*self.area*self.mirror*self.QE

        # Calculate the pixel scale
        self.px_scale = np.arctan(detector.px_size.to(u.mm)/telescope.focal_length.to(u.mm)).to(u.arcsec) #arcsecond per pixel
        
        # Calculate the skyglow
        self.filtered_skyglow = self.bandpass.filter_SED(atmosphere.lam, atmosphere.flux)

        # Check if the instrument is photometric
        if detector.instrument_type == 'phot':

            # Calculate the number of pixels
            self.N_px = (np.pi*self.seeing**2)/self.px_scale**2 #number of pixels due to scale and seeing
            self.skyglow_area = self.filtered_skyglow*(self.px_scale**2*self.N_px*self.area) 
            self.total_skyglow = simps(self.skyglow_area, atmosphere.lam) #photons per second
            
        # Check if the instrument is spectroscopic
        if detector.instrument_type == 'spec':
            self.skyglow_area = self.filtered_skyglow*(self.px_scale**2*self.area)
        
    def get_counts(self):

        '''
        PURPOSE:
                Calculate the expected signal counts per second given the class's attributes
        
        INPUTS:
                [adjusted_spectrum; np.array, float]:  Adjusted spectrum
                       [wavelegnth; np.array, float]:  Wavelength array of the adjusted spectrum
        
        RETURNS:
                [counts; float]:  Counts per second
        '''
        
        # Calculate the counts
        counts = simps(self.adjusted_spectrum, self.wavelength)

        # Print the counts
        print(np.round(counts, 2),'photons per second')

        return counts

    class Throughput:

        # Initialize the class
        def __init__(self, data, header, aperture, standard_spec, filter, filter_response, standard_spec_table, fwhm):

            # Define the atrributes
            self.data = data
            self.header = header
            self.area = np.pi*(aperture/2)**2
            self.standard_spec = standard_spec
            self.filter = filter
            self.filter_response = filter_response
            self.standard_spec_table = standard_spec_table
            self.fwhm = fwhm

            # Extract magnitude from standard
            self.standard_M = standard_spec_table['V'][0]

        def throughput(self):

            '''
            PURPOSE:
                    Calculate the throughput of the instrument.

            INPUTS:
                    [actual_flux; float]:  Actual flux of the star.
                    [standard_flux; float]:  Standard flux of the star.

            OUTPUTS:
                    [throughput; float]:  Throughput of the instrument.
            '''

            # Convert to counts
            h = 6.626e-27 # ergs/s
            c = 3e10 # cm/s

            # Define the wavelength and magnitude from the spectrum
            wavelength = self.standard_spec[0]
            mag = self.standard_spec[1]

            # Convert magnitude to flux
            flux = 10**(-0.4*(mag + 48.6))

            # Get the filter response properties
            filter_wavelength = self.filter_response[0]
            filter_function   = self.filter_response[1]

            # Interpolate the filter response to the wavelength of the spectrum
            interpolate = interp1d(filter_wavelength, filter_function, fill_value = 0.0, bounds_error = False)
            interpolated_filter_response = interpolate(wavelength)

            # Filter the flux
            filtered_flux = flux*interpolated_filter_response
            
            # counts in ergs/s/cm^2/Hz
            photon_flux_nu = (filtered_flux*wavelength)/(h*c)
            wavelength = wavelength
            
            # Convert to ergs/s/cm^2/Angstrom
            conversion = 2.998e10/(wavelength**2)
        
            # counts in ergs/s/cm^2/Angstrom
            photon_flux = photon_flux_nu*conversion
            
            # Convert to photons per second
            true_count_flux = simps(photon_flux, wavelength) * self.area
            
            # Extract source positions
            x, y = self.standard_spec_table['X'], self.standard_spec_table['Y']

            # Define the source positions
            source_positions = np.column_stack((x, y))

            # Measure the observed count flux
            self.phot = Photometry.ap_phot(self, source_positions)
            
            # Extract the photometry
            observed_count_flux = self.phot[self.filter + ' counts_sec'][0]
            
            # Calculate the throughput
            self.throughput = observed_count_flux/true_count_flux
            
            print('Expected Counts:', true_count_flux)
            print('Observed_Counts:', observed_count_flux)
            print('Throughput:', self.throughput)
            
            return self.throughput
    
class Signal_to_Noise:

    '''
    PURPOSE:
            Class for calculating needed exposure time for observation
    
    INPUTS:
            [Counts_Equation; class]:  Class with all observing factors used to calculate signal
                        [SNR; float]:  Desired SNR
                 [wavelegnth; array]:  Wavelength array for spectroscopic observation in Angstroms
    '''

    # Initialize the signal to noise class
    def __init__(self, Counts_Equation, SNR, wavelength=None):

        # Define the attributes
        self.SNR = SNR  
        self.sigma_rn = Counts_Equation.detector.read_noise
        
        # Check if the instrument is photometric
        if Counts_Equation.detector.instrument_type == 'phot':

            # Define the attributes
            self.counts = Counts_Equation.get_counts()
            self.N_px = Counts_Equation.N_px
            self.skyglow = Counts_Equation.total_skyglow
        
        # Check if the instrument is spectroscopic
        if Counts_Equation.detector.instrument_type == 'spec':

            # Define the attributes
            self.wavelength = wavelength
            self.wavelength_array = Counts_Equation.wavelength
            self.dispersion = Counts_Equation.detector.dispersion
            self.spectrum = Counts_Equation.adjusted_spectrum
            
            # Find the indices where the wavelength is within the dispersion
            index = np.where((self.wavelength_array >= self.wavelength - self.dispersion/2) & (self.wavelength_array <= self.wavelength + self.dispersion/2))

            # Calculate the counts
            self.counts = simps(self.spectrum[index],self.wavelength_array[index])
            self.N_px = 1

            # Calculate the skyglow
            self.skyglow_area = Counts_Equation.skyglow_area
            
            # Find the indices where the wavelength is within the dispersion
            index2 = np.where((Counts_Equation.atmosphere.lam >= self.wavelength - self.dispersion/2) & (Counts_Equation.atmosphere.lam <= self.wavelength + self.dispersion/2))
            
            # Define the skyglow
            self.index2 = index2
            self.skyglow = simps(self.skyglow_area[index2], np.array(Counts_Equation.atmosphere.lam)[index2]) # Photons per second
        
    def calc_exptime(self, plot=False):

        '''
        PURPOSE:
                Calculates needed exposure time for desired SNR.

        INPUTS:
                  [counts; float]:  Counts per second
                 [skyglow; float]:  Skyglow per second
                      [N_px; int]:  Number of pixels
                [sigma_rn; float]:  Read noise in e-
                     [SNR; float]:  Desired SNR
                     [plot; bool]:  True or False

        RETURNS:
                [time; float]:  Exposure time in seconds

        NOTES:
                Currently only factors in counts uncertainty
        '''

        # Define the exposure time
        def snr_eq(t):

            '''
            PURPOSE:
                    Calculates SNR at given exposure time

            INPUTS:
                      [t; float]:  Exposure time in seconds

            RETURNS:
                    [snr; float]:  Signal to noise at given exposure time
            '''

            return (self.counts*t)/np.sqrt(self.counts*t+self.skyglow*t+self.N_px*self.sigma_rn**2)-self.SNR 
        
        # Find the roots of the SNR equation
        roots = fsolve(snr_eq, 0.1)

        # Extract the exposure time
        time = roots[0]

        # Print the exposure time
        print(' ')
        print(str(np.round(time,2)),'seconds for SNR='+str(self.SNR))
        
        # Check if the plotting
        if (plot == True):

            # Define the time array  
            t = np.linspace(0, 2*time, 1_000)

            # Filter the warnings
            with warnings.catch_warnings():            
                warnings.filterwarnings("ignore", category = RuntimeWarning)  

                # Calculate the SNR
                snr_val = snr_eq(t)+self.SNR
            
            # Plot the SNR
            plt.plot(t,snr_val)
            plt.axvline(time, color = 'red', linestyle = 'dashed', label = str(np.round(time, 2)) + ' seconds')
            plt.axhline(self.SNR, color = 'black', linestyle = 'dashed', label = 'SNR=' + str(self.SNR))
            plt.xlabel('time (s)')
            plt.ylabel('SNR')
            plt.legend()
            plt.show()
        
        return time
    
    def calc_SNR(self, time, plot=False):

        '''
        PURPOSE:
                Calculates SNR at given time 
            
        INPUTS:
                [counts; float]:  Counts per second
               [skyglow; float]:  Skyglow per second
                    [N_px; int]:  Number of pixels
              [sigma_rn; float]:  Read noise in e-
                   [SNR; float]:  Desired SNR
                  [time; float]:  Time in seconds
                   [plot; bool]:  True or False

        RETURNS:
                [snr; float]:  Signal to noise at given exposure time
        '''
        
        # Calculate the SNR given the time
        snr = (self.counts*time)/np.sqrt(self.counts*time + self.skyglow*time + self.N_px*self.sigma_rn**2)
        
        # Check if the plotting
        if plot == True:

            # Define the time array
            t = np.linspace(0, time, 1_000)

            # Filter the warnings
            with warnings.catch_warnings():            
                warnings.filterwarnings("ignore", category = RuntimeWarning)

                # Calculate the SNR
                snr_val = (self.counts*t)/np.sqrt(self.counts*t+self.skyglow*t+self.N_px*self.sigma_rn**2)

            # Make the plot
            plt.plot(t, snr_val)
            plt.axvline(time, color = 'red', linestyle = 'dashed', label = str(np.round(time, 2)) + ' seconds')
            plt.axhline(snr, color = 'black', linestyle = 'dashed', label = 'SNR=' + str(snr))
            plt.xlabel('time (s)')
            plt.ylabel('SNR')
            plt.legend()
            plt.show()

        return snr