from astropy.io import fits
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import astropy.units as u
from photutils.centroids import centroid_com
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.detection import DAOStarFinder
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from photutils.utils import calc_total_error
from astropy.stats import sigma_clip
from astropy.table import Table
import pandas as pd
import uncertainties
from matplotlib.colors import LogNorm
from scipy.interpolate import UnivariateSpline
from astropy.stats import sigma_clipped_stats
from scipy.interpolate import interp1d
from photutils.centroids import centroid_sources
from scipy.integrate import simps
import astroalign as aa
from astropy.wcs import WCS
from IPython.display import display

class DataReduction:

    # Initialize the class
    def __init__(self, files, filter):

        # Define the atrributes
        self.files = files
        self.filter = filter

    def median_combine(self):

        '''
        PURPOSE:
                Create median combined image from list of files

        INPUTS:
                [files; list]:  List of directories to fits files. First image is used as reference for alignment

        OUTPUTS:
                [median_image]:  Data for median combined image
                     [header0]:  Header of first reference image
        '''

        # Print the files to be stacked
        print(len(self.files), 'files to be stacked')

        # Loop over the files
        for i, file in enumerate(self.files):

            # Check if first image
            if i == 0:

                # Get data for first image to be used as reference
                image0 = fits.open(file)

                # Subtract bias
                data0 = DataReduction.bias_subtract(image0[0].data)
                header0 = image0[0].header
                stack_array = data0
            else:
                image = fits.open(file)

                # Subtract bias
                data = DataReduction.bias_subtract(image[0].data)
                
                # Centroid align current image to first image
                registered_image, footprint = aa.register(data, data0, fill_value = 'nan')
                
                #add current image to 3d stack
                stack_array = np.dstack((stack_array,registered_image))
    
        # Median combine 3D datacube
        median_image = np.nanmedian(stack_array, axis = 2)
        
        return median_image, header0
    
    def bias_subtract(data):

        '''
        PURPOSE:
                This function subtracts the bias from the data.

        INPUTS:
                [data; np.array, float]:  The data to be bias subtracted.

        OUTPUTS:
                [new_data; np.array, float]: The bias subtracted data.
        '''
        
        # Make empty array to store new data
        new_data = np.empty(data.shape)

        # Loop over columns
        for i in range(data.shape[1]):

            # Get overscan data
            overscan_data = data[-16:, i]

            # Get median of overscan data
            overscan_median = np.median(overscan_data)

            # Subtract overscan_median from each column
            new_data[:, i] = data[:, i] - overscan_median  

        return new_data
    
    def master_flat(self, stack_array=None):

        '''
        PURPOSE:
                This function creates a master flat.

        INPUTS:
                [files; list]:  List of directories to fits files.
                [filter; str]:  The filter used for the data.
        
        OPT.
        INPUTS:
                [stack_array; boolean]:  Whether to stack the data or not.
        
        OUTPUTS:
                [master_flat; np.array, float]: The master flat.
        '''

        # Loop over files
        for i, file in enumerate(self.files):
            
            # Extract data and header
            image = fits.open(file)
            header = image[0].header

            # Check if the filter is correct
            if header['FILTER'] != self.filter:
                continue

            # Subtract bias
            data = DataReduction.bias_subtract(image[0].data)
        
            # Stack the array
            if stack_array is None:
                stack_array = data / np.nanmedian(data)
            else:
                stack_array = np.dstack((stack_array, data / np.nanmedian(data)))

        # Get the master flat
        master_flat = np.nanmedian(stack_array, axis = 2)

        return master_flat

class FullWidth:

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
                horizontal, vertical = FullWidth.profiles(self.data, xcenter, ycenter)

                # Interpolate the width     
                r1h, r2h, fwhm_x = FullWidth.interpolate_width(horizontal, background)
                r1v, r2v, fwhm_y = FullWidth.interpolate_width(vertical, background)

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
        plt.savefig('FWHM median.pdf', bbox_inches = 'tight',facecolor = 'white')
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
        median_fwhm = FullWidth.GetFWHM(self)

        # Get the pixel scale of the image
        wcs = WCS(self.header)

        # Get pixel scale in arcsec
        px_scale = wcs.proj_plane_pixel_scales()[0].to(u.arcsec)

        # Get the seeing
        seeing = median_fwhm*px_scale
        print('FWHM:',median_fwhm,'pixels')
        print('Seeing:',seeing)
        return seeing

class Photometry:

    # Initialize the class
    def __init__(self, data, header, phot_table, fwhm,filter):

        # Define the atrributes
        self.data = data
        self.header = header
        self.phot_table = phot_table
        self.fwhm = fwhm
        self.filter = filter
        if filter=='SG':
            self.standard_M = phot_table['V']
        if filter=='SR':
            self.standard_M = phot_table['V']-phot_table['V-R']
        if filter=='SI':
            self.standard_M = (phot_table['V']-phot_table['V-R'])-phot_table['R-I']
    def ap_phot(self, source_positions):

        '''
        PURPOSE:
                Aperture photometry function for standard stars.

        INPUTS:
                    [data; np.array, float]:  Image data, 2d array
                           [header; object]:  Fits header for image
           [source_positions; tuple, float]:  xy positions of stars in pixel coordinates
                        [standard_M; float]:  Standard magnitude array corresponding to xy positions. set to None if you just want counts
                              [fwhm; float]:  Full width half maximum of stars. Radius is 2*fwhm and annulus is 2*fwhm+5 to 2*fwhm+10

        RETURNS:
                [phot_table; object, float]:  Table of photometry including instrumental magnitudes and individual zeropoints
        '''

        # Make the apertures
        aperture = CircularAperture(source_positions, r = 2*self.fwhm)
        annulus = CircularAnnulus(source_positions, r_in = 2*self.fwhm + 5, r_out = 2*self.fwhm + 10)

        # Store the apertures as a list
        phot_aper = [aperture, annulus]
    
        # Make the photometry table
        phot_table = aperture_photometry(self.data, phot_aper)

        # Calculate the background
        bkg_mean = phot_table['aperture_sum_1'] / annulus.area

        # Calculate the background counts sum
        bkg_ap_sum = bkg_mean * aperture.area
        
        # Subtract the background
        final_sum = phot_table['aperture_sum_0'] - bkg_ap_sum

        # Store the background subtracted counts
        phot_table[self.filter + ' bg_subtracted_counts'] = final_sum
        
        # get counts per second
        phot_table[self.filter + ' counts_sec'] = phot_table[self.filter + ' bg_subtracted_counts']/self.header['EXPTIME']
        
        # Calculate the instrumental magnitude and individual zeropoints
        phot_table[self.filter + ' Instr_Mag'] = -2.5*np.log10(phot_table[self.filter + ' counts_sec'])
        phot_table[self.filter + ' zeropoint'] = self.standard_M - phot_table[self.filter + ' Instr_Mag']
        
        return phot_table
    
class Detector:

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
        throughput = observed_count_flux/true_count_flux
        
        print('Expected Counts:',true_count_flux)
        print('Observed_Counts:',observed_count_flux)
        print('Throughput:',throughput)
        
        return throughput