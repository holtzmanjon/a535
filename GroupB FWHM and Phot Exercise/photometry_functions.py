from astropy.io import fits
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import astropy.units as u
from photutils.centroids import centroid_com
from photutils import CircularAperture, CircularAnnulus,aperture_photometry
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
from photutils.centroids import centroid_sources
import astroalign as aa
from astropy.wcs import WCS

def median_combine(files):
    '''
    Create median combined image from list of files
    INPUTS:
        files: list of directories to fits files. First image is used as reference for alignment
    OUTPUTS:
        median_image: data for median combined image
        header0: header of first reference image
    '''
    print(len(files),'files to be stacked')
    for i,file in enumerate(files):
        if i==0:
            #get data for first image to be used as reference
            image0 = fits.open(file)
            data0 = bias_subtract(image0[0].data)
            header0 = image0[0].header
            stack_array = data0
        else:
            image = fits.open(file)
            data = bias_subtract(image[0].data)
            
            #centroid align current image to first image
            registered_image, footprint = aa.register(data, data0,fill_value='nan')
            
            #add current image to 3d stack
            stack_array = np.dstack((stack_array,registered_image))
   
    #median combine 3d datacube
    median_image = np.nanmedian(stack_array,axis=2) #median combine cube
    
    return median_image,header0

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
    for i in range(data.shape[1]):  # Loop over columns
        # Get overscan data
        overscan_data = data[-16:, i]
        # Get median of overscan data
        overscan_median = np.median(overscan_data)
        new_data[:, i] = data[:, i] - overscan_median  # Subtract overscan_median from each column

    return new_data

def ap_phot(data,header,source_positions,standard_M,fwhm):
    '''
    Aperture photometry function for standard stars.
    INPUTS:
        data: image data, 2d array
        header: fits header for image
        source_positions: xy positions of stars in pixel coordinates
        standard_M: standard magnitude array corresponding to xy positions. set to None if you just want counts
        fwhm: full width half maximum of stars. Radius is 2*fwhm and annulus is 2*fwhm+5 to 2*fwhm+10
    RETURNS:
        phot_table: table of photometry including instrumental magnitudes and individual zeropoints
    '''
    median_FWHM = 4.5
    # Make the apertures
    aperture = CircularAperture(source_positions, r=2*median_FWHM)
    annulus = CircularAnnulus(source_positions, r_in=2*median_FWHM+5, r_out=2*median_FWHM+10)

    # Store the apertures as a list
    phot_aper = [aperture, annulus]
 
    # Make the photometry table
    phot_table = aperture_photometry(data, phot_aper)

    # Calculate the background
    bkg_mean = phot_table['aperture_sum_1'] / annulus.area

    # Calculate the background counts sum
    bkg_ap_sum = bkg_mean * aperture.area
    
    # Subtract the background
    final_sum = phot_table['aperture_sum_0']-bkg_ap_sum

    # Store the background subtracted counts
    phot_table['bg_subtracted_counts'] = final_sum
    
    # get counts per second
    phot_table['counts_sec'] = phot_table['bg_subtracted_counts']/header['EXPTIME']
    
    if standard_M !=None:
    
        # Calculate the instrumental magnitude
        phot_table['Instr_Mag'] = -2.5*np.log10(phot_table['bg_subtracted_counts']/header['EXPTIME'])

        # Calculate Zeropoint for each Star
        phot_table['zeropoint'] = standard_M-phot_table['Instr_Mag']

    
    
    return phot_table


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

def GetFWHM(data, sources, makeplot):

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
    mean, median, std = sigma_clipped_stats(data, sigma = 3.0)

    # Set the background
    background = median
       
    # Get the x and y coordinates
    xlist = (np.array(sources['xcentroid']))
    ylist = (np.array(sources['ycentroid']))
    
    # Get the centroids
    xlist, ylist = centroid_sources(data, xlist, ylist, box_size = 25, centroid_func = centroid_com)
    
    # Make empty array to store FWHMs
    FWHM = []

    # Loop over sources
    for ID in range(len(xlist)):

        # Get the x and y centers
        xcenter, ycenter = xlist[ID], ylist[ID]
        
        try:

            # Get the horizontal and vertical profiles
            horizontal, vertical = profiles(data, xcenter, ycenter)

            # Interpolate the width     
            r1h, r2h, fwhm_x = interpolate_width(horizontal, background)
            r1v, r2v, fwhm_y = interpolate_width(vertical, background)

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
                fig,(ax1,ax2,ax3) = plt.subplots(1, 3, figsize = (10, 4))
                
                ax1.imshow(data[int(ycenter) - 25:int(ycenter) + 25, int(xcenter) - 25:int(xcenter) + 25], norm = LogNorm(), origin = 'lower')
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