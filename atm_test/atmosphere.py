import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats


#PLOT PARAMETERS
f=8
SMALL_SIZE = 18
MEDIUM_SIZE = 28
BIGGER_SIZE = 30
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams["figure.figsize"] = (12,8)




#file name
# APO airmass extinction curve, prepped for IRAF
# wavelength, mag/airmass  (J.R.A. Davenport 2015)
file_path = 'apoextinct.dat'
#read in file
df = pd.read_csv(file_path, sep=' ', index_col=False, dtype=np.float64,skiprows=2 )
wave=df.iloc[:,0]
airmass=df.iloc[:,2]


def inter(lam, wave=wave, airmass=airmass):
    """
    calculate the transmission through the atmosphere

    Args:
        lam(numpy array): wavelength range you want values for
        wave(numpy array): set wavelength from Davenport table
        airmass(numpy array): set airmass transmission from Davenport table

    Returns:
        Numpy array: the interpolated value of tranmssion at a given wavelength.

    """
    #beers law?  I/Io=t

    trans=scipy.interpolate.interp1d(wave,airmass,fill_value="extrapolate")
    return trans[lam]
