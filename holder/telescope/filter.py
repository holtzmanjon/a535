import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
file_path = 'ubvri.tsv'
#read in file
df = pd.read_csv(file_path, sep='\t', header=0)
df.drop(columns=['Quantity'])
waveff=df.iloc[0].to_numpy()
fv=df.iloc[2].to_numpy()
flam=df.iloc[3].to_numpy()


def magtoflux(mag,waveff=waveff,fv=fv,flam=flam):
    """UBVRI MAG to Flux converter
    
     Parameters:
     mag (nparray): UBVRI magnitudes in array in that order
     
     
     effective wavelength read from table, 
    
    Returns:
    int:Returning value
    """

    U=float(mag[0])
    B=float(mag[1])
    V=float(mag[2])
    R=float(mag[3])
    I=float(mag[4])
    waveff=waveff[:len(mag)]
    fv=fv[:len(mag)]
    flam=flam[:len(mag)]
    ftotnu=np.zeros_like(mag)
    ftotlam=np.zeros_like(mag)
    """for i,_ in enumerate(mag):
        _=float(_)
        i=int(i)
        print(fv[i])
        waveff[i]
        #note would multply then divide by wavelength effective
        ftotnu[i]=10**(_/(-2.5))*fv[i]*waveff[i]/waveff[i]
        ftotlam[i]=10**(_/(-2.5))*flam[i]*waveff[i]/waveff[i]"""
    ftotnu=10**(mag/(-2.5))*fv*waveff/waveff
    ftotlam=10**(mag/(-2.5))*flam*waveff/waveff
    Fnu=np.sum(ftotnu)
    Flam=np.sum(ftotlam)
    
    b=2.897771955e-3# WEINS CONSTSTAN WVE m K.
    
    return ftotnu,ftotlam, waveff


def filter(wavel,filtername):
    """
    Filter function to calculate throughput.

    Parameters
    ----------
    wavel : type

    Returns
    -------
    type
        throughput through a given filter

    Raises
    ------
    ExceptionType
        Description of when this exception is raised.
    """
    return throughput*constant