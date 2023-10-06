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
import scipy
from scipy import stats
from scipy.integrate import trapz
from scipy.interpolate import interp1d
import pandas as pd

def generate_sed(wavelengths, mag, spec_type='bb5500'):
    """Generates the spectral energy distribution of a given type at the given wavelengths

        Uses spectral models to create an array of photon fluxes at the given wavelengths.
        Currently only supports blackbody spectra. This is normalized so that a star with
        magnitude 0 returns Vega's flux at 5500 Angstroms.

        Parameters:
            wavelengths (np.array): numpy array of the wavelengths (with astropy units) at
                                    which the SED should be calculated
            mag (float): magnitude of the object for which the SED should be calculated; 0
                         is normalized to the flux of Vega
            spec_type (string): the type of SED to be generated; defaults to a blackbody with temperature
                                5500K

        Returns:
            np.array: numpy array of the same length as wavelengths with photon flux distributed
                      based on spec_type

    """
    sed = None
    sed = None
    if spec_type == 'star':
        planet='fagk81d266.dat.txt'
        header_row = 'wave, flux, extra 3\n'
        # Open the file for reading
        with open(planet, 'r') as file:
            # Read the existing content
            existing_content = file.read()
        if header_row not in existing_content:
            updated_content = header_row + existing_content
            with open(planet, 'w') as file:
                file.write(updated_content)
            print("file updated with header")
        else:
            print("")
        star = pd.read_csv(planet, delimiter='\s+', skiprows=1, header=None, names=['wave', 'flux', 'extra'])
        #print(star.head())
        wave = star['wave'].to_numpy() * u.Angstrom
        flux=star['flux'].to_numpy() * 10**16 * u.erg*u.cm**-2/u.s/u.Angstrom#flux ( ergs/cm/cm/s/A * 10**16 ) maybe better to call specific intensity
        interpolated_flux = interp1d(wave, flux, kind='linear', fill_value="extrapolate")
        desired_flux=interpolated_flux(wavelengths) #flux density need to integrate over wavelength
        scaled_flux=desired_flux*10**(-0.4 * mag)
        sed1 = trapz(scaled_flux, x=wavelengths)# ergs/cm/cm/s  FLUXlamda
        #scale=sed1** 10**(-0.4 * mag)
        energy = const.c * const.h / wavelengths
        sed = sed1 / energy
    elif 'bb' in spec_type:
        temp = float(spec_type[2:]) * u.K
        bb = models.BlackBody(temperature=temp, scale=1*u.J*u.m**-2/u.s/u.micron/u.sr)
        norm_factor = (3.63e-9*u.erg/u.cm**2/u.s/u.Angstrom) / (bb(5500 * u.Angstrom) * u.sr) * 10**(-0.4 * mag)
        energy = const.c * const.h / wavelengths
        sed = bb(wavelengths) * norm_factor * u.sr / energy
    elif 'instmag' in spec_type:
        inst_mag = float(spec_type[7:])
        flux = 10 ** (-inst_mag / 2.5) / u.s  # in photons per second
        sed = np.ones(wavelengths.shape) * flux / (np.max(wavelengths) - np.min(wavelengths))

    else:
        print('Spectral type currently unsupported. Null value returned.')

    return sed


def atmos_transmission(wavelengths, model='flat0.8'):
    """Generates the atmospheric transmission curve

        Calculates the transmission curve of the atmosphere based on the model entered.
        Currently only supports a flat model.

        Parameters:
            wavelengths (np.array): numpy array of the wavelengths (with astropy units) at
                                    which the transmission curve should be calculated
            model (string): model to use for the atmospheric transmission; currently only
                            supports a flat transmission curve


        Returns:
            np.array: atmospheric transmission at the given wavelengths

    """
    transmission = np.ones(wavelengths.shape)
    
    #trans=scipy.interpolate.interp1d(wavelengths,airmass,fill_value="extrapolate")
    #trans*=np.ones(data.shape)
    

    if 'flat' in model:
        trans_value = float(model[4:])
        transmission *= trans_value

    return transmission


def instrument_transmission(wavelengths, model='flat0.5'):
    """Generates the instrumental transmission curve

        Calculates the transmission curve of the instrument based on the model entered.
        Currently only supports a flat model.

        Parameters:
            wavelengths (np.array): numpy array of the wavelengths (with astropy units) at
                                    which the transmission curve should be calculated
            model (string): model to use for the instrumental transmission; currently only
                            supports a flat transmission curve


        Returns:
            np.array: instrumental transmission at the given wavelengths

    """
    transmission = np.ones(wavelengths.shape)

    if 'flat' in model:
        trans_value = float(model[4:])
        transmission *= trans_value

    return transmission


def filter_transmission(wavelengths, filter='flat0.8'):
    """Generates the filter transmission curve

        Calculates the transmission curve of the filter based on the model entered.
        Currently only supports a flat model.

        Parameters:
            wavelengths (np.array): numpy array of the wavelengths (with astropy units) at
                                    which the transmission curve should be calculated
            filter (string): type of filter; currently only supports a flat transmission curve


        Returns:
            np.array: filter transmission at the given wavelengths

    """
    transmission = np.ones(wavelengths.shape)

    if 'flat' in filter:
        trans_value = float(filter[4:])
        transmission *= trans_value

    return transmission


def mirror_transmission(wavelengths, filter='flat0.8'):
    """Generates the mirror transmission curve

        Calculates the transmission curve of the mirror based on the model entered.
        Currently only supports a flat model.

        Parameters:
            wavelengths (np.array): numpy array of the wavelengths (with astropy units) at
                                    which the transmission curve should be calculated
            filter (string): type of filter; currently only supports a flat transmission curve


        Returns:
            np.array: mirror transmission at the given wavelengths

    """
    transmission = np.ones(wavelengths.shape)

    if 'flat' in filter:
        trans_value = float(filter[4:])
        transmission *= trans_value

    return transmission


def detector_response(wavelengths, filter='flat0.8'):
    """Generates the detector response for given pixel location

        Calculates the response curve of the detector based on the model entered.
        Currently only supports a flat model.

        Parameters:
            wavelengths (np.array): numpy array of the wavelengths (with astropy units) at
                                    which the transmission curve should be calculated
            filter (string): type of filter; currently only supports a flat transmission curve


        Returns:
            np.array: detector response at the given wavelengths

    """
    response = np.ones(wavelengths.shape)

    if 'flat' in filter:
        trans_value = float(filter[4:])
        response *= trans_value

    return response


def tele_area(name='APO'):
    """Gives telescope area based on telescope name

            Calculates the telescope's area based on its name (or alternately it's radius)

            Parameters:
                name (string): name of the telescope; can also be 'radxx' to use a telescope
                               with radius xx


            Returns:
                float: area of the telescope

    """
    area = None
    if 'rad' in name:
        radius = float(name[3:]) * u.m
        area = np.pi * radius**2
    elif name == 'APO':
        radius = 3.5 / 2 * u.m
        area = np.pi * radius ** 2
    else:
        print('Telescope name not recognized. Null value returned.')

    return area


def calc_monochrome_signal(sed, transmission, area=(np.pi*(3.5/2*u.m)**2), exp_time=(100*u.s), use_area=True):
    """Calculates the monochromatic flux of the given SED

        Uses the signal equation, along with the given transmission curve, to calculate
        the monochromatic signal at the given wavelengths. Wavelengths and sed should
        have units using astropy.units.

        Parameters:
            sed (np.array): numpy array of the same size as wavelengths which contains
                            the flux at the given wavelengths
            transmission (np.array): transmission spectrum (combined to include any relevant factors
                                     including atmosphere, detector effeciency, etc.)
            area (Quantity): telescope area; defaults to the area of a telescope with diameter
                                  3.5 m; can use 1 (with no units) to receive signal per area
            exp_time (Quantity): exposure time for the observation; defaults to 100 s; can use 1 (with
                                 no units) to receive photons per second rather than total photons


        Returns:
            np.array: numpy array of the same length as wavelengths of predicted photon counts
                      at each wavelength

    """
    if not use_area:
        area = 1
    monochrome_signal = (area * exp_time * sed * transmission)
    try:
        monochrome_signal = monochrome_signal.to(1/u.Angstrom)
    except astropy.units.core.UnitConversionError:
        monochrome_signal = monochrome_signal.to(1 / (u.Angstrom * u.s))

    return monochrome_signal


def atmos_emission(wavelengths, type='eso', area=(np.pi*(3.5/2*u.m)**2)):
    """Calculate emission from the atmosphere

            Uses an ESO sky model to calculate the emission due to the atmosphere at the given
            wavelengths. Currently only supports one model of atmospheric emission. Emission
            is given per square arcsec.

    Args:
        wavelengths: wavelengths at which to calculate the background emission from the atmosphere

    Returns:
        np.array: array with astropy units of the sky emission at the given wavelengths
    """
    if type == 'eso':
        lambdas = []
        fluxes = []
        with open('sky_0.0.csv', 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                data = line.split(',')
                lambdas.append(float(data[0]))
                fluxes.append(float(data[1]))
        lambdas = np.array(lambdas) * u.nm
        fluxes = np.array(fluxes) / u.m**2 / u.s / u.micron
        new_fluxes = np.interp(wavelengths, lambdas, fluxes) * area
        plt.plot(lambdas,fluxes)
    elif 'instmag' in type:
        inst_mag = float(type[7:])
        bg = 10**(-inst_mag/2.5) / u.s  # photons per second per square arcsec
        new_fluxes = np.ones(wavelengths.shape) * bg / (np.max(wavelengths) - np.min(wavelengths))
    else:
        print("Type not currently supported")
        new_fluxes = 0

    return new_fluxes


def exp_snr_from_time(sed, transmission, wavelengths,time,tele_area=(np.pi*(3.5/2*u.m)**2),
                      bg_area=np.pi, platescale=0.5, read_noise=5):
    """Calculates the snr of a given object and exposure time

        Uses the signal equation, along with the given transmission curve, to calculate
        the monochromatic snr at the given wavelengths. Wavelengths and sed should
        have units using astropy.units.

        Parameters:
            sed (np.array): numpy array of the same size as wavelengths which contains
                            the flux at the given wavelengths
            transmission (np.array): transmission spectrum (combined to include any relevant factors
                                     including atmosphere, detector effeciency, etc.)
            wavelengths (np.array): numpy array of wavelengths corresponding to the SED
            tele_area (Quantity): telescope area; defaults to the area of a telescope with diameter
                                  3.5 m; can use 1 (with no units) to receive signal per area
            bg_area (float): area (in square arcseconds) of the aperature over which the stars light
                             will be measured
            platescale (float): platescale of the detector in arcsecon/pixel
            read_noise (float): predicted read noise from the detector in e/pixel


        Returns:
            float: snr value 
    """
    F0=3.63e-9 *u.erg*u.cm**-2/u.s/u.Angstrom#ergs/cm2/s/ang STMAG at 5500
    h=6.626176e-27 * u.erg/u.s#ergs s
    c=2.99792458e10 * u.cm/u.s #cm/s
    atmos_trans=atmos_transmission(wavelengths)
    mirror_trans=mirror_transmission(wavelengths)
    instrument_trans=instrument_transmission(wavelengths)
    filter_trans = filter_transmission(wavelengths)
    detect_respon = detector_response(wavelengths)
    #F=F0*10**(-0.4*mag)
    #integral=quad(F,wavelengths[0],wavelengths[-1])
    
    S = calc_monochrome_signal(sed, transmission, area=(np.pi*(3.5/2*u.m)**2), exp_time=time, use_area=False)
    #not sure if i should calculate a given S or intgrate 
    #signal=tele_area*time*integral
    aperature=1 #1 " seeing radius
    platescale=0.5 # 0.5"/pixel
    m=-2.5*np.log10(S)

    A=np.pi*aperature**2
    Npix=aperature/platescale
    background_area=aperature
    surf_bright=1
    read_noise=5 #5 e/pixel
    snr=S/np.sqrt(S+background_area*surf_bright+read_noise*Npix)
    
    return snr
    

def exp_time_from_snr(snr, sed, transmission, wavelengths, B_monochrome, tele_area=(np.pi*(3.5/2*u.m)**2),
                      bg_area=np.pi, platescale=2, read_noise=5, monochrome=True):
    """Calculates the the exposure time needed to get the required SNR

            Uses the signal equation, along with the given transmission curve, to calculate
            the monochromatic signal at the given wavelengths. Wavelengths and sed should
            have units using astropy.units.

            Parameters:
                snr (float): desired SNR of the observation
                sed (np.array): numpy array of the same size as wavelengths which contains
                                the flux at the given wavelengths
                transmission (np.array): transmission spectrum (combined to include any relevant factors
                                         including atmosphere, detector effeciency, etc.)
                wavelengths (np.array): numpy array of wavelengths corresponding to the SED
                B_monochrome (np.array): background photons per wavelength per time
                tele_area (Quantity): telescope area; defaults to the area of a telescope with diameter
                                      3.5 m; can use 1 (with no units) to receive signal per area
                bg_area (float): area (in square arcseconds) of the aperature over which the stars light
                                 will be measured
                platescale (float): platescale of the detector
                read_noise (float): predicted read noise from the detector
                monochrome (bool): if True, calculates assuming a monochromatic SNR

            Returns:
                Quantity: exposure time needed to achieve the given SNR
        """
    if monochrome:
        S = calc_monochrome_signal(sed, transmission, area=tele_area, exp_time=1)
        a = -S ** 2
        b = snr ** 2 * (S + bg_area * B_monochrome)
        c = snr ** 2 * platescale * bg_area * read_noise
        exp_time = ((-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)).to(u.s)
    else:
        S_monochrome = calc_monochrome_signal(sed, transmission, area=tele_area, exp_time=1,use_area=False)
        S = 0 / u.s
        B = 0 / u.s
        for i in range(len(S_monochrome)-1):
            delta_lambda = wavelengths[i + 1] - wavelengths[i]
            S += delta_lambda * S_monochrome[i]
            B += delta_lambda * B_monochrome[i]
        a = -S**2
        b = snr**2 * (S + bg_area * B)
        c = snr**2 * platescale*bg_area*read_noise
        exp_time = ((-b - np.sqrt(b**2 - 4*a*c)) / (2*a)).to(u.s)
    return exp_time


def main():
    wavelengths = (5500, 5550) * u.Angstrom
    V_mag = 20
    atmos_trans = atmos_transmission(wavelengths)
    filter_trans = filter_transmission(wavelengths)
    instrument_trans = instrument_transmission(wavelengths)
    transmission = np.array([1,1])  # atmos_trans * filter_trans * instrument_trans
    area = tele_area(name='rad0.3')
    photon_sed = generate_sed(wavelengths, V_mag, spec_type='instmag-8.5')
    bg = atmos_emission(wavelengths, type='instmag-2.5')
    signal = calc_monochrome_signal(photon_sed, transmission, area=area, exp_time=(1), use_area=False)
    print(signal, 'signal')
    print(signal*850*u.Angstrom)
    print(exp_time_from_snr(100,photon_sed,transmission,wavelengths, bg, monochrome=False), 'exposure time from snr')
    #exp_snr_from_time
    #print(exp_snr_from_time(100,photon_sed*u.s,transmission,wavelengths), 'snr from time')

if __name__ == "__main__":
    main()
