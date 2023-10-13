from astropy.io import fits
import numpy as np
import pyds9

dir='/home/holtz/raw/apo/dec06/UT061215'
d=pyds9.DS9()

def rd(file) :
    ''' return data array from input FITS file '''
    return fits.open(dir+'/'+file)[0].data

def disp(im) :
    ''' Display an input array in DS9 window '''
    d.set_np2arr(im)

def biassub(im) :
    ''' subtract mean bias from input array '''
    bias=im[10:1000,1050:1070].mean()
    print 'bias level: ', bias
    return im-bias

def norm(im) :
    ''' normalize input array from mean of central region'''
    norm=im[400:600,400:600].mean()
    print 'normalization level: ', norm
    return im/norm

def combine(lis) :
    ''' Median combine input list of arrays'''
    cube=np.array(lis)
    return np.median(cube,axis=0)
   
def mkflat(files) :
    ''' Create flat field from input list of files '''
    list=[]
    for file in files :
        im=rd(file)
        list.append(norm(biassub(im)))
    return combine(list)

def reduce(file,flat) :
    ''' Reduce an input file given a flat field '''
    im=rd(file)
    return biassub(im)/flat

gflat=mkflat(['flat_g.0010.fits','flat_g.0011.fits','flat_g.0012.fits'])
sn17135_r=reduce('SN17135_r.0103.fits',gflat)
