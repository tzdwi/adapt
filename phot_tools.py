#Tools to help with extraction of photometry
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from photutils.centroids import centroid_com
from photutils import SkyCircularAperture, SkyCircularAnnulus

def generate_regions(hdu, approx_location, centering_width = 80, ap_rad = 7.0, in_rad = 8.0, out_rad = 15.0):
    """
    Generates source and background regions for aperture photometry. 
    Given an image and the approximate RA/Dec of the source, finds the 
    centroid within centering_width pixels and generates regions with 
    the given parameters
    
    Parameters
    ----------
    hdu : `~astropy.io.fits.hdu.image.PrimaryHDU`
        HDU object containing the FITS image from which regions are generated.
        Should be just the primary hdu (e.g., hdu[0]).
    approx_location : `~astropy.coordinates.SkyCoord`
        `astropy.coordinates.SkyCoord` with the RA and Dec of the object you 
        want to generate a region for.
    ap_rad : float, optional
        Radius of source region in arcseconds.
    in_rad : float, optional
        Inner radius of background annulus in arcseconds
    out_rad : float, optional
        Outer radius of background annulus in arcseconds
    
        
    Returns
    -------
    src : `~photutils.SkyCircularAperture`
        Aperture object for source
    bkg : `~photutils.SkyCircularAnnulus`
        Aperture object for background
        
    """
    
    #Make data and wcs objects
    data = hdu.data
    wcs = WCS(hdu)
    
    #Make the right shape array of coordinates
    world_loc = np.array([[approx_location.ra.value,approx_location.dec.value]])
    
    #Convert to pixel coordinates from the FITS image, 0 indexed b.c. we're working with
    #a numpy array
    approx_pix = wcs.wcs_world2pix(world_loc,0)[0]
    
    #Convert to pixel locations of the window.
    min_x = int(approx_pix[0] - centering_width/2.0)
    min_y = int(approx_pix[1] - centering_width/2.0)
    max_x = int(approx_pix[0] + centering_width/2.0)
    max_y = int(approx_pix[1] + centering_width/2.0)
    
    #Make a little cutout around the object
    #Numpy arrays are weird, so x->y, y->x
    stamp = data[min_y:max_y,min_x:max_x]
    
    #Calculate the centroid of the stamp
    x_stamp_centroid, y_stamp_centroid = centroid_com(stamp)
    
    #Add back in the boundaries of the box to get centroid in data coords
    x_centroid = x_stamp_centroid + min_x
    y_centroid = y_stamp_centroid + min_y
    
    #Convert back to RA/Dec. Remember, these are 0-indexed pixels.
    centroid = wcs.wcs_pix2world(np.array([[x_centroid,y_centroid]]),0)
    
    #Convert centroid to SkyCoords object
    location = SkyCoord(ra = centroid[0,0] * u.degree, dec = centroid[0,1] * u.degree)
    
    #Generate regions based on coordinates and given radii.
    src = SkyCircularAperture(location, r=ap_rad * u.arcsecond)
    bkg = SkyCircularAnnulus(location, r_in=in_rad * u.arcsecond, r_out=out_rad * u.arcsecond)
    
    return src,bkg