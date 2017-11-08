#Tools to help with extraction of photometry
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from photutils.centroids import centroid_com
from photutils import aperture_photometry, SkyCircularAperture, SkyCircularAnnulus
from photutils.utils import calc_total_error
from astropy.table import Table
from astropy.time import Time

def generate_regions(hdu, approx_location, centering_width = 80, ap_rad = 6.5, in_rad = 7.0, out_rad = 14.0):
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
    centering_width : int, optional
        Size of box around source region to find the centroid of in pixels.
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

def extract_photometry(filename, approx_location, centering_width = 80, ap_rad = 7.0, in_rad = 8.0, out_rad = 15.0):
    """
    Does aperture photometry on the reduced image in filename, at the location 
    specified by approx_location
    
    Parameters
    ----------
    filename : str
        Name of the reduced image.
    approx_location : `~astropy.coordinates.SkyCoord`
        `astropy.coordinates.SkyCoord` with the RA and Dec of the object you 
        want to extract. Passed to generate_regions.
    centering_width : int, optional
        Size of box around source region to find the centroid of in pixels. 
        Passed to generate_regions.
    ap_rad : float, optional
        Radius of source region in arcseconds. Passed to generate_regions.
    in_rad : float, optional
        Inner radius of background annulus in arcseconds. Passed to generate_regions.
    out_rad : float, optional
        Outer radius of background annulus in arcseconds. Passed to generate_regions.
    
        
    Returns
    -------
    phot_table : `~astropy.table.Table`
        A table containing the photometry information. This is the original filename, 
        the time of the start of the observation, the exposure time, the time associated 
        with the data point, the filter, the center of the extraction region, the radius 
        of the source region, the area of the source region in pixels, the source counts, 
        the source count error, the center of the background region, the inner radius of 
        the background region, the outer radius of the background region, the area of the 
        background region, the background counts, the background count error, then the net
        counts, net count error, and the instrumental magnitude and instrumental magnitude
        error.
        
    """
    #Open up the file
    hdu = fits.open(filename)[0]
    
    #Let's do some bookkeeping
    #Get the TAI time from the header, convert to an astropy.time.Time object
    time_header = hdu.header['DATE-OBS']
    obs_time_obj = Time(time_header, scale = 'tai')
    #Get the actual value of the MJD out of the astropy object
    obs_time = obs_time_obj.mjd
    
    exp_time = hdu.header['EXPTIME'] * u.second
    time_obj = obs_time_obj + (exp_time/2.0)
    #Get the value of the MJD out of the astropy object again.
    time = time_obj.mjd
    
    #Get the name of the filter
    filt = hdu.header['FILTER']
    
    #Now for some actual photometry
    #Give me the data!
    data = hdu.data
    
    #Major source of error in the image assumed to be readout noise and photon shot noise.
    #The latter is handled by calc_total_error. The former is just an array the same size
    #as the data, filled with the read out noise.
    ron = hdu.header['GTRON11']
    error_arr = np.full(data.shape,ron)
    #Calculate the total error array from the data + gain (Poisson noise) and readout noise.
    gain = hdu.header['GTGAIN11']
    error = calc_total_error(data,error_arr,gain)
        
    #Generate source regions
    src,bkg = generate_regions(hdu,approx_location, centering_width, 
                                                    ap_rad, in_rad, out_rad)
    #grab centers of source and background regions
    src_center = (src.positions.ra.value,src.positions.dec.value)
    bkg_center = (bkg.positions.ra.value,bkg.positions.dec.value)
    
    apers = [src,bkg]
      
    #Do some aperture photometry!
    phot_table = aperture_photometry(hdu, apers, error=error)
        
    #Calculate grab counts
    src_cts = phot_table['aperture_sum_0'].data[0]
    src_cts_err = phot_table['aperture_sum_err_0'].data[0]
    
    bkg_cts = phot_table['aperture_sum_1'].data[0]
    bkg_cts_err = phot_table['aperture_sum_err_1'].data[0]
        
    #We need source and background region areas, convert sky regions to pix regions
    wcs = WCS(hdu)
    src_pix = src.to_pixel(wcs)
    bkg_pix = bkg.to_pixel(wcs)
        
    #Calculate region areas
    src_area = src_pix.area()
    bkg_area = bkg_pix.area()
        
    #Scale the background counts
    bkg_scaled = bkg_cts * (src_area / bkg_area)
    bkg_scaled_err = bkg_cts_err * (src_area / bkg_area)
        
    #Net flux = Source - Bkg
    net_cts = src_cts - bkg_scaled
    net_cts_err = np.sqrt(src_cts_err**2.0 + bkg_scaled_err**2.0)
        
    inst_mag = -2.5*np.log10(net_cts/exp_time.value)
    inst_mag_err = 2.5 * net_cts_err / (net_cts*np.log(10.0))
    
    if type(bkg_area[0]) == float:
        
        print(type(bkg_area))
        out_table = Table([[filename],[obs_time],[exp_time.value],[time],[filt],[src_center],[ap_rad],
                       [src_area],[src_cts],[src_cts_err],[bkg_center],[in_rad],[out_rad],
                       [bkg_area],[bkg_cts],[bkg_cts_err],[net_cts[0]],[net_cts_err[0]],
                       [inst_mag[0]],[inst_mag_err[0]]],
                      names = ['Filename','Obs_start','Exptime','Time','Filter','Src_center',
                               'Src_rad','Src_area','Src_cts','Src_cts_err','Bkg_center','Bkg_in_rad',
                              'Bkg_out_rad','Bkg_area','Bkg_cts','Bkg_cts_err','Net_cts','Net_cts_err',
                              'Inst_mag','Inst_mag_err'])
        return out_table
        
    out_table = Table([[filename],[obs_time],[exp_time.value],[time],[filt],[src_center],[ap_rad],
                       [src_area],[src_cts],[src_cts_err],[bkg_center],[in_rad],[out_rad],
                       [bkg_area[0]],[bkg_cts],[bkg_cts_err],[net_cts[0]],[net_cts_err[0]],
                       [inst_mag[0]],[inst_mag_err[0]]],
                      names = ['Filename','Obs_start','Exptime','Time','Filter','Src_center',
                               'Src_rad','Src_area','Src_cts','Src_cts_err','Bkg_center','Bkg_in_rad',
                              'Bkg_out_rad','Bkg_area','Bkg_cts','Bkg_cts_err','Net_cts','Net_cts_err',
                              'Inst_mag','Inst_mag_err'])
    
    print(type(bkg_area))
    
    return out_table
