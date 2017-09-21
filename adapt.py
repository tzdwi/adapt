import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from glob import glob
import os

#Author: Trevor Dorn-Wallenstein
#Date: 9.20.17
#Author's note: the data reduction portion of this does almost exactly what acronym 
#(Weisenburger et al. 2017, https://github.com/kweis/acronym) does, with the exception of 
#performing an overscan subtraction before trimming. I wrote this to be slightly more explicit 
#(read: slower) so that a PreMAP student can run any portion of the reduction and see what it does.
#If you want to reduce ARCTIC data and you are not my PreMAP student, PLEASE use Kolby's code!

#Hell, even if you are my PreMAP student, try Kolby's code too!

#If you want to run the pipeline from the command line without any customization,
#do python adapt.py datadir caldir reddir
#where datadir is where the data are, caldir is where master cals go, and reddir is
#where the reduced data go

def split_data_overscan(hdu):
    """
    Search through the image header and return python indices
    for where the overscan region is
    
    Parameters
    ---------
    hdu : `~astropy.io.fits.hdu.image.PrimaryHDU`
        HDU object containing the raw FITS image from which overscan and data arrays 
        are generated.
        
    Returns
    -------
    data : `~numpy.ndarray`
        Trimmed data
    overscan : `~numpy.ndarray`
        Overscan region
        
    """
    
    #Give me the overscan and the header
    image = hdu.data
    header = hdu.header
    
    #Some string manipulation to get the actual values.
    overscan_str = hdu.header['BSEC11'].lstrip('[').rstrip(']').split(',')
    data_str = hdu.header['DSEC11'].lstrip('[').rstrip(']').split(',')
    overscan_x_min,overscan_x_max = overscan_str[1].split(':')
    overscan_y_min,overscan_y_max = overscan_str[0].split(':')
    data_x_min,data_x_max = data_str[1].split(':')
    data_y_min,data_y_max = data_str[0].split(':')
    
    #Remember, FITS is 1-indexed, and sections are inclusive,
    #whereas python is 0-indexed, and exlusive to the end of a section
    data = image[int(data_x_min)-1:int(data_x_max),int(data_y_min)-1:int(data_y_max)]
    overscan = image[int(overscan_x_min)-1:int(overscan_x_max),int(overscan_y_min)-1:int(overscan_y_max)]
    
    return data, overscan

def trim_subtract_overscan(hdu,fit_degree = 8):
    """
    Search through the image header and return python indices
    for where the overscan region is
    
    Parameters
    ---------
    hdu : `~astropy.io.fits.hdu.image.PrimaryHDU`
        HDU object containing the raw FITS image from which overscan is trimmed, then
        fit and subtracted
    fit_degree: int, optional
        Order of the polynomial used to fit the overscan
        
    Returns
    -------
    data_subtracted : `~numpy.ndarray`
        Trimmed and overscan-subtracted data
    header : `~astropy.io.fits.header.Header`
        `~astropy.io.fits.header.Header` object of original fits file,
        modified to say the data have been trimmed and overscan subtracted
        
    """
    
    #Trim data
    data,overscan = split_data_overscan(hdu)
    header = hdu.header
    
    #Average along columns
    avg_overscan = np.mean(overscan,axis=1)
    
    #Index array, then fit!
    idx = np.arange(len(avg_overscan))
    p = np.polyfit(idx,avg_overscan,deg=fit_degree)
    #Calculate array from fit, then transpose into a column
    fit_overscan = np.poly1d(p)(idx)
    fit_overscan_col = fit_overscan[:,np.newaxis]
    #Subtract column!
    data_subtracted = data - fit_overscan_col
    
    #Edit the header
    header.set('COMMENT','Overscan Subtracted + Trimmed')
    header.set('COMMENT','Overscan Fit Order = {0}'.format(fit_degree))
    
    return data_subtracted,header

def master_bias(biaslist,overscan_fit_degree = 8, caldir = None, overwrite = False):
    """
    Construct a master bias using median combination
    
    Parameters
    ---------
    biaslist : list
        List of filenames, should be complete filenames. Use glob to construct. 
        If the list is empty, nothing will happen
    overscan_fit_degree : int, optional
        Order of polynomial to fit overscan with
    caldir : str, optional
        Directory to place master bias into.
    overwrite : bool, optional
        If True, and caldir/master_bias.fits exists, it will be overwritten
        
    Returns
    -------
    masterbias : `~numpy.ndarray`
        Bias array. Note: this will be saved as outdir/master_bias.fits
        
    """
    
    if len(biaslist) == 0:
        print('feed me biases!')
        return None
    
    master_biases =[]
    for bias_name in biaslist:
        hdu = fits.open(bias_name)[0]
        data,header = trim_subtract_overscan(hdu,fit_degree=overscan_fit_degree)
        master_biases.append(data)
        
    master_biases = np.array(master_biases)
    master_bias = np.median(master_biases,axis=0)
    header.set('COMMENT','Biases median-combined')
    header.set('COMMENT','Composed of raw bias frames:')
    for bias_name in biaslist:
        header.set('COMMENT',bias_name)
        
    bias_hdu = fits.PrimaryHDU(master_bias,header)
    if caldir == None:
        bias_hdu.writeto('master_bias.fits', overwrite=overwrite)
    else:
        bias_hdu.writeto(caldir+'master_bias.fits', overwrite=overwrite)
        
    print('Master bias constructed')
    
    return master_bias

def master_dark(darklist,exptime,overscan_fit_degree = 8, caldir = None, overwrite = False):
    """
    Construct a master dark frame using median combination
    
    Parameters
    ---------
    darklist : list
        List of filenames, should be complete filenames. Use glob to construct. 
        If the list is empty, nothing will happen
    exptime : float
        float of exposure time for the dark in seconds. Will be appended to the filename
    overscan_fit_degree : int, optional
        Order of polynomial to fit overscan with
    caldir : str, optional
        Directory to place master dark into.
    overwrite : bool, optional
        If True, and caldir/master_dark_exptime.fits exists, it will be overwritten
        
    Returns
    -------
    masterdark : `~numpy.ndarray`
        dark array. Note: this will be saved as outdir/master_dark_exptime.fits
        
    """
    
    if len(darklist) == 0:
        print('feed me darks!')
        return None
    
    if caldir == None:
        bias = fits.getdata('master_bias.fits')
    else:
        bias = fits.getdata(caldir+'master_bias.fits')
        
    master_darks = []
    for dark_name in darklist:
        hdu = fits.open(dark_name)[0]
        data,header = trim_subtract_overscan(hdu,fit_degree=overscan_fit_degree)
        data -= bias
        master_darks.append(data)
        
    master_darks = np.array(master_darks)
    master_dark = np.median(master_darks,axis=0)
    
    #Some bookkeeping
    header.set('COMMENT','Darks median-combined')
    header.set('COMMENT','Composed of raw dark frames:')    
    for dark_name in darklist:
        header.set('COMMENT',dark_name)
    if caldir == None:
        header.set('COMMENT', 'Bias subtraction done with master_bias.fits')
    else:
        header.set('COMMENT', 'Bias subtraction done with {0}master_bias.fits'.format(caldir))
        
    dark_hdu = fits.PrimaryHDU(master_dark,header)
    if caldir == None:
        dark_hdu.writeto('master_dark_{0}.fits'.format(exptime), overwrite=overwrite)
    else:
        dark_hdu.writeto(caldir+'master_dark_{0}.fits'.format(exptime), overwrite=overwrite)
        
    print('Master dark for {0}s constructed'.format(exptime))
    
    return master_dark

def get_dark(exptime,caldir = None):
    """
    Fetch the appropriate dark frame! If it doesn't exist, scale the longest dark
    
    Parameters
    ---------
    exptime : str
        float of exposure time for the bias in seconds. Will be appended to the filename
    caldir : str, optional
       Directory to search for master dark. 
        
    Returns
    -------
    dark : `~np.ndarray`
        master dark array.
    darkname : str
        name of the file for later reference
        
    """
    
    #Search for all possible dark frames
    available_darks = glob(caldir+'master_dark*')
    available_times = []
    #Check the exposure time. If any match, use that dark.
    for darkname in available_darks:
        dark_hdu = fits.open(darkname)[0]
        dark_time = dark_hdu.header['EXPTIME']
        available_times.append(dark_time)
        if exptime == dark_time:
            dark = dark_hdu.data
            return dark,darkname
        
    #If we're here, then no darks with matching exposure times were found. Scale the longest 
    #dark down to the given exposure time!
    #Find the index with the longest time, grab that time and the corresponding dark frame
    max_dark_idx = np.argmax(available_times)
    max_dark_time = available_times[max_dark_idx]
    darkname = available_darks[max_dark_idx]
    long_dark = fits.getdata(darkname)
    #Scale to the exposure time!
    dark = long_dark * exptime / max_dark_time
    return dark,darkname

def master_flat(flatlist,filt,overscan_fit_degree = 8, caldir = None, overwrite = False):
    """
    Construct a master flat using median combination
    
    Parameters
    ---------
    flatlist : list
        List of filenames, should be complete filenames. Use glob to construct. 
        If the list is empty, nothing will happen
    filt : str
        Name of filter that you're constructing a flat field for.
    overscan_fit_degree : int, optional
        Order of polynomial to fit overscan with
    caldir : str, optional
        Directory to place master dark into.
    overwrite : bool, optional
        If True, and caldir/master_dark_exptime.fits exists, it will be overwritten
        
    Returns
    -------
    masterdark : `~numpy.ndarray`
        dark array. Note: this will be saved as outdir/master_dark_exptime.fits
        
    """
    
    if len(flatlist) == 0:
        print('feed me flats!')
        return None
    
    if caldir == None:
        bias = fits.getdata('master_bias.fits')
    else:
        bias = fits.getdata(caldir+'master_bias.fits')
        
    master_flats = []
    for flat_name in flatlist:
        hdu = fits.open(flat_name)[0]
        data,header = trim_subtract_overscan(hdu,fit_degree=overscan_fit_degree)
        flat_exptime = hdu.header['EXPTIME']
        dark,darkname = get_dark(flat_exptime, caldir = caldir)
        data -= bias
        data -= dark
        master_flats.append(data)
        
    master_flats = np.array(master_flats)
    master_flat = np.median(master_flats,axis=0)
    master_flat /= np.max(master_flat)
    
    #Some bookkeeping
    header.set('COMMENT','Flats median-combined')
    header.set('COMMENT','Composed of raw flat frames:')
    for flat_name in flatlist:
        header.set('COMMENT',flat_name)
    if caldir == None:
        header.set('COMMENT', 'Bias subtraction done with master_bias.fits')
    else:
        header.set('COMMENT', 'Bias subtraction done with {0}master_bias.fits'.format(caldir))
    header.set('COMMENT', 'Dark subtraction done with {0}'.format(darkname))
        
    flat_hdu = fits.PrimaryHDU(master_flat,header)
    if caldir == None:
        flat_hdu.writeto('master_flat_{0}.fits'.format(filt), overwrite=overwrite)
    else:
        flat_hdu.writeto(caldir+'master_flat_{0}.fits'.format(filt), overwrite=overwrite)
        
    print('Master flat for {0} filter constructed'.format(filt))
    
    return master_flat

def get_flat(hdu,caldir = None):
    """
    Fetch the appropriate master flat! If it doesn't exist, return 1.0
    
    Parameters
    ---------
    hdu : `~astropy.io.fits.hdu.image.PrimaryHDU`
        `~astropy.io.fits.hdu.image.PrimaryHDU` to find a flat for
    caldir : str, optional
       Directory to search for master flat. 
        
    Returns
    -------
    flat : `~np.ndarray`
        master flat array.
        
    """
    
    our_filt = hdu.header['FILTER']
    #Search for all possible dark frames
    available_flats = glob(caldir+'master_flat*')
    #Check the filter. If any match, use that dark.
    for flatname in available_flats:
        flat_hdu = fits.open(flatname)[0]
        flat_filt = flat_hdu.header['FILTER']
        if our_filt == flat_filt:
            flat = flat_hdu.data
            return flat,flatname
        
    #If we're here, then no matching master flats with the same filter were found.
    print('No flat for {0} found! Setting flat = 1'.format(hdu.header['FILENAME']))
    flat = 1.0
    flatname = 'NONE FOUND'
    return flat,flatname

def reduce_science(sciencelist,overscan_fit_degree = 8, caldir = None, reddir = None, overwrite = False, out_pref = 'red_'):
    """
    Reduce the science!
    
    Parameters
    ---------
    sciencelist : list
        list of filenames to reduce!
    overscan_fit_degree : int, optional
        Order of polynomial to fit overscan with
    caldir : str, optional
        Directory to place master dark into.
    reddir : str, optional
        Directory to place reduced science image into
    overwrite : bool, optional
        If True, and reddir/master_dark_exptime.fits exists, it will be overwritten
    out_pref : str, optional
        Appends this string to the beginning of the filename
        
    Returns
    -------
    reduced_hdu : `~astropy.io.fits.hdu.image.PrimaryHDU`
        Reduced hdu.
        
    """
    
    print('Reducing {0} science frames!'.format(len(sciencelist)))
    for filename in sciencelist:
    
        #Read data
        hdu = fits.open(filename)[0]

        #Trim and subtract overscan
        data,header = trim_subtract_overscan(hdu,fit_degree=overscan_fit_degree)

        #Bias subtract!
        if caldir == None:
            bias = fits.getdata('master_bias.fits')
            header.set('COMMENT', 'Bias subtraction done with master_bias.fits')
        else:
            bias = fits.getdata(caldir+'master_bias.fits')
            header.set('COMMENT', 'Bias subtraction done with {0}master_bias.fits'.format(caldir))

        data -= bias

        #Dark subtract!!
        exptime = header['EXPTIME']
        dark,darkname = get_dark(exptime, caldir=caldir)
        header.set('COMMENT', 'Dark subtraction done with {0}'.format(darkname))

        data -= dark

        #Flat field!
        flat,flatname = get_flat(hdu, caldir=caldir)
        header.set('COMMENT', 'Flat fielding done with {0}'.format(flatname))

        data /= flat

        #Mess with some filename stuff so it saves to the right place...
        just_filename = filename.split('/')[-1]
        if reddir == None:
            outname = out_pref+just_filename
        else:
            outname = reddir+out_pref+just_filename

        reduced_hdu = fits.PrimaryHDU(data,header)
        reduced_hdu.writeto(outname,overwrite=overwrite)
    
    return 'Complete!! Hooray!'

def generate_lists(datadir = './',bias_keyword = 'Bias', dark_keyword = 'Dark', flat_keyword = 'Flat', science_keyword = 'Object'):
    """
    Generates lists of filetypes to feed into the pipeline
    
    Parameters
    ---------
    datadir : str, optional
        Directory where the data are stored. Should end in /
    bias_keyword : str, optional
        How are bias files named?
    dark_keyword : str, optional
        How are dark files named?
    flat_keyword : str, optional
        How are flat files named?
    science_keyword : str, optional
        How are science files named?
        
    Returns
    -------
    biaslist : list
        list of biases
    darklists : list
        list of lists of darks, one list for each dark exposure time taken
    exptimes : list
        list of exposures times, one for each list of darks in darklists
    flatlists : list
        list of lists of flats, one for each filter taken
    filters : list
        list of filter names, one for each list of flats in flatlists. Note: because filter
        names are messy, this just uses the last letter of the filter. So SDSS g -> g, but 
        CU Ha -> a. Sorry...
    sciencelist :
        list of science images
        
    """
    
    files = glob(datadir+'*.fits')
    
    biaslist = []
    tmp_darklist = []
    dark_times = []
    tmp_flatlist = []
    flat_filts = []
    sciencelist = []
    
    #sort by file type 
    for file in files:
        hdu = fits.open(file)[0]
        filetype = hdu.header['IMAGETYP']
        
        if filetype == bias_keyword:
            biaslist.append(file)
            
        elif filetype == dark_keyword:
            tmp_darklist.append(file)
            exptime = hdu.header['EXPTIME']
            dark_times.append(exptime)
            
        elif filetype == flat_keyword:
            tmp_flatlist.append(file)
            filt = hdu.header['FILTER']
            flat_filts.append(filt)
            
        elif filetype == science_keyword:
            sciencelist.append(file)
            
    #now sort darks by exptime and flats by filter
    darklists = []
    exptimes = []
    for dark_time in np.unique(dark_times):
        darklist = np.array(tmp_darklist)[(dark_times == dark_time)]
        darklists.append(list(darklist))
        exptimes.append(dark_time)
        
    flatlists = []
    filters = []
    for filt in np.unique(flat_filts):
        flatlist = np.array(tmp_flatlist)[np.array(flat_filts) == filt]
        flatlists.append(list(flatlist))
        filters.append(filt[-1])
        
    return biaslist,darklists,exptimes,flatlists,filters,sciencelist

def run_pipeline_run(datadir = './', caldir = None, reddir = None,overscan_fit_degree = 8, overwrite = True, out_pref = 'red_'):
    """
    Runs the entire pipeline. Overscan subtract, trim, bias, dark, flat, you name it.
    
    Parameters
    ---------
    datadir : str, optional
        Directory where the data are stored. Should end in /
    caldir : str, optional
        Directory where the master calibration data should go. 
    reddir : str,optional
        Directory where reduced data should go.
    overscan_fit_degree : int, optional
        Order of polynomial used to fit overscan region.
    overwrite : bool, optional
        If output files exist already and overwrite = True, overwrites files. Otherwise, exits
        with an error
    out_pref : str, optional
        String attached to filenames of reduced science data
        
    Returns
    -------
    biaslist : list
        list of biases
    darklists : list
        list of lists of darks, one list for each dark exposure time taken
    exptimes : list
        list of exposures times, one for each list of darks in darklists
    flatlists : list
        list of lists of flats, one for each filter taken
    filters : list
        list of filter names, one for each list of flats in flatlists. Note: because filter
        names are messy, this just uses the last letter of the filter. So SDSS g -> g, but 
        CU Ha -> a. Sorry...
    sciencelist :
        list of science images
        
    """
    
    biaslist,darklists,exptimes,flatlists,filters,sciencelist = generate_lists(datadir=datadir)
    
    if caldir != None:
        if not os.path.isdir(caldir):
            os.makedirs(caldir)
            
    if reddir != None:
        if not os.path.isdir(reddir):
            os.makedirs(reddir)
    
    print('Making Biases...')
    master_bias(biaslist=biaslist,overscan_fit_degree=overscan_fit_degree,caldir=caldir,overwrite=overwrite)
    
    print('Making Darks...')
    for darklist,exptime in zip(darklists,exptimes):
        master_dark(darklist=darklist,exptime=exptime,overscan_fit_degree=overscan_fit_degree,caldir=caldir,overwrite=overwrite)
    
    print('Making Flats')
    for flatlist,filt in zip(flatlists,filters):
        master_flat(flatlist=flatlist,filt=filt,overscan_fit_degree=overscan_fit_degree,caldir=caldir,overwrite=overwrite)
    
    print('Reducing Science!')
    reduce_science(sciencelist=sciencelist,overscan_fit_degree=overscan_fit_degree,caldir=caldir,reddir=reddir,overwrite=overwrite,out_pref=out_pref)
    
    return 'Complete!!'

if __name__ == '__main__':
    
    from sys import argv
    
    if len(argv) == 4:
        datadir = argv[1]
        caldir = argv[2]
        reddir = argv[3]
    else:
        print('Wrong number of arguments, using defaults')
        datadir = './'
        caldir = None
        reddir = None
        
    run_pipeline_run(datadir=datadir,caldir=caldir,reddir=reddir)
        