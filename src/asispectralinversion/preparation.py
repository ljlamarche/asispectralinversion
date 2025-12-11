import numpy as np
import matplotlib.pyplot as plt
import h5py
import datetime
import os
from apexpy import Apex
from PIL import Image
from .preprocessing import common_grid, interpolate_reggrid, background_brightness, wavelet_denoise, gaussian_denoise, to_rayleighs 

"""
Purpose of this script:
    - takes in ASI/GLOW information, preparing for preprocessing and inversion
    - runs preprocessing and inversion functions
    - returns Q, E0, SigmaP, and SigmaH in regularized, geomagnetic coordinates
"""



def prepare_data(dtdate, redimgs, greenimgs, blueimgs, skymap_file, blur_deg_EW=0.4, blur_deg_NS=0.04, nshifts=30, background_method='patches', dec=2, plot=True):
    """
    Purpose: 
        - prepares ASI images for inversion with necessary smoothing, ext
    Input:
        dtdate: datetime.date
            Date the inversion is calculated on
        redimgs: list of str
            List of filenames (png) of red images to include in the inversion
        greenimgs: list of str
            List of filenames (png) of green images to include in the inversion
        blueimgs: list of str
            List of filenames (png) of blue images to include in the inversion
        skymap_file: str (optional)
            Path to skymap file when not using default (PKR)
        blur_deg_EW: float (optional)
            East-West bluring in degrees
        blur_deg_NS: float (optional)
            North-South bluring in degrees
        n_shift: int (optional)
            N shift
        background_method: str (optional)
            Method for determining the corners - 'corners' or 'patches'
        dec : int (optional)
            Number of pixels to decimate image by
        plot: bool (optional)
            Whether or not to generate intermediate plots (default=False)
    
    """
    
    print("Pulling information from data files and lookup tables...")

    #dtdate = datetime.date(int(date[:4]),int(date[4:6]),int(date[6:])) # creating datetime object from given date
#    # These should be function parameters
#    blur_deg_EW = 0.4 # gaussian blur width in degrees maglon
#    blur_deg_NS = 0.04 # gaussian blur width in degrees maglat
#    n_shifts = 50 # integer determining shift-invariance of wavelets
#    background_method = 'corners' # set to 'patches' or 'corners'
#    dec = 2 # 'dec = 2' returns given resolution

    # Load PNGs
    redims = list()
    for src_file in redimgs:
        with Image.open(src_file) as img:
            redims.append(np.array(img))

    greenims = list()
    for src_file in greenimgs:
        with Image.open(src_file) as img:
            greenims.append(np.array(img))

    blueims = list()
    for src_file in blueimgs:
        with Image.open(src_file) as img:
            blueims.append(np.array(img))


    # Coadd images
    redimcoadd = sum(redims)/len(redims)
    greenimcoadd = sum(greenims)/len(greenims)
    blueimcoadd = sum(blueims)/len(blueims)

    # Plot coadded images
    if plot:
        plt.imshow(redimcoadd, cmap='Reds')
        plt.title('Red Imagery')
        plt.show()
        
        plt.imshow(greenimcoadd, cmap='Greens')
        plt.title('Green Imagery')
        plt.show()
        
        plt.imshow(blueimcoadd, cmap='Blues')
        plt.title('Blue Imagery')
        plt.show()

    # Dark Frame subtraction nominally occurs here??
    
    # Load skymap file
    with h5py.File(skymap_file, 'r') as h5:
        skymapred = [h5['/magnetic_footpointing/180km/lat'][:],
                     h5['/magnetic_footpointing/180km/lon'][:]]
        skymapgreen = [h5['/magnetic_footpointing/110km/lat'][:],
                       h5['/magnetic_footpointing/110km/lon'][:]]
        skymapblue = [h5['/magnetic_footpointing/107km/lat'][:],
                      h5['/magnetic_footpointing/107km/lon'][:]]

    # Define masks where image not defined
    bmask = np.isnan(skymapblue[0])
    gmask = np.isnan(skymapgreen[0])
    rmask = np.isnan(skymapred[0])


    # Calculate background brightness
    bluebgbright, sig = background_brightness(blueimcoadd, bmask, background_method=background_method, plot=plot)
    greenbgbright, sig = background_brightness(greenimcoadd, gmask, background_method=background_method, plot=plot)
    redbgbright, sig = background_brightness(redimcoadd, rmask, background_method=background_method, plot=plot)

    
    # Wavelet Denoise
    blueimdenoise = wavelet_denoise(blueimcoadd, bluebgbright, nshifts=nshifts)
    greenimdenoise = wavelet_denoise(greenimcoadd, greenbgbright, nshifts=nshifts)
    redimdenoise = wavelet_denoise(redimcoadd, redbgbright, nshifts=nshifts)

    # Plot Wavelet Denoise Images
    if plot:
        plt.imshow(redimdenoise, cmap='Reds')
        plt.title('Red Wavelet Denoise')
        plt.show()
        
        plt.imshow(greenimdenoise, cmap='Greens')
        plt.title('Green Wavelet Denoise')
        plt.show()
        
        plt.imshow(blueimdenoise, 'Blues')
        plt.title('Blue Wavelet Denoise')
        plt.show()


    
    # Define common, regular grid
    blat, blon = skymapblue
    glat, glon = skymapgreen
    rlat, rlon = skymapred

    gridlat, gridlon = common_grid(blat, blon, glat, glon, rlat, rlon)

    # Interpolate images to new common grid
    blueimreg = interpolate_reggrid(blueimdenoise, blon, blat, gridlon, gridlat)
    greenimreg = interpolate_reggrid(greenimdenoise, glon, glat, gridlon, gridlat)
    redimreg = interpolate_reggrid(redimdenoise, rlon, rlat, gridlon, gridlat)


    # Plot Regridded Images
    if plot:
        plt.pcolormesh(gridlon, gridlat, redimreg, cmap='Reds')
        plt.title('Red Regrid')
        plt.xlabel('E-W')
        plt.ylabel('N-S')
        plt.show()
        
        plt.pcolormesh(gridlon, gridlat, greenimreg, cmap='Greens')
        plt.title('Green Regrid')
        plt.xlabel('E-W')
        plt.ylabel('N-S')
        plt.show()
        
        plt.pcolormesh(gridlon, gridlat, blueimreg, cmap='Blues')
        plt.title('Blue Regrid')
        plt.xlabel('E-W')
        plt.ylabel('N-S')
        plt.show()


    # # Grid steps for our new footpointed grid - the new grid is very nearly Cartesian in footlat/footlon
    # dlon = np.mean(np.diff(gridmlon, axis=0))
    # dlat = np.mean(np.diff(gridmlat, axis=1))

#    # Gaussian Denoise
#    blueimdenoise = gaussian_denoise(blueimdenoise, dlat, dlon, bluebgbright, EW_deg=blur_deg_EW, NS_deg=blur_deg_NS)
#
#    # Plot Gaussian Denoise Images
#    if plot:
#        plt.pcolormesh(lon0, lat0, blueimdenoise)
#        plt.title('Blue Gaussian Denoise')
#        plt.xlabel('E-W')
#        plt.ylabel('N-S')
#        plt.show()


    # Convert to Rayleighs
    redray,greenray,blueray = to_rayleighs(redimreg, greenimreg, blueimreg, redbgbright, greenbgbright, bluebgbright)

    # NaN any invalid (negative) pixels
    redray[np.where(redray<0)]=np.nan
    greenray[np.where(greenray<0)]=np.nan
    blueray[np.where(blueray<0)]=np.nan
    badrange = np.where(np.isnan(redray+blueray+greenray))
    
    redray[badrange] = np.nan
    greenray[badrange] = np.nan
    blueray[badrange] = np.nan

    # Plot images in Rayleighs
    if plot:
        plt.pcolormesh(gridlon, gridlat, redray, vmin=0., cmap='Reds')
        plt.colorbar(label='Rayleighs')
        plt.title('red')
        plt.show()
        
        plt.pcolormesh(gridlon, gridlat, greenray, vmin=0., cmap='Greens')
        plt.colorbar(label='Rayleighs')
        plt.title('green')
        plt.show()
        
        plt.pcolormesh(gridlon, gridlat, blueray, vmin=0., cmap='Blues')
        plt.colorbar(label='Rayleighs')
        plt.title('blue')
        plt.show()


    # Visualize all three colors
    if plot:
        ngreen = (1/np.nanstd(greenray))**(6.5/8)
        nred = (1/np.nanstd(redray))**(6.5/8)
        nblue = (1/np.nanstd(blueray))**(6.5/8)
        
        greenmin = np.nanmin(greenray)
        bluemin = np.nanmin(blueray)
        redmin = np.nanmin(redray)
        
        #colormat = np.asarray([3*(redframe-redmin),(greenframe-greenmin),10*(blueframe-bluemin)]).astype(float)
        rednorm = nred*(redray-redmin)
        greennorm = ngreen*(greenray-greenmin)
        bluenorm = nblue*(blueray-bluemin)
        
        colormat = np.array([rednorm, greennorm, bluenorm])
        
        maxbright = np.nanmax(colormat)
        
        colormat /= maxbright
        
        mesh = plt.pcolormesh(gridlon.T,gridlat.T,colormat.transpose((2,1,0)), shading='auto')
        plt.xlabel('E-W')
        plt.ylabel('N-S')
        plt.title('mapped image, original, geodetic coords')
        plt.show()

    
    print("Decimating images...")

    redraydec = redray[::dec, ::dec]
    blueraydec = blueray[::dec, ::dec]
    greenraydec = greenray[::dec, ::dec]
    decgridlat = gridlat[::dec,::dec]
    decgridlon = gridlon[::dec,::dec]

    # Plot decimated image
    if plot:
        rednorm = nred*(redraydec-redmin)
        greennorm = ngreen*(greenraydec-greenmin)
        bluenorm = nblue*(blueraydec-bluemin)
        
        colormat = np.array([rednorm, greennorm, bluenorm])
        
        maxbright = np.nanmax(colormat)
        
        colormat /= maxbright
        
        mesh = plt.pcolormesh(decgridlon.T,decgridlat.T,colormat.transpose((2,1,0)), shading='auto')
        plt.title('mapped image, decimated, geodetic coords')
        plt.xlabel('E-W')
        plt.ylabel('N-S')        
        plt.show()


    return redraydec, greenraydec, blueraydec, decgridlon, decgridlat


