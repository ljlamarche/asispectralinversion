# Generate skymap file
# This is a scratch script for the acesii cameras for now.  May generalize later.

import numpy as np
import h5py
from apexpy import Apex

filename6300 = '/Users/e30737/Downloads/ACESII_AllSky_skibotn_6300.hdf5'
filename5577 = '/Users/e30737/Downloads/ACESII_AllSky_skibotn_5577.hdf5'

A = Apex(2022)

with h5py.File(filename6300, 'r') as h5:
    lat6300 = h5['lat'][:]
    lon6300 = h5['long'][:]

mlat6300, mlon6300 = A.geo2apex(lat6300, lon6300, height=180.)



with h5py.File(filename5577, 'r') as h5:
    lat5577 = h5['lat'][:]
    lon5577 = h5['long'][:]

mlat5577, mlon5577 = A.geo2apex(lat5577, lon5577, height=110.)


with h5py.File('skymap_acesii.mat', 'w') as h5:
    h5.create_group('/vertical_footpointing')
    h5.create_group('/vertical_footpointing/107km')
    h5.create_dataset('vertical_footpointing/107km/lat', data=lat5577)
    h5.create_dataset('vertical_footpointing/107km/lon', data=lon5577)
    h5.create_group('/vertical_footpointing/110km')
    h5.create_dataset('vertical_footpointing/110km/lat', data=lat5577)
    h5.create_dataset('vertical_footpointing/110km/lon', data=lon5577)
    h5.create_group('/vertical_footpointing/180km')
    h5.create_dataset('vertical_footpointing/180km/lat', data=lat6300)
    h5.create_dataset('vertical_footpointing/180km/lon', data=lon6300)


    h5.create_group('/magnetic_footpointing')
    h5.create_group('/magnetic_footpointing/107km')
    h5.create_dataset('magnetic_footpointing/107km/lat', data=mlat5577)
    h5.create_dataset('magnetic_footpointing/107km/lon', data=mlon5577)
    h5.create_group('/magnetic_footpointing/110km')
    h5.create_dataset('magnetic_footpointing/110km/lat', data=mlat5577)
    h5.create_dataset('magnetic_footpointing/110km/lon', data=mlon5577)
    h5.create_group('/magnetic_footpointing/180km')
    h5.create_dataset('magnetic_footpointing/180km/lat', data=mlat6300)
    h5.create_dataset('magnetic_footpointing/180km/lon', data=mlon6300)
