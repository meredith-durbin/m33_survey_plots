#!/usr/bin/env python

"""Stuff I use in multiple notebooks.
"""

import numpy as np
import vaex

__all__ = ['open_and_select', 'deg_to_deproj']

def open_and_select(photfile='legacy_phot/M33_full_matched.hdf5'):
    '''Read in photometry file and make ST and GST cuts.

    Inputs
    ------
    photfile : string or file-like object, or list of strings
        Input photometry file(s).

    Returns
    -------
    ds : vaex.DataFrame
        Data table with ST and GST selections.
    '''
    if type(photfile) == list:
        ds = vaex.open_many(photfile)
    else:
        ds = vaex.open(photfile)
    ds.select('(F275W_SNR > 4) & (F275W_SHARP**2 < 0.15)', name='F275W_ST')
    ds.select('(F275W_ST & (F275W_CROWD < 1.30))', name='F275W_GST')
    ds.select('(F336W_SNR > 4) & (F336W_SHARP**2 < 0.15)', name='F336W_ST')
    ds.select('F336W_ST & (F336W_CROWD < 1.30)', name='F336W_GST')
    ds.select('F275W_ST | F336W_ST', name='UV_ST')
    ds.select('F275W_GST & F336W_GST', name='UV_GST')

    ds.select('(F475W_SNR > 4) & (F475W_SHARP**2 < 0.2)', name='F475W_ST')
    ds.select('(F475W_ST & (F475W_CROWD < 2.25))', name='F475W_GST')
    ds.select('(F814W_SNR > 4) & (F814W_SHARP**2 < 0.2)', name='F814W_ST')
    ds.select('F814W_ST & (F814W_CROWD < 2.25)', name='F814W_GST')
    ds.select('F475W_ST | F814W_ST', name='OPT_ST')
    ds.select('F475W_GST & F814W_GST', name='OPT_GST')

    ds.select('(F110W_SNR > 4) & (F110W_SHARP**2 < 0.15)', name='F110W_ST')
    ds.select('F110W_ST & (F110W_CROWD < 2.25)', name='F110W_GST')
    ds.select('(F160W_SNR > 4) & (F160W_SHARP**2 < 0.15)', name='F160W_ST')
    ds.select('F160W_ST & (F160W_CROWD < 2.25)', name='F160W_GST')
    ds.select('F110W_ST | F160W_ST', name='IR_ST')
    ds.select('F110W_GST & F160W_GST', name='IR_GST')
    return ds

def deg_to_deproj(ra, dec, ra0=23.4625, dec0=30.6602,
                  i_deg=49, pa_deg=21.1, dmod=24.67):
    '''Converts RA and Dec to deprojected radii.
    
    Inputs
    ------
    ra, dec : float or array-like
        Input coordinates in decimal degrees.
    ra0, dec0 : float
        Central coordinates in decimal degrees.
        Ref: 2019ApJ...872...24V
    i_deg, pa_deg : float
        Inclination and position angle in degrees.
        Ref: 1997ApJ...479..244C
    dmod: float
        Distance modulus. Ref: 2017SSRv..212.1743D

    Returns
    -------
    distA, distB : float or array-like
        Deprojected radii along major and minor axes respectively in kpc
    '''
    i, pa = np.radians([i_deg, pa_deg])
    dist_kpc = 10**(1+dmod/5) / 1000
    dRA, dDec = (ra - ra0), (dec - dec0)
    ra_adj = dRA*np.cos(pa) - dDec*np.sin(pa)
    dec_adj = dRA*np.sin(pa) + dDec*np.cos(pa)
    distA = ra_adj/np.cos(i) * dist_kpc * np.pi/180
    distB = dec_adj * dist_kpc * np.pi/180
    return distA, distB


