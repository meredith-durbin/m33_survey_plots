import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.wcs import WCS
from skimage.draw import polygon

if __name__ == '__main__':
    import sys
    det = sys.argv[1] if (len(sys.argv) > 1) else 'all'
    df = pd.read_csv('../header_info.csv', index_col=0)
    if det != 'all':
        df = df.query(f'DETECTOR == "{det}"')
    w = WCS('data/M33_SDSS9_r.fits')

    img = np.zeros(w._naxis[::-1], dtype=np.float32)
    for i, row in df.iterrows():
        ra = row.filter(regex='^RA_CHIP1_[0-3]$').astype(float)
        dec = row.filter(regex='^DEC_CHIP1_[0-3]$').astype(float)
        xpix, ypix = np.round(w.all_world2pix(ra, dec, 0)).astype(int)
        rr, cc = polygon(ypix, xpix, img.shape)
        img[rr, cc] += row.EXPTIME
        if np.isfinite(row.filter(regex='^RA_CHIP2_[0-3]$').astype(float)).all():
            ra = row.filter(regex='^RA_CHIP2_[0-3]$').astype(float)
            dec = row.filter(regex='^DEC_CHIP2_[0-3]$').astype(float)
            xpix, ypix = np.round(w.all_world2pix(ra, dec, 0)).astype(int)
            rr, cc = polygon(ypix, xpix, img.shape)
            img[rr, cc] += row.EXPTIME

    wh2 = w.to_header()
    hdu = fits.PrimaryHDU(header=wh2, data=img)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(f'data/exposure_map_{det}_small.fits', overwrite=True)