import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.wcs import WCS
from skimage.draw import polygon
from skimage.transform import downscale_local_mean

if __name__ == '__main__':
    import sys
    det = sys.argv[1]
    df = pd.read_csv('../header_info.csv', index_col=0)
    w = WCS('data/M33_SDSS9_r.fits')
    wh = w.to_header()
    wh['CDELT1'] /= 10
    wh['CDELT2'] /= 10
    wh['CRPIX1'] *= 10
    wh['CRPIX2'] *= 10
    w2 = WCS(wh)
    w2._naxis = [i * 10 for i in w._naxis]

    img = np.zeros(w2._naxis[::-1], dtype=np.float32)
    for i, row in df.iterrows():
        if row.DETECTOR == det:
            ra = row.filter(regex='^RA_CHIP1_[0-3]$').astype(float)
            dec = row.filter(regex='^DEC_CHIP1_[0-3]$').astype(float)
            xpix, ypix = np.round(w2.all_world2pix(ra, dec, 0)).astype(int)
            rr, cc = polygon(ypix, xpix, img.shape)
            img[rr, cc] += row.EXPTIME
            if np.isfinite(row.filter(regex='^RA_CHIP2_[0-3]$').astype(float)).all():
                ra = row.filter(regex='^RA_CHIP2_[0-3]$').astype(float)
                dec = row.filter(regex='^DEC_CHIP2_[0-3]$').astype(float)
                xpix, ypix = np.round(w2.all_world2pix(ra, dec, 0)).astype(int)
                rr, cc = polygon(ypix, xpix, img.shape)
                img[rr, cc] += row.EXPTIME

    wh2 = w.to_header()
    wh2['CDELT1'] /= 5
    wh2['CDELT2'] /= 5
    wh2['CRPIX1'] *= 5
    wh2['CRPIX2'] *= 5

    img2 = downscale_local_mean(img, (2, 2))

    hdu = fits.PrimaryHDU(header=wh2, data=img2)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(f'data/exposure_map_{det}.fits', overwrite=True)