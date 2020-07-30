import glob
import dask.dataframe as dd
from multiprocessing import Pool, cpu_count

from astropy.io import fits
from astropy.wcs import WCS

def get_pixscale(i, other_kw=['exptime']):
    kw_dict = {}
    with fits.open(f'realign/{i}', mode='readonly') as f:
        w = WCS(f[1].header, fobj=f)
        for k in other_kw:
            kw_dict[k] = f[0].header[k.upper()]
    df = pd.DataFrame(index=[i], columns=['xsc', 'ysc'] + other_kw)
    df.loc[i, ['xsc', 'ysc']] = np.abs(np.linalg.eigh(w.pixel_scale_matrix * 3600 * 1000)[0])
    for k, v in kw_dict.items():
        df.loc[i, k] = v
    return df

def read_shifts(detector, basepath='realign/shifts_?db6??'):    
    paths = glob.glob(f'{basepath}_{detector}.txt')
    df = dd.read_csv(paths, comment='#', delim_whitespace=True,
                     names=['name','xshift','yshift','rot',
                            'scale','rmsxpix','rmsypix']).compute()
    df.sort_values(by='name', inplace=True)
    df.set_index('name', inplace=True)
    with Pool(cpu_count()-1) as p:
        df_other = p.map(get_pixscale, df.index)
    df = df.join(pd.concat(df_other))
    df = df.assign(rmsx = df.rmsxpix * df.xsc, rmsy = df.rmsypix * df.ysc)
    df = df.assign(rms=(df.rmsx**2 + df.rmsy**2)**0.5)
    df = df.assign(detector=detector)
    return df
    
df = pd.concat([read_shifts('acs'), read_shifts('ir'), read_shifts('uvis')])
df.to_csv('data/shifts_all.csv')