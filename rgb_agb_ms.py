import sys
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u
import matplotlib.patches as patches
import matplotlib.path as path
import vaex
import img_scale
from astropy.coordinates import Angle
from scipy.ndimage import gaussian_filter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import median_filter

import warnings
warnings.filterwarnings('ignore')

params = {'xtick.labelsize': 28,
          'ytick.labelsize': 28,
          'axes.labelsize': 28}
mpl.rcParams.update(params)

if __name__ == '__main__':

        ra_m33, dec_m33 = 23.462*u.deg, 30.66*u.deg
        dm33 = 859.*u.kpc
        dmod = 5.*np.log10(dm33.to('pc').value)-5.

        def deg2diff(ra,dec,ra0,dec0):
                dRA, dDec = (ra-ra0), (dec-dec0)
                return dRA*np.cos(dec0),dDec
        def diff2deg(dra,ddec,ra0,dec0):
                ra,dec = ra0+dra/np.cos(dec0),ddec+dec0
                return ra,dec

        # -------- Read in data ---------------------
        datadir = '/Users/asmerci/data/M33/'
        plotdir = 'Documents/Research/PHAT/plots/for_PHATTER-I/'
        ds = vaex.open(datadir+'M33_full_matched.hdf5')
        
        
        
        ################################################
        # ------- Stellar Selection ------------------ #
        ################################################
        ds.select('(F275W_SNR > 4) & (F275W_SHARP**2 < 0.15) & '
                  '(F275W_CROWD < 1.30)',name='F275W_GST')
                  
        ds.select('(F336W_SNR > 4) & (F336W_SHARP**2 < 0.15) & '
                  '(F336W_CROWD < 1.30)', name='F336W_GST')
        
        ds.select('F275W_GST & F336W_GST',name='UV_GST')
        
        ds.select('(F475W_SNR > 4) & (F475W_SHARP**2 < 0.2) & '
                  '(F475W_CROWD < 2.25) & '
                  '(F814W_SNR > 4) & (F814W_SHARP**2 < 0.2) & '
                  '(F814W_CROWD < 2.25)', name='OPT_GST')
                  
        ds.select('OPT_GST & '
                  '(F110W_SNR > 4) & (F110W_SHARP**2 < 0.15) & '
                  '(F110W_CROWD < 2.25) & '
                  '(F160W_SNR > 4) & (F160W_SHARP**2 < 0.15) & '
                  '(F160W_CROWD < 2.25)', name='IR_GST')
        
        # Define reddening-free F160W magnitude, q
        ds['q'] = ds.F160W_VEGA - ((ds.F110W_VEGA-ds.F160W_VEGA)-1.) * 0.2029/(0.3266 - 0.2029)
        ds['dist'] = np.sqrt( ((ds.RA-ra_m33.value)*np.cos(dec_m33.to('rad')))**2 + \
                              (ds.DEC-dec_m33.value)**2 )
        
        # Select old stars with both a shallow and deep selection (to scale RGB counts)
        ds.select('IR_GST & ~F275W_GST & '
                  '(dist > 0.02)',name='OLD_IR_outer')
        ds.select('IR_GST & ~F275W_GST & '
                  '(dist <= 0.02)',name='OLD_IR_inner')
        ds.select('OLD_IR_inner | OLD_IR_outer',name='OLD_IR')
        
        # RGB selection in F110W/F160W
        prgb_ir = path.Path([(0.6,22),(2,22),
                             (2,18.7),(0.9,18.7),
                             (0.6,22)])
        ds['RGB_yn'] = prgb_ir.contains_points(
                                np.vstack([ds.F110W_VEGA.evaluate()-ds.F160W_VEGA.evaluate(),ds.q.evaluate()]).T
                                ).astype(int)
        ds.select('RGB_yn & OLD_IR_outer & (F110W_VEGA < 24.5)',name='RGB_outer')
        ds.select('RGB_yn & OLD_IR_inner & (F110W_VEGA < 23.5) & (q < 21)',name='RGB_inner')
        ds.select('RGB_outer | RGB_inner',name='RGB')
        ds.select('RGB_yn & OLD_IR & (F110W_VEGA < 23.5) & (q < 21)',name='RGB_bright')
                
        # AGB selection in F110W/F160W
        pagb_ir = path.Path([(0.9,18.5),(2,18.5),
                             (2,16),(1.03,16),
                             (0.9,18.5)])
        ds['AGB_yn'] = pagb_ir.contains_points(
                                np.vstack([ds.F110W_VEGA.evaluate()-ds.F160W_VEGA.evaluate(),ds.q.evaluate()]).T
                                ).astype(int)
        ds.select('AGB_yn & IR_GST',name='AGB')
        
        # Compute scale factor between inner and outer disk
        ds.select('IR_GST & ~F275W_GST & '
                  '(F110W_VEGA < 24.5) & (q < 22) & '
                  '(dist > 0.0199) & (dist < 0.0201) & '
                  'RGB_yn',name='OUTER_SCALE')
        ds.select('IR_GST & ~F275W_GST & '
                  '(F110W_VEGA < 23.5) & (q < 21) & '
                  '(dist > 0.0199) & (dist < 0.0201) & '
                  'RGB_yn',name='INNER_SCALE')
        scale_factor = ds.count(selection='OUTER_SCALE')/ds.count(selection='INNER_SCALE')
         
        # Young star selection in F475W/F814W
        ds.select('UV_GST & OPT_GST & IR_GST',name='VERY_BRIGHT')
        
        pms = path.Path([(-0.4,24),(0.5,24),
                         (-0.1,18),(-0.4,18),
                         (-0.4,24)])
        ds['MS_yn'] = pms.contains_points(
                                np.vstack([ds.F475W_VEGA.evaluate()-ds.F814W_VEGA.evaluate(),ds.F814W_VEGA.evaluate()]).T
                                ).astype(int)
        ds.select('MS_yn & VERY_BRIGHT',name='MS')

        
        
        ################################################
        # ----- Plot CMD ----------------------------- #
        ################################################
        fig = plt.figure()
        fig.set_figheight(10)
        fig.set_figwidth(20)

        ax = plt.subplot(1,2,1)
        ax.set_xlim(-1.5,6.5)
        ax.set_ylim(28,16)
        ax.set_xlabel(r'F475W$-$F814W')
        ax.set_ylabel(r'F814W')
        
        cmap = mpl.cm.get_cmap('viridis')
        cmap.set_under(color='white')
                     
        # Optical CMD of all populations
        hess,hmag,hcol = np.histogram2d(ds.F814W_VEGA.evaluate(selection='OPT_GST'),
                                        ds.F475W_VEGA.evaluate(selection='OPT_GST')-ds.F814W_VEGA.evaluate(selection='OPT_GST'),
                                        bins=(400,400),
                                        range = [[16,28],[-2,7]],
                                        density=False)
        shess = img_scale.log(hess,scale_min=1.,scale_max=1.e4)
               
        ax.add_patch(patches.PathPatch(pms,facecolor='none',edgecolor='magenta',lw=1.5))
        im = ax.imshow(shess,interpolation='nearest',
                       extent=[hcol[0],hcol[-1],hmag[-1],hmag[0]],
                       aspect='auto', cmap=cmap,
                       vmin=0.01,vmax=1.)
                                        
        ax.text(-0.25,17.5,'MS',fontsize=25,color='magenta',ha='center')
        ax.text(6.3,16.7,'Optical GST',ha='right',fontsize=28)
                    
        ax2 = plt.subplot(1,2,2)
        ax2.set_xlim(-0.99,2.49)
        ax2.set_ylim(26,13)
        ax2.set_xlabel(r'F110W$-$F160W')
        ax2.set_ylabel(r'$q_{\rm F160W}$')
        
        # IR CMD of old populations
        hess,hmag,hcol = np.histogram2d(ds.q.evaluate(selection='OLD_IR'),
                                        ds.F110W_VEGA.evaluate(selection='OLD_IR')-ds.F160W_VEGA.evaluate(selection='OLD_IR'),
                                        bins=(400,400),
                                        range = [[13,26],[-2,3]],
                                        density=False)
        shess = img_scale.log(hess,scale_min=1.,scale_max=1.e4)
                        
        ax2.add_patch(patches.PathPatch(prgb_ir,facecolor='none',edgecolor='dodgerblue',lw=1.5))
        ax2.add_patch(patches.PathPatch(pagb_ir,facecolor='none',edgecolor='orange',lw=1.5))
        im = ax2.imshow(shess,interpolation='nearest',
                        extent=[hcol[0],hcol[-1],hmag[-1],hmag[0]],
                        aspect='auto', cmap=cmap,
                        vmin=0.01,vmax=1.)
                
        ax2.text(2.1,21,'RGB',fontsize=25,color='dodgerblue')
        ax2.text(2.1,17.5,'AGB',fontsize=25,color='orange')
        ax2.hlines(21,0.7,2,linestyle='--',color='dodgerblue',lw=2)
        ax2.text(2.4,13.8,'IR GST',ha='right',fontsize=28)
        ax2.text(2.4,14.6,'No F275W',ha='right',fontsize=28)
                
        plt.tight_layout()
        plt.savefig(plotdir+'m33_selection_CMDs.pdf')

        
        
        ################################################
        # ---------- Density Maps -------------------- #
        ################################################
        select = ['RGB','AGB','MS']
        rawh,map = [],[]
        map_x,map_y = [],[]
        mapmin,mapmax = [],[]
        for slct in select:
                if slct == 'RGB':
                        # Scale inner counts (~2x for more stars in deeper cut @ threshold radius)
                        ra = np.hstack([ds.evaluate("RA",selection=slct),
                                        ds.evaluate("RA",selection=slct+'_inner')])
                        dec = np.hstack([ds.evaluate("DEC",selection=slct),
                                         ds.evaluate("DEC",selection=slct+'_inner')])
                        dra0,ddec0 = deg2diff(ra*u.deg,dec*u.deg,ra_m33,dec_m33)
                        dra,ddec = dra0.value,ddec0.value
                
                        h,hdy,hdx = np.histogram2d(ddec,dra,bins=200,density=False)
                        
                        map_x.append(hdx)
                        map_y.append(hdy)
                elif slct == 'AGB':
                        ra = ds.evaluate("RA",selection=slct)
                        dec = ds.evaluate("DEC",selection=slct)
                        dra0,ddec0 = deg2diff(ra*u.deg,dec*u.deg,ra_m33,dec_m33)
                        dra,ddec = dra0.value,ddec0.value
                        h,hdy2,hdx2 = np.histogram2d(ddec,dra,bins=100,density=False)
                else:
                        ra = ds.evaluate("RA",selection=slct)
                        dec = ds.evaluate("DEC",selection=slct)
                        dra0,ddec0 = deg2diff(ra*u.deg,dec*u.deg,ra_m33,dec_m33)
                        dra,ddec = dra0.value,ddec0.value
                        h,_,_ = np.histogram2d(ddec,dra,bins=[hdy,hdx],density=False)
                
                # Smooth and scale each map
                f = gaussian_filter(h,1)
                smin,smax = np.percentile(f.ravel()[h.ravel()>0],0.97),f.ravel().max()
                hs = img_scale.log(f,scale_min=smin,scale_max=smax)
                
                rawh.append(h)
                map.append(hs)
                mapmin.append(smin)
                mapmax.append(smax)
        map_x,map_y = np.asarray(map_x)[0],np.asarray(map_y)[0]
        pixa = (map_x[1]-map_x[0]) * (map_y[1]-map_y[0]) * 3600.**2
        pixa2 = pixa*4.
        
        # ----- Plot -------------
        fig = plt.figure()
        fig.set_figheight(10)
        fig.set_figwidth(27)
        bw = 'black'
        
        gs = gridspec.GridSpec(1,3,width_ratios=[1,1,1],
                               left=0.06,right=0.95,top=0.99,bottom=0.11,
                               wspace=0.185)
        
        # Plot RGB map
        ax1 = plt.subplot(gs[0])
        ax1.set_xlim(np.max(map_x),np.min(map_x))
        ax1.set_ylim(np.min(map_y),np.max(map_y))
        ax1.set_xlabel('RA')
        ax1.set_ylabel('Dec')
        xticks,yticks = diff2deg(ax1.get_xticks()*u.deg,ax1.get_yticks()*u.deg,ra_m33,dec_m33)
        xlabels = [str(round(tick.value,2)) for tick in xticks]
        ax1.set_xticklabels(xlabels)
        ylabels = [str(round(tick.value,2)) for tick in yticks]
        ax1.set_yticklabels(ylabels)

        cmap1 = mpl.cm.get_cmap('inferno')
        cmap1.set_under(color='white')
        cmap1.set_over(color='white')

        im = ax1.imshow(map[0],interpolation='nearest',origin='lower',
                        aspect='auto',cmap=cmap1,vmin=0.01,vmax=1.,
                        extent=[map_x[0],map_x[-1],map_y[0],map_y[-1]])
        
        # RGB colorbar
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", "5%")
        cbar = plt.colorbar(im,cax=cax)
        cbar.ax.tick_params(labelsize=28)
        cbar.set_label(r'$\Sigma_{\rm RGB}\ ({\rm arcsec^{-2}})$',fontsize=28,
                rotation=270.,ha='center',va='center',labelpad=30)
        cbints = 10.**np.arange(-1,np.log10(mapmax[0]/pixa),0.5)*pixa
        cbticks = np.log10(cbints)/np.log10(mapmax[0]-mapmin[0])
        cbticks[0] += 0.01
        cbar.set_ticks(cbticks)
        cblabels = [str(np.round(tick/pixa,1)) for tick in cbints]
        cblabels[-2:] = [lbl[:-2] for lbl in cblabels[-2:]]
        cbar.set_ticklabels(cblabels)
        
        ax1.text(-0.075,0.16,'RGB',color='black',fontsize=35,ha='center')
        
        # Plot AGB map
        ax2 = plt.subplot(gs[1])
        ax2.set_xlim(np.max(hdx2),np.min(hdx2))
        ax2.set_ylim(np.min(hdy2),np.max(hdy2))
        ax2.set_xlabel('RA')
        ax2.set_yticklabels([])
        ax2.set_xticklabels(xlabels)

        cmap2 = mpl.cm.get_cmap('magma')
        cmap2.set_under(color='white')
        cmap2.set_over(color='white')

        im = ax2.imshow(map[1],interpolation='nearest',origin='lower',
                        aspect='auto',cmap=cmap2,vmin=0.01,vmax=1.,
                        extent=[hdx2[0],hdx2[-1],hdy2[0],hdy2[-1]])
              
        # AGB colorbar
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", "5%")
        cbar = plt.colorbar(im,cax=cax)
        cbar.ax.tick_params(labelsize=28)
        cbar.set_label(r'$\Sigma_{\rm AGB}\ ({\rm arcsec^{-2}})$',fontsize=28,
                rotation=270.,ha='center',va='center',labelpad=25)
        cbints = 10.**np.arange(-2,np.log10(mapmax[1]/pixa),0.5)*pixa2
        cbticks = np.log10(cbints)/np.log10(mapmax[1]-mapmin[1])
        cbticks[0] += 0.01
        cbar.set_ticks(cbticks)
        cblabels = [str(np.round(tick/pixa2,2)) for tick in cbints]
        cblabels[-1:] = [lbl[:-2] for lbl in cblabels[-1:]]
        cbar.set_ticklabels(cblabels)

        ax2.text(-0.075,0.16,'AGB',color='black',fontsize=35,ha='center')
        
        # Plot MS map
        ax3 = plt.subplot(gs[2])
        ax3.set_xlim(np.max(map_x),np.min(map_x))
        ax3.set_ylim(np.min(map_y),np.max(map_y))
        ax3.set_xlabel('RA')
        ax3.set_yticklabels([])
        ax3.set_xticklabels(xlabels)

        cmap3 = mpl.cm.get_cmap('plasma_r')
        cmap3.set_under(color='white')

        im = ax3.imshow(map[2],interpolation='nearest',origin='lower',
                        aspect='auto',cmap=cmap3,vmin=0.01,vmax=1,
                        extent=[map_x[0],map_x[-1],map_y[0],map_y[-1]])
        
        # MS colorbar
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", "5%")
        cbar = plt.colorbar(im,cax=cax)
        cbar.ax.tick_params(labelsize=28)
        cbar.set_label(r'$\Sigma_{\rm MS}\ ({\rm arcsec^{-2}})$',fontsize=28,
                       rotation=270.,ha='center',va='center',labelpad=30)
        cbints = 10.**np.arange(-1,np.log10(mapmax[2]/pixa),0.5)*pixa
        cbticks = np.log10(cbints)/np.log10(mapmax[2]-mapmin[1])
        cbticks[0] += 0.01
        cbar.set_ticks(cbticks)
        cblabels = [str(np.round(tick/pixa,1)) for tick in cbints]
        cblabels[-1:] = [lbl[:-2] for lbl in cblabels[-1:]]
        cbar.set_ticklabels(cblabels)
        
        ax3.text(-0.075,0.16,'MS',color='black',fontsize=35,ha='center')
        
        plt.savefig(plotdir+'m33_RGB-AGB-MS_map.pdf')


        
        ################################################
        # ---------- AGB/RGB Ratio ------------------- #
        ################################################
        # Calculate raw ratio
        ra = ds.evaluate("RA",selection='RGB_bright')
        dec = ds.evaluate("DEC",selection='RGB_bright')
        dra0,ddec0 = deg2diff(ra*u.deg,dec*u.deg,ra_m33,dec_m33)
        dra,ddec = dra0.value,ddec0.value
        hrgb,hdy,hdx = np.histogram2d(ddec,dra,bins=65,density=False)

        ra2 = ds.evaluate("RA",selection='AGB')
        dec2 = ds.evaluate("DEC",selection='AGB')
        dra02,ddec02 = deg2diff(ra2*u.deg,dec2*u.deg,ra_m33,dec_m33)
        dra2,ddec2 = dra02.value,ddec02.value
        hagb,_,_ = np.histogram2d(ddec2,dra2,bins=[hdy,hdx],density=False)
        
        ratio = hagb/hrgb
        
        # Remove low-S/N pixels and smooth
        bd, = np.where((hagb.ravel()<3)|(hrgb.ravel()<3))
        rratio = ratio.ravel()
        rratio[bd] = 0.
        ratio = rratio.reshape(ratio.shape)
        fratio = median_filter(ratio,7,mode='mirror')
        
        # Clip edge pixels
        fdx,fdy = (np.meshgrid(hdx[:-1],hdy[:-1])[0].ravel(),
                   np.meshgrid(hdx[:-1],hdy[:-1])[1].ravel())
        wh, = np.where( (np.sqrt(fdx**2+fdy**2)>0.07) & (fratio.ravel()<0.07) )
        rfratio = fratio.ravel()
        rfratio[wh] = 0.
        fratio = rfratio.reshape(fratio.shape)
        
        # Calculate radial profile (Inclination and PA from Corbelli & Walterbos 2007)
        inc,pa = Angle('47.5 deg').rad,Angle('73.98 deg').rad
        ba = np.arccos(inc)
        r = np.arange(0,0.2,0.008)
        rphys,ratio_radial = [],[]
        for i in range(np.size(r)):
                if i > 0:
                        eq1_rgb = (dra*np.cos(pa)+ddec*np.sin(pa))**2 / r[i]**2 + \
                                  (dra*np.sin(pa)-ddec*np.cos(pa))**2 / (r[i]*ba)**2
                        eq2_rgb = (dra*np.cos(pa)+ddec*np.sin(pa))**2 / r[i-1]**2 + \
                                  (dra*np.sin(pa)-ddec*np.cos(pa))**2 / (r[i-1]*ba)**2
                        wh_rgb, = np.where( (eq1_rgb <= 1) & (eq2_rgb > 1) )
                                                        
                        eq1_agb = (dra2*np.cos(pa)+ddec2*np.sin(pa))**2 / r[i]**2 + \
                                  (dra2*np.sin(pa)-ddec2*np.cos(pa))**2 / (r[i]*ba)**2
                        eq2_agb = (dra2*np.cos(pa)+ddec2*np.sin(pa))**2 / r[i-1]**2 + \
                                  (dra2*np.sin(pa)-ddec2*np.cos(pa))**2 / (r[i-1]*ba)**2
                        wh_agb, = np.where( (eq1_agb <= 1) & (eq2_agb > 1) )
                        
                        ratio_radial.append(np.size(wh_agb)/np.size(wh_rgb))
                        rphys.append( (r[i]/np.cos(inc))*np.pi/180. * dm33.value )
                                                
        rphys = np.hstack(rphys)
        ratio_radial = np.hstack(ratio_radial)
        
        # ---- Plot ----
        fig = plt.figure()
        fig.set_figheight(10)
        fig.set_figwidth(18)
        
        gs = gridspec.GridSpec(1,2,width_ratios=[1,1],
                       left=0.08,right=0.99,top=0.97,bottom=0.1,
                       wspace=0.1)
        
        # Plot AGB/RGB map
        ax = plt.subplot(gs[0])
        ax.set_xlim(np.max(hdx),np.min(hdx))
        ax.set_ylim(np.min(hdy),np.max(hdy))
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        xticks,yticks = diff2deg(ax.get_xticks()*u.deg,ax.get_yticks()*u.deg,ra_m33,dec_m33)
        xlabels = [str(round(tick.value,2)) for tick in xticks]
        ax.set_xticklabels(xlabels)
        ylabels = [str(round(tick.value,2)) for tick in yticks]
        ax.set_yticklabels(ylabels)
        
        cmap = mpl.cm.get_cmap('PuOr')
        cmap.set_under(color='white')
        
        im = ax.imshow(fratio,
                  interpolation='nearest',origin='lower',aspect='auto',
                  cmap=cmap,vmin=0.05,vmax=0.11,
                  extent=[hdx[0],hdx[-1],hdy[0],hdy[-1]])
        
        # Representative ellipse
        phi = np.linspace(0,2.*np.pi,10000)
        x = 0.08*np.cos(phi)
        y = 0.08*ba * np.sin(phi)
        xp = x*np.cos(pa) - y*np.sin(pa)
        yp = x*np.sin(pa) + y*np.cos(pa)
        ax.plot(xp,yp,'--',color='red',lw=2.5)
        
        # Plot approximate Bar angle and extent (from Corbelli & Walterbos 2007)
        xb = 0.0263*np.cos(phi)
        yb = 0.0263*(1.-0.3) * np.sin(phi)
        theta = 10.*np.pi/180.
        xbp = xb*np.cos(theta) + yb*np.sin(theta)
        ybp = xb*np.sin(theta) - yb*np.cos(theta)
        ax.plot(xbp,ybp,'--',color='blue',lw=2.5)
        
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%")
        cbar = plt.colorbar(im,cax=cax)
        cbar.ax.set_yticklabels([])
        
        # Plot radial profile
        ax2 = plt.subplot(gs[1])
        ax2.set_xlim(0,4)
        ax2.set_ylim(0.0501,0.11)
        ax2.set_xlabel('SMA (kpc)')
        
        ax2.scatter(rphys,ratio_radial,s=600,c=ratio_radial,
                    cmap=cmap,vmin=0.05,vmax=0.11,edgecolors='0.5')
        ax2.vlines(0.6,0.0501,0.09,color='blue',linestyle='--',lw=2.5)
        ax2.text(0.6,0.092,r'$\sim$Bar',ha='center',fontsize=25,color='blue')
        
        fig.text(0.615,0.9,r'$\rm\bf AGB/RGB$',ha='center',fontsize=25)
        plt.savefig(plotdir+'m33_AGB-to-RGB.pdf')
