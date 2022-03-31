#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

def vertical_section(x,y,data,levs,label,xlab,ylab,tit,col,name,axis):
    data=np.ma.masked_invalid(data)
    y_stretch=xr.where(y>-200,y,-200+(y+200)/5)
    ticks=np.concatenate((-200+(np.array([-6000, -5000, -4000, -3000, -2000, -1000, -500])+200)/5,
                np.array([-200, -150, -100, -50, 0])))
    ticklabels=[-6000, -5000, -4000, -3000, -2000, -1000, -500, -200, -150, -100, -50, 0]
    fig=plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    #plt.figure(figsize=(12, 5))
    method=getattr(plt.cm,col)
    norm = BoundaryNorm(levs,ncolors=method.N,clip=True)
    cs = plt.pcolormesh(x,y_stretch,data,vmin=levs[0],vmax=levs[-1],cmap=method,norm=norm,linewidth=0)
    cbar=plt.colorbar(cs)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label,fontsize=15)
    plt.ylabel(ylab,fontsize=15)
    plt.title(tit,fontsize=15)
    plt.xlim(axis[0],axis[1])
    plt.ylim(-200+(axis[2]+200)/5,axis[3])
    plt.yticks(ticks)
    ax.set_yticklabels(ticklabels)
    plt.xlabel(xlab,fontsize=15)
    #plt.set_size_inches(18.5, 10.5)
    plt.savefig(name+'.eps')
    plt.savefig(name+'.png',bbox_inches='tight',dpi=100)
    return cbar

def global_map(lons,lats,data,levs,legend,title,colormap,name,proj):
    fig = plt.figure(figsize=(10, 7))
    #exec('ax = fig.add_subplot(1, 1, 1, projection=ccrs.'+proj+'())')
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=253.5))
    #gl = ax.gridlines(draw_labels=True, xlocs=np.arange(-150, 151, 60), ylocs=np.arange(-90, 91, 15))
    ax.add_feature(cfeature.LAND.with_scale('110m'))
    cmap = plt.get_cmap(colormap)
    norm = BoundaryNorm(levs, ncolors=cmap.N, clip=True)
    mesh=ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm,
                transform=ccrs.PlateCarree())
    #cbar=plt.colorbar(mesh,location='bottom',pad="6%",ticks=levs)
    cbar=plt.colorbar(mesh, orientation='horizontal', shrink=0.8,ticks=levs[0::2])
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(legend,fontsize=14)
    ax.coastlines()
    ax.set_global()
    plt.title(title,y=1.04,fontsize=14)
    plt.savefig(name+'.eps')
    plt.savefig(name+'.png',bbox_inches='tight',dpi=100)
    return ax,cbar

