#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from grid_methods import search_lon,search_lat,rolling_window
from mpl_toolkits.axes_grid1 import make_axes_locatable

def vertical_section(x,y,data,levs,label,xlab,ylab,tit,col,name,axis):
    data=np.ma.masked_invalid(data)
    y_stretch=xr.where(y>-200,y,-200+(y+200)/5)
    ticks=np.concatenate((-200+(np.array([-6000, -5000, -4000, -3000, -2000, -1000, -500])+200)/5,
                np.array([-200, -150, -100, -50, 0])))
    ticklabels=[-6000, -5000, -4000, -3000, -2000, -1000, -500, -200, -150, -100, -50, 0]
    fig=plt.figure(figsize=(20, 10))
    plt.rcParams.update({'font.size': 20})
    ax = fig.add_subplot(111)
    method=getattr(plt.cm,col)
    norm = BoundaryNorm(levs,ncolors=method.N,clip=True)
    cs = plt.pcolormesh(x,y_stretch,data,vmin=levs[0],vmax=levs[-1],cmap=method,norm=norm,linewidth=0)
    cbar=plt.colorbar(cs)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(label)
    plt.ylabel(ylab)
    plt.title(tit)
    plt.yticks(ticks)
    ax.set_yticklabels(ticklabels)
    plt.xlabel(xlab)
    plt.xlim(axis[0],axis[1])
    plt.ylim(-200+(axis[2]+200)/5,axis[3])
    plt.savefig(name+'.png',bbox_inches='tight',dpi=100)
    return cbar

def section_contour(x,y,data,levs,name):
    cs=plt.contour(x,y,data,levs,colors='k',linewidths=1)
    plt.clabel(cs,levs,fmt='%1.0f cm/s', fontsize=15)
    plt.savefig(name+'.png',bbox_inches='tight',dpi=100)

def global_map(lons,lats,data,levs,legend,title,colormap,name,proj):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=253.5))
    ax.add_feature(cfeature.LAND.with_scale('110m'))
    cmap = plt.get_cmap(colormap)
    norm = BoundaryNorm(levs, ncolors=cmap.N, clip=True)
    mesh=ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm,
                transform=ccrs.PlateCarree())
    cbar=plt.colorbar(mesh, orientation='horizontal', shrink=0.5,ticks=levs[0::2])
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(legend,fontsize=14)
    ax.coastlines()
    ax.set_global()
    plt.title(title,y=1.04,fontsize=14)
    plt.savefig(name+'.png',bbox_inches='tight',dpi=100)
    return ax,cbar

def global_quiver(ax,x,y,u,v,name,col):
    x=np.array(x); y=np.array(y); u=np.array(u); v=np.array(v);
    proj = ccrs.Mercator(central_longitude=253.5)
    src_crs = ccrs.PlateCarree()
    u_rot, v_rot = proj.transform_vectors(src_crs,x,y,u/np.cos(y/180*np.pi),v)
    renorm = np.sqrt((u**2 + v**2) / (u_rot**2 + v_rot**2))
    ax.quiver(x[::6,::6],y[::6,::6],u_rot[::6,::6]*renorm[::6,::6],v_rot[::6,::6]*renorm[::6,::6],
            transform=ccrs.PlateCarree(),color=col,scale=4000,headwidth=2,width=0.002)
    plt.savefig(name+'.png',bbox_inches='tight',dpi=100)

def global_contour(ax,x,y,v,levs,fmt,name,col):
    cs=ax.contour(x,y,v,levs,colors=col,linewidths=1,transform=ccrs.PlateCarree())
    plt.clabel(cs,levs,fmt=fmt, fontsize=10)
    plt.savefig(name+'.png',bbox_inches='tight',dpi=100)

def global_hatching(ax,x,y,u_bot,levs_ubot,name):
    ax.contourf(x,y,u_bot,levs_ubot,colors='None',hatches=['.'],transform=ccrs.PlateCarree())
    plt.savefig(name+'.png',bbox_inches='tight',dpi=100)

def atl_zonal(data,mesh,names,ylim,ylabel,title,name,width,xlim):
    # display of zonal profile at specific latitudes of the Atlantic
    cmap = plt.cm.get_cmap('Dark2')
    cols = cmap(np.linspace(0,1,8))
    plt.figure(figsize=(width, 8))
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams.update({'font.size': 18})
    plt.plot(mesh.glamt,data.ztrdad, color=cols[0])
    plt.plot(mesh.glamt,data.ztrdldf, color=cols[1])
    plt.plot(mesh.glamt,data.ztrdtau, color=cols[2])
    plt.plot(mesh.glamt,data.ztrdtaub, color=cols[3])
    plt.plot(mesh.glamt,data.ztrd_betaphys, color=cols[4])
    plt.plot(mesh.glamt,data.ztrd_stretchphys, color=cols[5])
    plt.plot(mesh.glamt,data.ztrdhpg, color=cols[6])
    plt.plot(mesh.glamt,data.ztrdnum, color=cols[7])
    plt.plot(mesh.glamt,(data.ztrdad+data.ztrdldf+data.ztrd_betaphys+data.ztrd_stretchphys+data.ztrdhpg+data.ztrdtau+data.ztrdtaub+data.ztrdnum), 'k')
    plt.ylim(ylim)
    plt.xlim(xlim[0],xlim[1])
    plt.xlabel('Longitude ($^{\circ}E$)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(name+'.png',bbox_inches='tight',dpi=100)
    plt.savefig(name+'.eps')
    plt.legend(names,loc='upper center',fontsize=20)
    plt.savefig(name+'_leg.eps')

def zonal_profile(data,meshmask,lim):
    lonmin=search_lon(lim[1],meshmask.glamt); lonmax=search_lon(lim[2],meshmask.glamt);
    meshmask_lon=meshmask.isel(x=slice(lonmin.data-5,lonmax.data+5))
    data_lon=data.isel(x=slice(lonmin.data-5,lonmax.data+5))
    lat=search_lat(lim[0],meshmask_lon.gphit);
    data_lonlat=data_lon.isel(y=lat.data)
    return data_lonlat

def atl_map_zonal_lines(lons,lats,data,levs,colormap,name):
    fig = plt.figure(figsize=(3.5, 10))
    plt.rcParams.update({'font.size': 10})
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator(central_longitude=253.5))
    gl = ax.gridlines(draw_labels=True, xlocs=np.arange(-150, 151, 15), ylocs=np.arange(-90, 91, 15))
    ax.set_extent((-100, 30, -30, 65), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale('110m'))
    cmap = plt.get_cmap(colormap)
    norm = BoundaryNorm(levs, ncolors=cmap.N, clip=True)
    mesh=ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm,
                transform=ccrs.PlateCarree())
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.35, axes_class=plt.Axes)
    fig.add_axes(ax_cb)
    cbar=plt.colorbar(mesh, cax=ax_cb, ticks=levs[0::4])
    plt.title('depth (m)')
    ax.coastlines()
    plt.savefig(name+'.png',bbox_inches='tight',dpi=400)
    return ax

def atl_line(ax,x,y,name):
    ax.plot(x,y,'k',linewidth=2,transform=ccrs.PlateCarree())
    plt.savefig(name+'.png',bbox_inches='tight',dpi=100)
