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
    fig=plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
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
    #plt.savefig(name+'.eps')
    plt.savefig(name+'.png',bbox_inches='tight',dpi=100)
    return cbar

def section_contour(x,y,data,levs,name):
    cs=plt.contour(x,y,data,levs,colors='w',linewidths=1)
    plt.clabel(cs,levs,fmt='%1.0f cm/s', fontsize=10)
    #plt.savefig(name+'.eps')
    plt.savefig(name+'.png',bbox_inches='tight',dpi=100)

def global_map(lons,lats,data,levs,legend,title,colormap,name,proj):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=253.5))
    #gl = ax.gridlines(draw_labels=True, xlocs=np.arange(-150, 151, 60), ylocs=np.arange(-90, 91, 15))
    ax.add_feature(cfeature.LAND.with_scale('110m'))
    cmap = plt.get_cmap(colormap)
    norm = BoundaryNorm(levs, ncolors=cmap.N, clip=True)
    mesh=ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm,
                transform=ccrs.PlateCarree())
    #cbar=plt.colorbar(mesh,location='bottom',pad="6%",ticks=levs)
    cbar=plt.colorbar(mesh, orientation='horizontal', shrink=0.5,ticks=levs[0::2])
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(legend,fontsize=14)
    ax.coastlines()
    ax.set_global()
    plt.title(title,y=1.04,fontsize=14)
    #plt.savefig(name+'.eps')
    plt.savefig(name+'.png',bbox_inches='tight',dpi=100)
    return ax,cbar

def global_quiver(ax,x,y,u,v,name,col):
    x=np.array(x); y=np.array(y); u=np.array(u); v=np.array(v);
    proj = ccrs.Robinson(central_longitude=253.5)
    src_crs = ccrs.PlateCarree()
    u_rot, v_rot = proj.transform_vectors(src_crs,x,y,u/np.cos(y/180*np.pi),v)
    renorm = np.sqrt((u**2 + v**2) / (u_rot**2 + v_rot**2))
    ax.quiver(x[::6,::6],y[::6,::6],u_rot[::6,::6]*renorm[::6,::6],v_rot[::6,::6]*renorm[::6,::6],
            transform=ccrs.PlateCarree(),color=col,scale=4000,headwidth=2,width=0.002)
    #plt.savefig(name+'.eps')
    plt.savefig(name+'.png',bbox_inches='tight',dpi=100)

def global_contour(ax,x,y,v,levs,fmt,name,col):
    cs=ax.contour(x,y,v,levs,colors=col,linewidths=1,transform=ccrs.PlateCarree())
    plt.clabel(cs,levs,fmt=fmt, fontsize=10)
    #plt.savefig(name+'.eps')
    plt.savefig(name+'.png',bbox_inches='tight',dpi=100)

def global_hatching(ax,x,y,u_bot,levs_ubot,name):
    ax.contourf(x,y,u_bot,levs_ubot,colors='None',hatches=['.'],transform=ccrs.PlateCarree())
    #plt.savefig(name+'.eps')
    plt.savefig(name+'.png',bbox_inches='tight',dpi=100)

def atl_line(ax,x,y,name):
    ax.plot(x,y,'k',linewidth=2,transform=ccrs.PlateCarree())
    plt.savefig(name+'.png',bbox_inches='tight',dpi=100)

