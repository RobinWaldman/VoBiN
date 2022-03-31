"""
Methods used to operate on the NEMO grid
"""

import numpy as np
import xarray as xr

def gridt_to_f(x_t,tmask):
    """
    4-point interpolation of variable from t-grid to f-grid
    """
    x_f=(x_t*tmask+ip1(jp1(x_t*tmask))+ip1(x_t*tmask)+jp1(x_t*tmask))/4
    return x_f

def gridf_to_t(x_f,fmask):
    """
    4-point interpolation of variable from t-grid to f-grid
    """
    x_t=(x_f*fmask+ip1(jp1(x_f*fmask))+ip1(x_f*fmask)+jp1(x_f*fmask))/4
    return x_t

def gridu_to_v(x_u,meshmask):
    """
    4-point ponderate interpolation of variable from u-grid to v-grid, designed for Coriolis trend
    """
    xu=x_u*meshmask.e2u*meshmask.e3u_0*meshmask.umask
    x_v=(xu+im1(jp1(xu))+im1(xu)+jp1(xu))/(4*meshmask.e2v*meshmask.e3v_0*meshmask.vmask)
    #x_v=(x_u*meshmask.umask+im1(jp1(x_u*meshmask.umask))+im1(x_u*meshmask.umask)+jp1(x_u*meshmask.umask))/4
    return x_v

def gridv_to_u(x_v,meshmask):
    """
    4-point ponderate interpolation of variable from u-grid to v-grid, designed for Coriolis trend
    """
    xv=x_v*meshmask.e1v*meshmask.e3v_0*meshmask.vmask
    x_u=(xv+ip1(jm1(xv))+ip1(xv)+jm1(xv))/(4*meshmask.e1u*meshmask.e3u_0*meshmask.umask)
    #x_u=(x_v*meshmask.vmask+im1(jp1(x_v*meshmask.vmask))+im1(x_v*meshmask.vmask)+jp1(x_v*meshmask.vmask))/4
    return x_u

def ip1(f):
    return f.roll(x=-1)

def im1(f):
    return f.roll(x=1)

def jp1(f):
    return f.roll(y=-1)

def jm1(f):
    return f.roll(y=1)

def ip12(f):
    return (f+ip1(f))/2

def jp12(f):
    return (f+jp1(f))/2

def di(f):
    return ip1(f)-f

def dj(f):
    return jp1(f)-f

def curl(uo,vo,meshmask):
    """
    Vertical curl operator as defined in NEMO
    """
    Zeta=(ip1(vo)*ip1(meshmask.e2v)-vo*meshmask.e2v-jp1(uo)*jp1(meshmask.e1u)+uo*meshmask.e1u)/meshmask.e1f/meshmask.e2f
    return Zeta

def zint(var,e3):
    var_zint=(var*e3).sum(dim='lev')
    var_zint=xr.where(var_zint!=0,var_zint,np.nan)
    return var_zint

def zmean(var,e3):
    var_zmean=zint(var,e3)/e3.sum(dim='lev')
    var_zmean=xr.where(var_zmean!=0,var_zmean,np.nan)
    return var_zmean

def rolling_window(mask,var,nb):
    var_smooth=var.rolling(x=nb, y=nb, min_periods=1).mean()*mask
    return var_smooth

def search_lon(lon0,lon):
    ilon0=np.argmin(np.abs(lon.mean(dim='y')-lon0))
    return ilon0

def ubot(u_norm,meshmask):
    x = np.arange(len(meshmask.x))
    y = np.arange(len(meshmask.y))
    X,Y = np.meshgrid(x,y)
    bottom_u=u_norm.isel(lev=0).copy()
    bottom_u.data=u_norm.data[meshmask.mbathy-1,Y,X]
    return bottom_u

def ucline(u_norm,meshmask):
    u_cline=xr.where(u_norm>5e-3,meshmask.gdept_0.data,np.nan)
    u_cline=u_cline.max(dim='lev')
    return u_cline

