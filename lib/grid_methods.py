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
    return x_f

def gridu_to_v(x_u,umask):
    """
    4-point interpolation of variable from u-grid to v-grid
    """
    x_v=(x_u*umask+im1(jp1(x_u*umask))+im1(x_u*umask)+jp1(x_u*umask))/4
    return x_v

def gridv_to_u(x_v,vmask):
    """
    4-point interpolation of variable from u-grid to v-grid
    """
    x_u=(x_v*vmask+ip1(jm1(x_v*vmask))+ip1(x_v*vmask)+jm1(x_v*vmask))/4
    return x_v

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
    return var_zint

def zmean(var,e3):
    var_zmean=zint(var,e3)/e3.sum(dim='lev')
    return var_zmean

def n_point_smoothing(meshmask,var):
    return var_smooth

