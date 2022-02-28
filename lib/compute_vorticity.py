"""
compute_vorticity.py

Compute the vorticity balances:
    - BT momentum balance and 3 derived BV balances
    - Depth-dependent vorticity balance and derived depth-integral vorticity balance
"""

import numpy as np
import xarray as xr
import sys
import os
sys.path.append(os.path.abspath("../lib"))
from general_methods import norm
from grid_methods import gridt_to_f,gridf_to_t,gridu_to_v,gridv_to_u,ip1,jp1,im1,jm1,curl,zint
from pvo_methods import masked_f_e3f,f_triads,pvo_offline,pvo_contrib,ztrdpvo_contrib,pvo_contrib_int,Momentum_balances,BT_momentum_balances,Vorticity_balances,BV_balances,Transp_vorticity_balances

rho0=1025

def compute_vorticity(meshmask,utrd,u,tau):
    """
    Wrapping function that computes the vorticity balances
    """

    ### Method 1: depth-integral vorticity balance

    # Decomposition of the depth-dependent Coriolis trend:
    utrd2=pvo_decomposition(meshmask,u)

    # Computation of the depth-dependent vorticity balance with Coriolis decomposition:
    ztrd,ztrd2=Vorticity_balance(meshmask,utrd,tau.tauuo,tau.tauvo,utrd2,meshmask.f_u,meshmask.f_v)

    # Computation of the depth-integral vorticity balance:
    ztrd_int,ztrd2_int=Integral_vorticity_balance(meshmask,ztrd,ztrd2)

    ### Methods 2, 3 and 4: BT vorticity balances from the BT momentum balance divided by 1 (BV balance), h (depth average vorticity balance) or f (transport vorticity balance)

    # Decomposition of the depth-integral, depth-average and transport Coriolis trends:
    utrd2_int=BT_pvo_decomposition(meshmask,u,utrd2,xr.ones_like(meshmask.gphit),xr.ones_like(meshmask.gphit))
    utrd2_av=BT_pvo_decomposition(meshmask,u,utrd2,meshmask.bathy_u,meshmask.bathy_v)
    utrd2_transp=BT_pvo_decomposition(meshmask,u,utrd2,meshmask.f_u_nan,meshmask.f_v)

    # Computation of the depth-integral, depth-average and transport momentum trend:
    utrd_int=BT_momentum_balance(meshmask,utrd,tau,xr.ones_like(meshmask.gphit),xr.ones_like(meshmask.gphit))
    utrd_av=BT_momentum_balance(meshmask,utrd,tau,meshmask.bathy_u,meshmask.bathy_v)
    utrd_transp=BT_momentum_balance(meshmask,utrd,tau,meshmask.f_u_nan,meshmask.f_v)

    # Computation of the BV balance, depth-average vorticity balance and transport vorticity balance:
    curl_utrd_int,curl_utrd2_int=Vorticity_balance(meshmask,utrd_int,utrd_int.tauuo,utrd_int.tauvo,utrd2_int,meshmask.f_u,meshmask.f_v)
    curl_utrd_av,curl_utrd2_av=Vorticity_balance(meshmask,utrd_av,utrd_av.tauuo,utrd_av.tauvo,utrd2_av,meshmask.f_u/meshmask.bathy_u,meshmask.f_v/meshmask.bathy_v)
    curl_utrd_transp,curl_utrd2_transp=Vorticity_balance(meshmask,utrd_transp,utrd_transp.tauuo,utrd_transp.tauvo,utrd2_transp,xr.ones_like(meshmask.gphit),xr.ones_like(meshmask.gphit))

    ### 4-point interpolation back to T-grid and ~1000km-smoothing
    ztrd,ztrd2,ztrd_int,ztrd2_int,curl_utrd_int,curl_utrd2_int,curl_utrd_av,curl_utrd2_av,curl_utrd_transp,curl_utrd2_transp=gridftot(ztrd,ztrd2,ztrd_int,ztrd2_int,curl_utrd_int,curl_utrd2_int,curl_utrd_av,curl_utrd2_av,curl_utrd_transp,curl_utrd2_transp,meshmask)

    ### Diagnosis of dominant balances for all four methods
    balances=Dynamical_balances(utrd,utrd2,utrd_int,utrd2_int,ztrd,ztrd2,ztrd_int,ztrd2_int,curl_utrd_int,curl_utrd2_int,curl_utrd_av,curl_utrd2_av,curl_utrd_transp,curl_utrd2_transp,meshmask)

    return utrd2,ztrd,ztrd2,ztrd_int,ztrd2_int,utrd2_int,utrd2_av,utrd2_transp,utrd_int,utrd_av,utrd_transp,curl_utrd_int,curl_utrd2_int,curl_utrd_av,curl_utrd2_av,curl_utrd_transp,curl_utrd2_transp,balances

def pvo_decomposition(meshmask,u):
    """
    Decomposition of the depth-dependent Coriolis trend
    """

    ### Offline Coriolis as computed online in NEMO with the EEN scheme

    # 0-masked e3f
    zwz=masked_f_e3f(meshmask)

    # f triads in NE, NW, SE and SW corners
    ztne,ztnw,ztse,ztsw=f_triads(zwz)

    # Coriolis trend as in NEMO's EEN scheme
    utrdpvo_full,vtrdpvo_full=pvo_offline(meshmask,u,ztne,ztnw,ztse,ztsw)

    ### Separation of u-v and f contributions
    u1nm,u2nm,u3nm,u4nm,u1,u2,u3,u4,v1nm,v2nm,v3nm,v4nm,v1,v2,v3,v4,fu1,fu2,fu3,fu4,fv1,fv2,fv3,fv4=pvo_contrib(meshmask,u,ztne,ztnw,ztse,ztsw,utrdpvo_full,vtrdpvo_full)

    ### Decomposition of the Coriolis trend
    utrdpvo_phys=xr.where(u.uo!=0,gridv_to_u(u.vo,meshmask.vmask)*meshmask.f_u,np.nan);
    vtrdpvo_phys=xr.where(u.vo!=0,-gridu_to_v(u.uo,meshmask.umask)*meshmask.f_v,np.nan);
    utrdpvo_num=utrdpvo_full-utrdpvo_phys
    vtrdpvo_num=vtrdpvo_full-vtrdpvo_phys

    ### Storage into a dataset object
    utrd_pvo_names=['utrdpvo_phys','utrdpvo_num','utrdpvo_full','vtrdpvo_phys','vtrdpvo_num','vtrdpvo_full',
            'u1nm','u2nm','u3nm','u4nm','v1nm','v2nm','v3nm','v4nm',
            'u1','u2','u3','u4','v1','v2','v3','v4',
            'fu1','fu2','fu3','fu4','fv1','fv2','fv3','fv4','ztne','ztnw','ztse','ztsw']
    utrd2=[]
    for name in utrd_pvo_names:
        utrd2.append(xr.DataArray(eval(name),name=name))
    utrd2=xr.merge(utrd2)

    return utrd2

def Vorticity_balance(meshmask,utrd,tauuo,tauvo,utrd2,f_u_factor,f_v_factor):
    """
    Computation of the depth-dependent vorticity balance with Coriolis decomposition
    """

    ### Computation of the vorticity of the momentum trend
    utrd_names=['hpg','keg','ldf','pvo','rvo','spg','tot','zad','zdf']
    data = []
    for name in utrd_names:
        ds = curl(eval('utrd.utrd'+name),eval('utrd.vtrd'+name),meshmask)
        data.append(xr.DataArray(ds,name='ztrd'+name))
    data.append(xr.DataArray(curl(tauuo,tauvo,meshmask),name='ztrdtau'))
    ztrd = xr.merge(data)

    ### Decomposition of the Coriolis vorticity trend
    ztrd2=ztrdpvo_contrib(utrd2.utrdpvo_full,utrd2.vtrdpvo_full,utrd2.u1nm,utrd2.u2nm,utrd2.u3nm,utrd2.u4nm,utrd2.u1,utrd2.u2,utrd2.u3,utrd2.u4,utrd2.v1nm,utrd2.v2nm,utrd2.v3nm,utrd2.v4nm,utrd2.v1,utrd2.v2,utrd2.v3,utrd2.v4,utrd2.fu1,utrd2.fu2,utrd2.fu3,utrd2.fu4,utrd2.fv1,utrd2.fv2,utrd2.fv3,utrd2.fv4,meshmask,f_u_factor,f_v_factor)

    return ztrd,ztrd2

def Integral_vorticity_balance(meshmask,ztrd,ztrd2):
    """
    Computation of the depth-integral vorticity balance
    """

    ztrd_int=zint(ztrd,meshmask.e3f)
    ztrd_int['ztrdtau']=ztrd.ztrdtau
    ztrd2_int=zint(ztrd2,meshmask.e3f)

    return ztrd_int,ztrd2_int

def BT_pvo_decomposition(meshmask,u,utrd2,factor_u,factor_v):
    """
    Decomposition of the depth-integral Coriolis trend
    """

    ### Depth-integral Coriolis trend as in NEMO's EEN scheme
    utrdpvo_full=zint(utrd2.utrdpvo_full,meshmask.e3u_0)/factor_u
    vtrdpvo_full=zint(utrd2.vtrdpvo_full,meshmask.e3v_0)/factor_v

    ### Depth-integration of u-v contribution and depth-average of f contributions times factor
    u1nm,u2nm,u3nm,u4nm,u1,u2,u3,u4,v1nm,v2nm,v3nm,v4nm,v1,v2,v3,v4,fu1,fu2,fu3,fu4,fv1,fv2,fv3,fv4=pvo_contrib_int(meshmask,utrd2,factor_u,factor_v)

    ### Decomposition of the Coriolis trend
    utrdpvo_phys=zint(u.vo*meshmask.f_u,meshmask.e3u_0)/factor_u; utrdpvo_phys.data[utrdpvo_full.data==0]=np.nan;
    vtrdpvo_phys=-zint(u.uo*meshmask.f_v,meshmask.e3v_0)/factor_v; vtrdpvo_phys.data[vtrdpvo_full.data==0]=np.nan;
    utrdpvo_num=utrdpvo_full-utrdpvo_phys
    vtrdpvo_num=vtrdpvo_full-vtrdpvo_phys

    ### Storage into a dataset object
    utrd_pvo_names=['utrdpvo_phys','utrdpvo_num','utrdpvo_full','vtrdpvo_phys','vtrdpvo_num','vtrdpvo_full',
            'u1nm','u2nm','u3nm','u4nm','v1nm','v2nm','v3nm','v4nm',
            'u1','u2','u3','u4','v1','v2','v3','v4','fu1','fu2','fu3','fu4','fv1','fv2','fv3','fv4']
    utrd2=[]
    for name in utrd_pvo_names:
        utrd2.append(xr.DataArray(eval(name),name=name))
    utrd2=xr.merge(utrd2)

    return utrd2

def BT_momentum_balance(meshmask,utrd,tau,factor_u,factor_v):
    """
    Computation of the depth-integral momentum trend
    """

    ### Computation of the depth-integral momentum trend and deduction of the depth-average and transport momentum trends
    utrd_names=['hpg','keg','ldf','pvo','rvo','spg','tot','zad','zdf']
    data = [];
    for name in utrd_names:
        ds = zint(eval('utrd.utrd'+name),meshmask.e3u_0)/factor_u
        data.append(xr.DataArray(ds,name='utrd'+name))
        ds = zint(eval('utrd.vtrd'+name),meshmask.e3v_0)/factor_v
        data.append(xr.DataArray(ds,name='vtrd'+name))
    data.append(xr.DataArray(tau.tauuo/factor_u,name='tauuo'))
    data.append(xr.DataArray(tau.tauvo/factor_v,name='tauvo'))
    utrd_int = xr.merge(data)

    return utrd_int

def gridftot(ztrd,ztrd2,ztrd_int,ztrd2_int,curl_utrd_int,curl_utrd2_int,curl_utrd_av,curl_utrd2_av,curl_utrd_transp,curl_utrd2_transp,meshmask):
    names=['ztrd','ztrd2','ztrd_int','ztrd2_int','curl_utrd_int','curl_utrd2_int','curl_utrd_av','curl_utrd2_av','curl_utrd_transp','curl_utrd2_transp']
    for name in names:
        exec(name+'=gridf_to_t('+name+',meshmask.e3f)')
    return ztrd,ztrd2,ztrd_int,ztrd2_int,curl_utrd_int,curl_utrd2_int,curl_utrd_av,curl_utrd2_av,curl_utrd_transp,curl_utrd2_transp

def Dynamical_balances(utrd,utrd2,utrd_int,utrd2_int,ztrd,ztrd2,ztrd_int,ztrd2_int,curl_utrd_int,curl_utrd2_int,curl_utrd_av,curl_utrd2_av,curl_utrd_transp,curl_utrd2_transp,meshmask):

    ### Depth-dependent momentum balance:
    utrd_balances=xr.DataArray(Momentum_balances(norm(utrd.utrdpvo,utrd.vtrdpvo),
            norm(utrd.utrdhpg,utrd.vtrdhpg),
            norm(utrd2.utrdpvo_num,utrd2.vtrdpvo_num),
            norm(utrd.utrdkeg+utrd.utrdrvo+utrd.utrdzad,utrd.vtrdkeg+utrd.vtrdrvo+utrd.vtrdzad),
            norm(utrd.utrdldf,utrd.vtrdldf),
            norm(utrd.utrdzdf+utrd.utrdspg,utrd.vtrdzdf+utrd.vtrdspg)),
            name='utrd_balances')

    ### Depth-integral momentum balance:
    utrd_int_balances=xr.DataArray(BT_momentum_balances(norm(utrd_int.utrdpvo,utrd_int.vtrdpvo),
            norm(utrd_int.utrdhpg,utrd_int.vtrdhpg),
            norm(utrd2_int.utrdpvo_num,utrd2_int.vtrdpvo_num),
            norm(utrd_int.utrdkeg+utrd_int.utrdrvo+utrd_int.utrdzad,utrd_int.vtrdkeg+utrd_int.vtrdrvo+utrd_int.vtrdzad),
            norm(utrd_int.utrdldf,utrd_int.vtrdldf),
            norm(utrd_int.tauuo,utrd_int.tauvo)/rho0,
            norm(utrd_int.utrdzdf+utrd_int.utrdspg-utrd_int.tauuo/rho0,utrd_int.vtrdzdf+utrd_int.vtrdspg-utrd_int.tauvo/rho0)),
            name='utrd_int_balances')

    ### Depth-dependent vorticity balance:
    ztrd_topo_stretch=xr.ones_like(ztrd2.ztrd_stretchphys)*zint(ztrd2.ztrd_stretchphys*meshmask.tmask,meshmask.e3t_0)/zint(meshmask.tmask,meshmask.e3t_0)
    ztrd_balances=xr.DataArray(Vorticity_balances(np.abs(ztrd2.ztrd_stretchphys),
            np.abs(ztrd2.ztrd_betaphys),
            np.abs(ztrd2.ztrd_stretchnum+ztrd2.ztrd_betanum),
            np.abs(ztrd.ztrdkeg+ztrd.ztrdrvo+ztrd.ztrdzad),
            np.abs(ztrd.ztrdldf),
            np.abs(ztrd.ztrdzdf+ztrd.ztrdspg-ztrd_topo_stretch),
            np.abs(ztrd_topo_stretch)),
            name='ztrd_balances')

    ### Barotropic vorticity balances:
    ztrd_int_balances=xr.DataArray(BV_balances(np.abs(ztrd2_int.ztrd_betaphys),
            np.abs(ztrd2_int.ztrd_stretchnum+ztrd2_int.ztrd_betanum),
            np.abs(ztrd_int.ztrdkeg+ztrd_int.ztrdrvo+ztrd_int.ztrdzad),
            np.abs(ztrd_int.ztrdldf),
            np.abs(ztrd_int.ztrdtau/rho0),
            np.abs(ztrd_int.ztrdzdf+ztrd_int.ztrdspg-ztrd_int.ztrdtau/rho0),
            np.abs(ztrd2_int.ztrd_stretchphys)),
            name='ztrd_int_balances')
    curl_utrd_int_balances=xr.DataArray(BV_balances(np.abs(curl_utrd2_int.ztrd_betaphys),
            np.abs(curl_utrd2_int.ztrd_stretchnum+curl_utrd2_int.ztrd_betanum+curl_utrd2_int.ztrd_crossnum),
            np.abs(curl_utrd_int.ztrdkeg+curl_utrd_int.ztrdrvo+curl_utrd_int.ztrdzad),
            np.abs(curl_utrd_int.ztrdldf),
            np.abs(curl_utrd_int.ztrdtau/rho0),
            np.abs(curl_utrd_int.ztrdzdf+curl_utrd_int.ztrdspg-curl_utrd_int.ztrdtau/rho0),
            np.abs(curl_utrd_int.ztrdhpg)),
            name='curl_utrd_int_balances')
    curl_utrd_av_balances=xr.DataArray(BV_balances(np.abs(curl_utrd2_av.ztrd_betaphys),
            np.abs(curl_utrd2_av.ztrd_stretchnum+curl_utrd2_av.ztrd_betanum+curl_utrd2_av.ztrd_crossnum),
            np.abs(curl_utrd_av.ztrdkeg+curl_utrd_av.ztrdrvo+curl_utrd_av.ztrdzad),
            np.abs(curl_utrd_av.ztrdldf),
            np.abs(curl_utrd_av.ztrdtau/rho0),
            np.abs(curl_utrd_av.ztrdzdf+curl_utrd_av.ztrdspg-curl_utrd_av.ztrdtau/rho0),
            np.abs(curl_utrd_av.ztrdhpg)),
            name='curl_utrd_av_balances')
    curl_utrd_transp_balances=xr.DataArray(Transp_vorticity_balances(np.abs(curl_utrd_transp.ztrdhpg),
            np.abs(curl_utrd2_transp.ztrd_stretchnum+curl_utrd2_transp.ztrd_betanum+curl_utrd2_transp.ztrd_crossnum),
            np.abs(curl_utrd_transp.ztrdkeg+curl_utrd_transp.ztrdrvo+curl_utrd_transp.ztrdzad),
            np.abs(curl_utrd_transp.ztrdldf),
            np.abs(curl_utrd_transp.ztrdtau/rho0),
            np.abs(curl_utrd_transp.ztrdzdf+curl_utrd_transp.ztrdspg-curl_utrd_transp.ztrdtau/rho0)),
            name='curl_utrd_transp_balances')

    balances=xr.merge([utrd_balances,utrd_int_balances,ztrd_balances,ztrd_int_balances,curl_utrd_int_balances,curl_utrd_av_balances,curl_utrd_transp_balances])

    return balances
