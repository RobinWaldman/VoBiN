"""
Methods used to operate on the NEMO grid
"""

import numpy as np
import xarray as xr
from grid_methods import gridt_to_f,ip1,jp1,im1,jm1,ip12,jp12,di,dj,curl,zint,zmean

def masked_f_e3f(meshmask):
    ze3=meshmask.e3t_0*meshmask.tmask+jp1(meshmask.e3t_0*meshmask.tmask)+ip1(meshmask.e3t_0*meshmask.tmask)+jp1(ip1(meshmask.e3t_0*meshmask.tmask))
    zmsk=meshmask.tmask+jp1(meshmask.tmask)+ip1(meshmask.tmask)+jp1(ip1(meshmask.tmask))
    ze3f=xr.where(ze3>0,zmsk/ze3,0);
    zwz=ze3f*meshmask.f_f
    return zwz

def f_triads(zwz):
    ztne=im1(zwz)+zwz+jm1(zwz)
    ztnw=im1(jm1(zwz))+im1(zwz)+zwz
    ztse=zwz+jm1(zwz)+im1(jm1(zwz))
    ztsw=jm1(zwz)+im1(jm1(zwz))+im1(zwz)
    return ztne,ztnw,ztse,ztsw

def pvo_offline(meshmask,u,ztne,ztnw,ztse,ztsw):
    zwx=u.uo*meshmask.e2u*meshmask.e3u_0
    zwy=u.vo*meshmask.e1v*meshmask.e3v_0
    utrdpvo_full=1/12*(ztne*zwy+ip1(ztnw)*ip1(zwy)+ztse*jm1(zwy)+ip1(ztsw)*jm1(ip1(zwy)))/meshmask.e1u*meshmask.umask_nan
    vtrdpvo_full=-1/12*(jp1(ztsw)*jp1(im1(zwx))+jp1(ztse)*jp1(zwx)+ztnw*im1(zwx)+ztne*zwx)/meshmask.e2v*meshmask.vmask_nan
    return utrdpvo_full,vtrdpvo_full

def pvo_contrib(meshmask,u,ztne,ztnw,ztse,ztsw,utrdpvo_full,vtrdpvo_full):
    u1nm=1/4*jp1(im1(u.uo))*jp1(im1(meshmask.e2u))/meshmask.e2v; u1=u1nm*meshmask.vmask_nan
    u2nm=1/4*jp1(u.uo)*jp1(meshmask.e2u)/meshmask.e2v; u2=u2nm*meshmask.vmask_nan
    u3nm=1/4*im1(u.uo)*im1(meshmask.e2u)/meshmask.e2v; u3=u3nm*meshmask.vmask_nan
    u4nm=1/4*u.uo*meshmask.e2u/meshmask.e2v; u4=u4nm*meshmask.vmask_nan

    fv1=1/3*jp1(ztsw)*jp1(im1(meshmask.e3u_0))*meshmask.vmask_nan
    fv2=1/3*jp1(ztse)*jp1(meshmask.e3u_0)*meshmask.vmask_nan
    fv3=1/3*ztnw*im1(meshmask.e3u_0)*meshmask.vmask_nan
    fv4=1/3*ztne*meshmask.e3u_0*meshmask.vmask_nan

    v1nm=1/4*u.vo*meshmask.e1v/meshmask.e1u; v1=v1nm*meshmask.umask_nan
    v2nm=1/4*ip1(u.vo)*ip1(meshmask.e1v)/meshmask.e1u; v2=v2nm*meshmask.umask_nan
    v3nm=1/4*jm1(u.vo)*jm1(meshmask.e1v)/meshmask.e1u; v3=v3nm*meshmask.umask_nan
    v4nm=1/4*jm1(ip1(u.vo))*jm1(ip1(meshmask.e1v))/meshmask.e1u; v4=v4nm*meshmask.umask_nan

    fu1=1/3*ztne*meshmask.e3v_0*meshmask.umask_nan
    fu2=1/3*ip1(ztnw)*ip1(meshmask.e3v_0)*meshmask.umask_nan
    fu3=1/3*ztse*jm1(meshmask.e3v_0)*meshmask.umask_nan
    fu4=1/3*ip1(ztsw)*jm1(ip1(meshmask.e3v_0))*meshmask.umask_nan

    return u1nm,u2nm,u3nm,u4nm,u1,u2,u3,u4,v1nm,v2nm,v3nm,v4nm,v1,v2,v3,v4,fu1,fu2,fu3,fu4,fv1,fv2,fv3,fv4

def ztrdpvo_contrib(utrdpvo_full,vtrdpvo_full,u1nm,u2nm,u3nm,u4nm,u1,u2,u3,u4,v1nm,v2nm,v3nm,v4nm,v1,v2,v3,v4,fu1,fu2,fu3,fu4,fv1,fv2,fv3,fv4,meshmask,f_u_factor,f_v_factor):
    ztrdpvo2=curl(utrdpvo_full,vtrdpvo_full,meshmask)
    ztrd_mask=xr.where(np.abs(ztrdpvo2)<1e6,1,np.nan);
    z=xr.zeros_like(fv1)
    # Total model beta effect: 4-point averaged variations of the model's Coriolis parameter times masked velocities
    ztrd_betatot=-(ip12(u1*meshmask.e2v)*di(fv1)
            +ip12(u2*meshmask.e2v)*di(fv2)
            +ip12(u3*meshmask.e2v)*di(fv3)
            +ip12(u4*meshmask.e2v)*di(fv4)
            +jp12(v1*meshmask.e1u)*dj(fu1)
            +jp12(v2*meshmask.e1u)*dj(fu2)
            +jp12(v3*meshmask.e1u)*dj(fu3)
            +jp12(v4*meshmask.e1u)*dj(fu4))/meshmask.e1f/meshmask.e2f
    # Physical beta effect: 4-point averaged variations of the actual Coriolis parameter times unmasked velocities
    ztrd_betaphys=-(ip12(u1*meshmask.e2v)*di(jp1(im1(f_v_factor)))
            +ip12(u2nm*meshmask.e2v)*di(jp1(f_v_factor))
            +ip12(u3nm*meshmask.e2v)*di(im1(f_v_factor))
            +ip12(u4nm*meshmask.e2v)*di(f_v_factor)
            +jp12(v1nm*meshmask.e1u)*dj(f_u_factor)
            +jp12(v2nm*meshmask.e1u)*dj(ip1(f_u_factor))
            +jp12(v3nm*meshmask.e1u)*dj(jm1(f_u_factor))
            +jp12(v4nm*meshmask.e1u)*dj(jm1(ip1(f_u_factor))))/meshmask.e1f/meshmask.e2f
    ztrd_betanum=ztrd_betatot-ztrd_betaphys
    # Total vortex stretching: 4-point averaged divergence of horizontal velocities x horizontal scale factors times the model's Coriolis parameter
    ztrd_stretchtot=-(ip12(fv1)*di(u1*meshmask.e2v)
            +ip12(fv2)*di(u2*meshmask.e2v)
            +ip12(fv3)*di(u3*meshmask.e2v)
            +ip12(fv4)*di(u4*meshmask.e2v)
            +jp12(fu1)*dj(v1*meshmask.e1u)
            +jp12(fu2)*dj(v2*meshmask.e1u)
            +jp12(fu3)*dj(v3*meshmask.e1u)
            +jp12(fu4)*dj(v4*meshmask.e1u))/meshmask.e1f/meshmask.e2f
    # Physical vortex stretching: 4-point averaged divergence of horitontal transport computed from unmasked velocities and including also vertical scale factors, times the actual Coriolis parameter
    if len(ztrdpvo2.shape)==3:
        e3u=meshmask.e3u_0
        e3v=meshmask.e3v_0
        e3f=meshmask.e3f
    elif len(ztrdpvo2.shape)==2:
        e3u=xr.ones_like(meshmask.gphit)
        e3v=xr.ones_like(meshmask.gphit)
        e3f=xr.ones_like(meshmask.gphit)
    ztrd_stretchphys=-(ip12(fv1)*di(u1nm*meshmask.e2v*jp1(im1(e3u)))
            +ip12(fv2)*di(u2nm*meshmask.e2v*jp1(e3u))
            +ip12(fv3)*di(u3nm*meshmask.e2v*im1(e3u))
            +ip12(fv4)*di(u4nm*meshmask.e2v*e3u)
            +jp12(fu1)*dj(v1nm*meshmask.e1u*e3v)
            +jp12(fu2)*dj(v2nm*meshmask.e1u*ip1(e3v))
            +jp12(fu3)*dj(v3nm*meshmask.e1u*jm1(e3v))
            +jp12(fu4)*dj(v4nm*meshmask.e1u*jm1(ip1(e3v))))/meshmask.e1f/meshmask.e2f/e3f*ztrd_mask
    ztrd_stretchnum=ztrd_stretchtot-ztrd_stretchphys
    # 
    ztrd_crossnum=ztrdpvo2-ztrd_betatot-ztrd_stretchtot

    ztrd_pvo_names=['ztrdpvo2','ztrd_betatot','ztrd_betaphys','ztrd_betanum','ztrd_stretchtot','ztrd_stretchphys','ztrd_stretchnum','ztrd_crossnum']
    ztrd2=[]
    for name in ztrd_pvo_names:
        ztrd2.append(xr.DataArray(eval(name),name=name))
    ztrd2=np.squeeze(xr.merge(ztrd2))

    return ztrd2

def pvo_contrib_int(meshmask,utrd2,factor_u,factor_v):
    u1nm_int=zint(utrd2.u1nm,jp1(im1(meshmask.e3u_0)))
    u2nm_int=zint(utrd2.u2nm,jp1(meshmask.e3u_0))
    u3nm_int=zint(utrd2.u3nm,im1(meshmask.e3u_0))
    u4nm_int=zint(utrd2.u4nm,meshmask.e3u_0)
    u1_int=zint(utrd2.u1,meshmask.e3v_0)
    u2_int=zint(utrd2.u2,meshmask.e3v_0)
    u3_int=zint(utrd2.u3,meshmask.e3v_0)
    u4_int=zint(utrd2.u4,meshmask.e3v_0)

    v1nm_int=zint(utrd2.v1nm,meshmask.e3v_0)
    v2nm_int=zint(utrd2.v2nm,ip1(meshmask.e3v_0))
    v3nm_int=zint(utrd2.v3nm,jm1(meshmask.e3v_0))
    v4nm_int=zint(utrd2.v4nm,jm1(ip1(meshmask.e3v_0)))
    v1_int=zint(utrd2.v1,meshmask.e3u_0)
    v2_int=zint(utrd2.v2,meshmask.e3u_0)
    v3_int=zint(utrd2.v3,meshmask.e3u_0)
    v4_int=zint(utrd2.v4,meshmask.e3u_0)

    fv1_mean=zmean(utrd2.fv1,meshmask.e3v_0*meshmask.vmask)/factor_v
    fv2_mean=zmean(utrd2.fv2,meshmask.e3v_0*meshmask.vmask)/factor_v
    fv3_mean=zmean(utrd2.fv3,meshmask.e3v_0*meshmask.vmask)/factor_v
    fv4_mean=zmean(utrd2.fv4,meshmask.e3v_0*meshmask.vmask)/factor_v

    fu1_mean=zmean(utrd2.fu1,meshmask.e3u_0*meshmask.umask)/factor_u
    fu2_mean=zmean(utrd2.fu2,meshmask.e3u_0*meshmask.umask)/factor_u
    fu3_mean=zmean(utrd2.fu3,meshmask.e3u_0*meshmask.umask)/factor_u
    fu4_mean=zmean(utrd2.fu4,meshmask.e3u_0*meshmask.umask)/factor_u

    return u1nm_int,u2nm_int,u3nm_int,u4nm_int,u1_int,u2_int,u3_int,u4_int,v1nm_int,v2nm_int,v3nm_int,v4nm_int,v1_int,v2_int,v3_int,v4_int,fu1_mean,fu2_mean,fu3_mean,fu4_mean,fv1_mean,fv2_mean,fv3_mean,fv4_mean

def Momentum_balances(utrd_pvo,utrd_pg,utrd_num,utrd_adv,utrd_ldf,utrd_zdf):
    terms=xr.concat([utrd_pvo,utrd_pg,utrd_num,utrd_adv,utrd_ldf,utrd_zdf], dim="term")
    terms_sorted=terms.argsort(axis=0)
    balance_names=['Geostrophic','Ekman','Viscous','Inertial','Numerical']
    #utrd_balances=xr.where(terms_sorted.isel(term=5)+terms_sorted.isel(term=4)==1,0,np.nan)
    utrd_balances=xr.zeros_like(utrd_pvo)*np.nan
    utrd_balances.data[terms_sorted.data[-1,:,:,:]+terms_sorted.data[-2,:,:,:]==1]=0
    utrd_balances.data[(terms_sorted.data[-1,:,:,:]==5) | (terms_sorted.data[-2,:,:,:]==5)]=1
    utrd_balances.data[(terms_sorted.data[-1,:,:,:]==4) | (terms_sorted.data[-2,:,:,:]==4)]=2
    utrd_balances.data[(terms_sorted.data[-1,:,:,:]==3) | (terms_sorted.data[-2,:,:,:]==3)]=3
    utrd_balances.data[(terms_sorted.data[-1,:,:,:]==2) | (terms_sorted.data[-2,:,:,:]==2)]=4
    utrd_balances=xr.where(np.abs(utrd_pvo)<1e6,utrd_balances,np.nan);

    return utrd_balances #,balance_names

def BT_momentum_balances(utrd_pvo,utrd_pg,utrd_num,utrd_adv,utrd_ldf,utrd_tau,utrd_taub):
    terms=xr.concat([utrd_pvo,utrd_pg,utrd_num,utrd_adv,utrd_ldf,utrd_tau,utrd_taub], dim="term")
    terms_sorted=terms.argsort(axis=0)
    balance_names=['Geostrophic','Ekman','Bot. Ekman','Viscous','Inertial','Numerical']
    utrd_int_balances=xr.zeros_like(utrd_num)*np.nan
    utrd_int_balances.data[terms_sorted.data[-1,:,:]+terms_sorted.data[-2,:,:]==1]=0
    #utrd_int_balances.data[terms_sorted.isel(term=5)+terms_sorted.isel(term=4)==1]=0
    utrd_int_balances.data[(terms_sorted.data[-1,:,:]==5) | (terms_sorted.data[-2,:,:]==5)]=1
    utrd_int_balances.data[(terms_sorted.data[-1,:,:]==6) | (terms_sorted.data[-2,:,:]==6)]=2
    utrd_int_balances.data[(terms_sorted.data[-1,:,:]==4) | (terms_sorted.data[-2,:,:]==4)]=3
    utrd_int_balances.data[(terms_sorted.data[-1,:,:]==3) | (terms_sorted.data[-2,:,:]==3)]=4
    utrd_int_balances.data[(terms_sorted.data[-1,:,:]==2) | (terms_sorted.data[-2,:,:]==2)]=5
    utrd_int_balances=xr.where(utrd_pvo!=0,utrd_int_balances,np.nan);

    return utrd_int_balances #,balance_names

def Vorticity_balances(ztrd_stretch,ztrd_beta,ztrd_num,ztrd_adv,ztrd_ldf,ztrd_zdf,ztrd_topo_stretch):
    terms=xr.concat([ztrd_stretch,ztrd_beta,ztrd_num,ztrd_adv,ztrd_ldf,ztrd_zdf,ztrd_topo_stretch], dim="term")
    terms_sorted=terms.argsort(axis=0)
    balance_names=['Geostrophic','Ekman','Viscous','Inertial','Topographic','Numerical']
    ztrd_balances=xr.zeros_like(ztrd_num)*np.nan
    ztrd_balances.data[terms_sorted.data[-1,:,:,:]+terms_sorted.data[-2,:,:,:]==1]=0
    ztrd_balances.data[(terms_sorted.data[-1,:,:,:]==5) | (terms_sorted.data[-2,:,:,:]==5)]=1
    ztrd_balances.data[(terms_sorted.data[-1,:,:,:]==4) | (terms_sorted.data[-2,:,:,:]==4)]=2
    ztrd_balances.data[(terms_sorted.data[-1,:,:,:]==3) | (terms_sorted.data[-2,:,:,:]==3)]=3
    ztrd_balances.data[(terms_sorted.data[-1,:,:,:]==6) | (terms_sorted.data[-2,:,:,:]==6)]=4
    ztrd_balances.data[(terms_sorted.data[-1,:,:,:]==2) | (terms_sorted.data[-2,:,:,:]==2)]=5
    ztrd_balances=xr.where(np.abs(ztrd_stretch)<1e6,ztrd_balances,np.nan);

    return ztrd_balances #,balance_names

def BV_balances(ztrd_beta,ztrd_num,ztrd_adv,ztrd_ldf,ztrd_tau,ztrd_taub,ztrd_topo):
    terms=xr.concat([ztrd_beta,ztrd_num,ztrd_adv,ztrd_ldf,ztrd_tau,ztrd_taub,ztrd_topo], dim="term")
    terms_sorted=terms.argsort(axis=0)
    balance_names=['Sverdrup','Stommel','Viscous','Inertial','Topographic','Numerical']
    ztrd_int_balances=xr.zeros_like(ztrd_num)*np.nan
    ztrd_int_balances.data[(terms_sorted.data[-1,:,:]==0) | (terms_sorted.data[-2,:,:]==0) | (terms_sorted.data[-1,:,:]==4) | (terms_sorted.data[-2,:,:]==4)]=0
    ztrd_int_balances.data[(terms_sorted.data[-1,:,:]==5) | (terms_sorted.data[-2,:,:]==5)]=1
    ztrd_int_balances.data[(terms_sorted.data[-1,:,:]==3) | (terms_sorted.data[-2,:,:]==3)]=2
    ztrd_int_balances.data[(terms_sorted.data[-1,:,:]==2) | (terms_sorted.data[-2,:,:]==2)]=3
    ztrd_int_balances.data[(terms_sorted.data[-1,:,:]==6) | (terms_sorted.data[-2,:,:]==6)]=4
    ztrd_int_balances.data[(terms_sorted.data[-1,:,:]==1) | (terms_sorted.data[-2,:,:]==1)]=5
    ztrd_int_balances=xr.where((ztrd_beta!=0) & (np.abs(ztrd_beta)<1e6),ztrd_int_balances,np.nan);

    return ztrd_int_balances #,balance_names

def Transp_vorticity_balances(ztrd_pvo,ztrd_num,ztrd_adv,ztrd_ldf,ztrd_tau,ztrd_taub):
    terms=xr.concat([ztrd_pvo,ztrd_num,ztrd_adv,ztrd_ldf,ztrd_tau,ztrd_taub], dim="term")
    terms_sorted=terms.argsort(axis=0)
    balance_names=['Geostr+Ekman','Bot. Ekman','Viscous','Inertial','Numerical']
    curl_utrd_transp_balances=xr.zeros_like(ztrd_num)*np.nan
    curl_utrd_transp_balances.data[(terms_sorted.data[-1,:,:]==0) | (terms_sorted.data[-2,:,:]==0) | (terms_sorted.data[-1,:,:]==4) | (terms_sorted.data[-2,:,:]==4)]=0
    curl_utrd_transp_balances.data[(terms_sorted.data[-1,:,:]==5) | (terms_sorted.data[-2,:,:]==5)]=1
    curl_utrd_transp_balances.data[(terms_sorted.data[-1,:,:]==3) | (terms_sorted.data[-2,:,:]==3)]=2
    curl_utrd_transp_balances.data[(terms_sorted.data[-1,:,:]==2) | (terms_sorted.data[-2,:,:]==2)]=3
    curl_utrd_transp_balances.data[(terms_sorted.data[-1,:,:]==1) | (terms_sorted.data[-2,:,:]==1)]=4
    curl_utrd_transp_balances=xr.where((ztrd_pvo!=0) & (np.abs(ztrd_pvo)<1e6),curl_utrd_transp_balances,np.nan);

    return curl_utrd_transp_balances #,balance_names

