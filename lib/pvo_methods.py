"""
Methods used to operate on the NEMO grid
"""

import numpy as np
import xarray as xr
from grid_methods import gridt_to_f,ip1,jp1,im1,jm1,ip12,jp12,di,dj,curl,zint,zmean

def masked_f_e3f(meshmask):
    """
    Computation of the zero-masked f/e3f as in the EEN scheme

    input:
    meshmask: all useful coordinate, grid and mask variables

    output:
    zwz: zero-masked f/e3f. The zero-masking tends to reduce f near land points, hence causing a numerical Coriolis trend
    """

    ze3=meshmask.e3t_0*meshmask.tmask+jp1(meshmask.e3t_0*meshmask.tmask)+ip1(meshmask.e3t_0*meshmask.tmask)+jp1(ip1(meshmask.e3t_0*meshmask.tmask))
    zmsk=meshmask.tmask+jp1(meshmask.tmask)+ip1(meshmask.tmask)+jp1(ip1(meshmask.tmask))
    ze3f=xr.where(ze3>0,zmsk/ze3,0);
    zwz=ze3f*meshmask.f_f
    return zwz

def f_triads(zwz):
    """
    Computation of f triads (3-point averaged f/e3f) on each f point of grid cells

    input:
    zwz: zero-masked f/e3f

    output:
    ztne,ztnw,ztse,ztsw: 3-point averaged f/e3f over the NE, NW, SE and SW corners of each grid cell. The zero-masking tends to reduce f near land points, hence causing a numerical Coriolis trend
    """
    ztne=im1(zwz)+zwz+jm1(zwz)
    ztnw=im1(jm1(zwz))+im1(zwz)+zwz
    ztse=zwz+jm1(zwz)+im1(jm1(zwz))
    ztsw=jm1(zwz)+im1(jm1(zwz))+im1(zwz)
    return ztne,ztnw,ztse,ztsw

def pvo_offline(meshmask,u,ztne,ztnw,ztse,ztsw):
    """
    Offline computation the planetary vorticity (Coriolis) trend as done in NEMO's EEN scheme

    input:
    meshmask: all useful coordinate, grid and mask variables
    u: u and v velocities
    ztne,ztnw,ztse,ztsw: 3-point averaged f/e3f over the NE, NW, SE and SW corners of each grid cell.

    output:
    utrdpvo_full,vtrdpvo_full: full u and v planetary vorticity (Coriolis) trend reconstructed offline from u, v and the so-called f triads (3-point averaged f/e3f)
    """

    zwx=u.uo*meshmask.e2u*meshmask.e3u_0
    zwy=u.vo*meshmask.e1v*meshmask.e3v_0
    utrdpvo_full=1/12*(ztne*zwy+ip1(ztnw)*ip1(zwy)+ztse*jm1(zwy)+ip1(ztsw)*jm1(ip1(zwy)))/meshmask.e1u*meshmask.umask_nan
    vtrdpvo_full=-1/12*(jp1(ztsw)*jp1(im1(zwx))+jp1(ztse)*jp1(zwx)+ztnw*im1(zwx)+ztne*zwx)/meshmask.e2v*meshmask.vmask_nan
    return utrdpvo_full,vtrdpvo_full

def pvo_contrib(meshmask,u,ztne,ztnw,ztse,ztsw):
    """
    Offline decomposition of u-v and f contributions to the Coriolis trend from the EEN scheme

    input:
    meshmask: all useful coordinate, grid and mask variables
    u: u and v velocities
    ztne,ztnw,ztse,ztsw: 3-point averaged f/e3f over the NE, NW, SE and SW corners of each grid cell.

    output:
    ui: u velocities on the v-grid as in the EEN scheme
    uinm: u velocities on the v-grid as in the EEN scheme but without applying v-masking
    vi: v velocities on the u-grid as in the EEN scheme
    vinm: v velocities on the u-grid as in the EEN scheme but without applying u-masking
    fui and fvi: Coriolis parameter on the u-grid and v-grid as in the EEN scheme
    """

    # 4-point averaging from u-grid to v-grid, ponderated by e2u scale factors. To retrieve the v-point u from the EEN scheme, divide u*e2u by e2v. Unmasked velocities are required to retrieve full u transports
    u1nm=1/4*jp1(im1(u.uo))*jp1(im1(meshmask.e2u))/meshmask.e2v; u1=u1nm*meshmask.vmask_nan
    u2nm=1/4*jp1(u.uo)*jp1(meshmask.e2u)/meshmask.e2v; u2=u2nm*meshmask.vmask_nan
    u3nm=1/4*im1(u.uo)*im1(meshmask.e2u)/meshmask.e2v; u3=u3nm*meshmask.vmask_nan
    u4nm=1/4*u.uo*meshmask.e2u/meshmask.e2v; u4=u4nm*meshmask.vmask_nan

    # 4-point averaging from u-grid to v-grid. To retrieve f from the EEN scheme, multiply f/e3f triads by the e3u scale factor
    fv1=1/3*jp1(ztsw)*jp1(im1(meshmask.e3u_0))*meshmask.vmask_nan
    fv2=1/3*jp1(ztse)*jp1(meshmask.e3u_0)*meshmask.vmask_nan
    fv3=1/3*ztnw*im1(meshmask.e3u_0)*meshmask.vmask_nan
    fv4=1/3*ztne*meshmask.e3u_0*meshmask.vmask_nan

    # 4-point averaging from v-grid to u-grid, ponderated by e1v scale factors. To retrieve the u-point v from the EEN scheme, divide v*e1v by e1u. Unmasked velocities are required to retrieve full v transports
    v1nm=1/4*u.vo*meshmask.e1v/meshmask.e1u; v1=v1nm*meshmask.umask_nan
    v2nm=1/4*ip1(u.vo)*ip1(meshmask.e1v)/meshmask.e1u; v2=v2nm*meshmask.umask_nan
    v3nm=1/4*jm1(u.vo)*jm1(meshmask.e1v)/meshmask.e1u; v3=v3nm*meshmask.umask_nan
    v4nm=1/4*jm1(ip1(u.vo))*jm1(ip1(meshmask.e1v))/meshmask.e1u; v4=v4nm*meshmask.umask_nan

    # 4-point averaging from v-grid to u-grid. To retrieve f from the EEN scheme, multiply f/e3f triads by the e3v scale factor
    fu1=1/3*ztne*meshmask.e3v_0*meshmask.umask_nan
    fu2=1/3*ip1(ztnw)*ip1(meshmask.e3v_0)*meshmask.umask_nan
    fu3=1/3*ztse*jm1(meshmask.e3v_0)*meshmask.umask_nan
    fu4=1/3*ip1(ztsw)*jm1(ip1(meshmask.e3v_0))*meshmask.umask_nan

    return u1nm,u2nm,u3nm,u4nm,u1,u2,u3,u4,v1nm,v2nm,v3nm,v4nm,v1,v2,v3,v4,fu1,fu2,fu3,fu4,fv1,fv2,fv3,fv4

def ztrdpvo_contrib(utrdpvo_full,vtrdpvo_full,u1nm,u2nm,u3nm,u4nm,u1,u2,u3,u4,v1nm,v2nm,v3nm,v4nm,v1,v2,v3,v4,fu1,fu2,fu3,fu4,fv1,fv2,fv3,fv4,meshmask,f_u_factor,f_v_factor):
    """
    Offline decomposition of the vorticity of the Coriolis trend under the EEN scheme

    input:
    utrdpvo_full,vtrdpvo_full: full u and v planetary vorticity (Coriolis) trend reconstructed offline
    ui: u velocities on the v-grid as in the EEN scheme
    uinm: u velocities on the v-grid as in the EEN scheme but without applying v-masking
    vi: v velocities on the u-grid as in the EEN scheme
    vinm: v velocities on the u-grid as in the EEN scheme but without applying u-masking
    fui and fvi: Coriolis parameter on the u-grid and v-grid as in the EEN scheme
    meshmask: all useful coordinate, grid and mask variables
    f_u_factor and f_v_factor: either f, f/h or 1

    output: ztrd2 containing:
    ztrdpvo2: offline computation of the Coriolis vorticity as in the EEN scheme
    ztrd_betatot: offline computation of the full "beta effect", ie: the vorticity due to variations in f (or f/h) as defined in NEMO's EEN scheme
    ztrd_betaphys: offline computation of the physical "beta effect", ie: the vorticity due to variations in f (or f/h) deduced from the actual f and unmasked interpolated velocities
    ztrd_betanum: deduced as a residual from ztrd_betatot-ztrd_betaphys. Corresponds to the numerical torque due to the model approximations in f and to the masking of velocities in the interpolation procedure
    ztrd_stretchtot: offline computation of the full "planetary vortex stretching", ie: the vorticity due to the horizontal divergence as defined in NEMO's EEN scheme
    ztrd_stretchphys: offline computation of the physical "planetary vortex stretching", ie: the vorticity deduced from the actual model horizontal divergence with the correct e3u-e3v scale factors and the unmasked interpolated velocities
    ztrd_stretchnum: deduced as a residual from ztrd_stretchtot-ztrd_stretchphys. Corresponds to the numerical torque due to the model approximations in e3 scale factors and to the masking of velocities in the interpolation procedure
    ztrd_crossnum: deduced as a residual from ztrdvo2-ztrd_betatot-ztrd_stretchtot. It is non-zero only in the caseof depth-integral balances if there is a vertical covariance between beta and stretching effects.
    """

    # Offline computation of the full Coriolis trend vorticity
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
    # Physical vortex stretching: 4-point averaged divergence of horizontal transport computed from unmasked velocities and including the correct vertical scale factors, times the actual Coriolis parameter
    if len(ztrdpvo2.shape)==3: # in the depth-dependent case, we must multiply by e3u-e3v scale factors to get the horitontal divergence, and then divide by e3f to get an unintegrated vorticity trend
        e3u=meshmask.e3u_0
        e3v=meshmask.e3v_0
        e3f=meshmask.e3f
    elif len(ztrdpvo2.shape)==2: # in the depth-integral case, vertical integration has already been performed in the pvo_contrib_int decomposition
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
    # Crossed numerical term, only non-zero for depth-integral balances
    ztrd_crossnum=ztrdpvo2-ztrd_betatot-ztrd_stretchtot

    ztrd_pvo_names=['ztrdpvo2','ztrd_betatot','ztrd_betaphys','ztrd_betanum','ztrd_stretchtot','ztrd_stretchphys','ztrd_stretchnum','ztrd_crossnum']
    ztrd2=[]
    for name in ztrd_pvo_names:
        ztrd2.append(xr.DataArray(eval(name),name=name))
    ztrd2=np.squeeze(xr.merge(ztrd2))

    return ztrd2

def pvo_contrib_int(meshmask,utrd2,factor_u,factor_v):
    """
    Offline decomposition of depth integral u-v and f contributions to the Coriolis trend from the EEN scheme

    input:
    meshmask: all useful coordinate, grid and mask variables
    utrd2: offline decomposition of NEMO's Coriolis trend under the EEN scheme
    factor_u and factor_v: factor associated to f for the computation of depth-dependent and depth-averaged balances (factor 1), depth-averaged balances (factor 1/h) and transport vorticity balances (factor 1/f)

    output:
    ui: depth-integral u velocities on the v-grid as in the EEN scheme
    uinm: depth-integral u velocities on the v-grid as in the EEN scheme but without applying v-masking
    vi: depth-integral v velocities on the u-grid as in the EEN scheme
    vinm: depth-integral v velocities on the u-grid as in the EEN scheme but without applying u-masking
    fui_mean and fvi_mean: depth average Coriolis parameter on the u-grid and v-grid as in the EEN scheme, times factor_u or factor_v
    """

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
    """
    Diagnostic of the 5 main depth-dependent momentum balances as in Waldman et al JAMES in prep.

    input: norm of momentum trends:
    utrd_pvo: planetary vorticity momentum trend = Coriolis force
    utrd_pg: pressure gradient momentum trend = pressure force. Under time splitting, it is fully contained in the utrd_hpg term
    utrd_num: numerical force arising from f and velocity approximations in NEMO's Coriolis force under the EEN scheme, namely: modifications of f by EEN's triads near land points and masking of u-v velocities by application of the v-u masks upon interpolation
    utrd_adv: advective momentum trend, both horizontal (keg+rvo) and vertical (zad) = inertial force = nonlinear advection
    utrd_ldf: lateral momentum diffusion trend = lateral dissipation
    utrd_zdf: vertical momentum diffusion trend = vertical friction, both barotropic and baroclinic. Under time splitting, its barotropic component is contained in the time splitted trend (utrd_spg) and its baroclinic one in the vertical mixing trend (utrd_zdf)

    output: utrd_balances: 5 main depth-dependent momentum balances labeled from 0 to 4 for:
    0: geostrophic balance = the two largest terms of the momentum equation are Coriolis and pressure force
    1: Ekman balance = vertical friction is among the two largest terms of the momentum equation, and the other is either Coriolis or the pressure force
    2: Viscous balance = lateral dissipation is among the two largest terms of the momentum equation, and the other is neither inertial nor numerical
    3: Inertial balance = the inertial force is among the two largest terms of the momentum equation, and the other is not numerical
    4: Numerical balance = the numerical Coriolis force is among the two largest terms of the momentum equation
    """

    terms=xr.concat([utrd_pvo,utrd_pg,utrd_num,utrd_adv,utrd_ldf,utrd_zdf], dim="term")
    terms_sorted=terms.argsort(axis=0)
    utrd_balances=xr.zeros_like(utrd_pvo)*np.nan
    utrd_balances.data[terms_sorted.data[-1,:,:,:]+terms_sorted.data[-2,:,:,:]==1]=0
    utrd_balances.data[(terms_sorted.data[-1,:,:,:]==5) | (terms_sorted.data[-2,:,:,:]==5)]=1
    utrd_balances.data[(terms_sorted.data[-1,:,:,:]==4) | (terms_sorted.data[-2,:,:,:]==4)]=2
    utrd_balances.data[(terms_sorted.data[-1,:,:,:]==3) | (terms_sorted.data[-2,:,:,:]==3)]=3
    utrd_balances.data[(terms_sorted.data[-1,:,:,:]==2) | (terms_sorted.data[-2,:,:,:]==2)]=4
    utrd_balances=xr.where(np.abs(utrd_pvo)<1e6,utrd_balances,np.nan);

    return utrd_balances

def BT_momentum_balances(utrd_pvo,utrd_pg,utrd_num,utrd_adv,utrd_ldf,utrd_tau,utrd_taub):
    """
    Diagnostic of the 6 main depth-integral momentum balances as in Waldman et al JAMES in prep.

    input: norm of depth-integral momentum trends:
    utrd_pvo: planetary vorticity momentum trend = Coriolis force
    utrd_pg: pressure gradient momentum trend = pressure force. Under time splitting, it is fully contained in the utrd_hpg term
    utrd_num: numerical force arising from f and velocity approximations in NEMO's Coriolis force under the EEN scheme, namely: modifications of f by EEN's triads near land points and masking of u-v velocities by application of the v-u mask s upon interpolation
    utrd_adv: advective momentum trend, both horizontal (keg+rvo) and vertical (zad) = inertial force = nonlinear advection
    utrd_ldf: lateral momentum diffusion trend = lateral dissipation
    utrd_tau: surface wind stress momentum trend, deduced from |tau|/rho0
    utrd_taub: bottom stress momentum trend, deduced from the depth integral of the time splitted trend utrd_ts minus |tau|/rho0

    output: utrd_int_balances: 6 main depth-integral momentum balances labeled from 0 to 5 for:
    0: geostrophic balance = the two largest terms of the momentum equation are Coriolis and pressure force
    1: (surface) Ekman balance = surface wind stress is among the two largest terms of the momentum equation, and the other is either Coriolis or the pressure force
    2: Bottom Ekman balance = bottom stress is among the two largest terms of the momentum equation, and the other is either Coriolis, the pressure force or the wind stress
    3: Viscous balance = lateral dissipation is among the two largest terms of the momentum equation, and the other is neither inertial nor numerical
    4: Inertial balance = the inertial force is among the two largest terms of the momentum equation, and the other is not numerical
    5: Numerical balance = the numerical Coriolis force is among the two largest terms of the momentum equation
    """

    terms=xr.concat([utrd_pvo,utrd_pg,utrd_num,utrd_adv,utrd_ldf,utrd_tau,utrd_taub], dim="term")
    terms_sorted=terms.argsort(axis=0)
    utrd_int_balances=xr.zeros_like(utrd_num)*np.nan
    utrd_int_balances.data[terms_sorted.data[-1,:,:]+terms_sorted.data[-2,:,:]==1]=0
    utrd_int_balances.data[(terms_sorted.data[-1,:,:]==5) | (terms_sorted.data[-2,:,:]==5)]=1
    utrd_int_balances.data[(terms_sorted.data[-1,:,:]==6) | (terms_sorted.data[-2,:,:]==6)]=2
    utrd_int_balances.data[(terms_sorted.data[-1,:,:]==4) | (terms_sorted.data[-2,:,:]==4)]=3
    utrd_int_balances.data[(terms_sorted.data[-1,:,:]==3) | (terms_sorted.data[-2,:,:]==3)]=4
    utrd_int_balances.data[(terms_sorted.data[-1,:,:]==2) | (terms_sorted.data[-2,:,:]==2)]=5
    utrd_int_balances=xr.where(np.abs(utrd_pvo)<1e6,utrd_int_balances,np.nan);

    return utrd_int_balances

def Vorticity_balances(ztrd_stretch,ztrd_beta,ztrd_num,ztrd_adv,ztrd_ldf,ztrd_zdf,ztrd_topo_stretch):
    """
    Diagnostic of the 6 main depth-dependent vorticity balances as in Waldman et al JAMES in prep.

    input: absolute value of depth-dependent vorticity trends:
    ztrd_stretch: interior vorticity trend due to the physical planetary vortex stretching computed offline from the Coriolis parameter times the horizontal divergence (including e3u-e3v scale factor variations). Note that we have removed the depth-average vortex stretching related to bottom stretching in order to separate a purely redistributive interior stretching from a topographic bottom stretching
    ztrd_beta: vorticity trend due to the physical beta effect computed offline from variations of the Coriolis parameter times the unmasked horizontal velocities
    ztrd_num: numerical Coriolis torque arising from f and velocity approximations in NEMO's Coriolis force under the EEN scheme, namely: modifications of f by EEN's triads near land points and neglect of e3u-e3v scale factors in the computation of horizontal divergence
    ztrd_adv: inertial torque
    ztrd_ldf: viscous torque
    ztrd_zdf: vertical frictional torque
    ztrd_topo_stretch: bottom topographic stretching given at every depth as the depth average of the physical vortex stretching, namely the fraction of stretching that does not compensate over the vertical

    output: ztrd_balances: 6 main depth-dependent vorticity balances labeled from 0 to 5 for:
    0: geostrophic vorticity balance = the two largest terms of the vorticity equation are the interior vortex stretching and beta effect, both arising from the Coriolis force
    1: Ekman vorticity balance = the frictional torque is among the two largest terms of the vorticity equation, and the other is either the physical vortex stretching or beta effect
    2: Viscous vorticity balance = the viscous torque is among the two largest terms of the vorticity equation, and the other is either interior stretching/beta effect or frictional
    3: Inertial balance = the inertial torque is among the two largest terms of the vorticity equation, and the other is neither topograhic nor numerical
    4: Topographic balance = the bottom vortex stretching is among the two largest terms of the vorticity equation, and the other is not numerical
    5: Numerical balance = the numerical Coriolis torques arising near the topography are among the two largest terms of the vorticity equation
    """

    terms=xr.concat([ztrd_stretch,ztrd_beta,ztrd_num,ztrd_adv,ztrd_ldf,ztrd_zdf,ztrd_topo_stretch], dim="term")
    terms_sorted=terms.argsort(axis=0)
    ztrd_balances=xr.zeros_like(ztrd_num)*np.nan
    ztrd_balances.data[terms_sorted.data[-1,:,:,:]+terms_sorted.data[-2,:,:,:]==1]=0
    ztrd_balances.data[(terms_sorted.data[-1,:,:,:]==5) | (terms_sorted.data[-2,:,:,:]==5)]=1
    ztrd_balances.data[(terms_sorted.data[-1,:,:,:]==4) | (terms_sorted.data[-2,:,:,:]==4)]=2
    ztrd_balances.data[(terms_sorted.data[-1,:,:,:]==3) | (terms_sorted.data[-2,:,:,:]==3)]=3
    ztrd_balances.data[(terms_sorted.data[-1,:,:,:]==6) | (terms_sorted.data[-2,:,:,:]==6)]=4
    ztrd_balances.data[(terms_sorted.data[-1,:,:,:]==2) | (terms_sorted.data[-2,:,:,:]==2)]=5
    ztrd_balances=xr.where(np.abs(ztrd_stretch)<1e6,ztrd_balances,np.nan);

    return ztrd_balances

def BV_balances(ztrd_beta,ztrd_num,ztrd_adv,ztrd_ldf,ztrd_tau,ztrd_taub,ztrd_topo):
    """
    Diagnostic of the 6 main barotropic vorticity balances as in Waldman et al JAMES in prep. This method applies for the depth-integral vorticity balance, the vorticity of the depth-integral momentum balance and vorticity of the depth-average momentum balance

    input: absolute value of depth-integral vorticity trends:
    ztrd_beta: vorticity trend due to the physical beta effect computed offline from variations of the Coriolis parameter times the unmasked horizontal velocities (and divided by either 1 or h)
    ztrd_num: numerical Coriolis torque arising from f and velocity approximations in NEMO's Coriolis force under the EEN scheme, namely: modifications of f by EEN's triads near land points; neglect of e3u-e3v scale factors and u-v masking of v-u velocities upon interpolation in the computation of horizontal divergence
    ztrd_adv: inertial torque
    ztrd_ldf: viscous torque
    ztrd_tau: wind stress torque
    ztrd_taub: bottom stress torque, deduced from the depth integral of the time splitted trend ztrd_ts minus |tau|/rho0 (or |tau|/(rho0*h))
    ztrd_topo: bottom topographic torque arising either from the physical bottom vortex stretching, the bottom pressure torque or the baroclinic pressure torque (JEBAR)

    output: ztrd_int_balances: 6 main barotropic vorticity balances labeled from 0 to 5 for:
    0: Sverdrup = the two largest terms of the vorticity equation are the wind stress curl and beta effect
    1: Stommel = bottom stress torque is among the two largest terms of the vorticity equation, and the other is either the wind stress curl or beta effect
    2: Viscous vorticity balance = the viscous torque is among the two largest terms of the vorticity equation, and the other is either surface/bottom stress or beta effect
    3: Inertial balance = the inertial torque is among the two largest terms of the vorticity equation, and the other is neither topograhic nor numerical
    4: Topographic balance = the physical topograhic torque is among the two largest terms of the vorticity equation, and the other is not numerical
    5: Numerical balance = the numerical Coriolis torque arising near the topography is among the two largest terms of the vorticity equation
    """

    terms=xr.concat([ztrd_beta,ztrd_num,ztrd_adv,ztrd_ldf,ztrd_tau,ztrd_taub,ztrd_topo], dim="term")
    terms_sorted=terms.argsort(axis=0)
    ztrd_int_balances=xr.zeros_like(ztrd_num)*np.nan
    ztrd_int_balances.data[(terms_sorted.data[-1,:,:]==0) | (terms_sorted.data[-2,:,:]==0) | (terms_sorted.data[-1,:,:]==4) | (terms_sorted.data[-2,:,:]==4)]=0
    ztrd_int_balances.data[(terms_sorted.data[-1,:,:]==5) | (terms_sorted.data[-2,:,:]==5)]=1
    ztrd_int_balances.data[(terms_sorted.data[-1,:,:]==3) | (terms_sorted.data[-2,:,:]==3)]=2
    ztrd_int_balances.data[(terms_sorted.data[-1,:,:]==2) | (terms_sorted.data[-2,:,:]==2)]=3
    ztrd_int_balances.data[(terms_sorted.data[-1,:,:]==6) | (terms_sorted.data[-2,:,:]==6)]=4
    ztrd_int_balances.data[(terms_sorted.data[-1,:,:]==1) | (terms_sorted.data[-2,:,:]==1)]=5
    ztrd_int_balances=xr.where((ztrd_beta!=0) & (np.abs(ztrd_beta)<1e6),ztrd_int_balances,np.nan);

    return ztrd_int_balances

def Transp_vorticity_balances(ztrd_pvo,ztrd_num,ztrd_adv,ztrd_ldf,ztrd_tau,ztrd_taub):
    """
    Diagnostic of the 5 main transport vorticity balances as in Waldman et al JAMES in prep. This method applies for the vorticity of the "transport" momentum balance (depth-integral balance divided by f)

    input: absolute value of "transport" vorticity trends:
    ztrd_pvo: vorticity trend due to the pressure-driven (geostrophic) divergence
    ztrd_num: numerical Coriolis torque arising from f and velocity approximations in NEMO's Coriolis force under the EEN scheme, namely: modifications of f by EEN's triads near land points; neglect of e3u-e3v scale factors and u-v masking of v-u velocities upon interpolation in the computation of horizontal divergence. Without this numerical torque, the Coriolis torque is near zero because the depth-integral flow is approximately nondivergent.
    ztrd_adv: inertial torque-driven divergence
    ztrd_ldf: viscous torque-driven divergence
    ztrd_tau: surface Ekman pumping
    ztrd_taub: bottom Ekman pumping, deduced from the depth integral of the time splitted trend ztrd_ts minus |tau|/(rho0*f)

    output: ztrd_int_balances: 5 main transport vorticity balances labeled from 0 to 4 for:
    0: Geostrophic-Ekman = the two largest terms of the vorticity equation are the geostrophic and Ekman divergences
    1: Bottom Ekman = the bottom Ekman pumping is among the two largest terms of the vorticity equation, and the other is either the geostrophic or Ekman divergence
    2: Viscous balance = the viscous divergence is among the two largest terms of the vorticity equation, and the other is either surface/bottom Ekman pumping or geostrophic divergence
    3: Inertial balance = the inertial divergence is among the two largest terms of the vorticity equation, and the other is not numerical
    4: Numerical balance = the numerical Coriolis divergence arising near the topography is among the two largest terms of the vorticity equation
    """

    terms=xr.concat([ztrd_pvo,ztrd_num,ztrd_adv,ztrd_ldf,ztrd_tau,ztrd_taub], dim="term")
    terms_sorted=terms.argsort(axis=0)
    curl_utrd_transp_balances=xr.zeros_like(ztrd_num)*np.nan
    curl_utrd_transp_balances.data[(terms_sorted.data[-1,:,:]==0) | (terms_sorted.data[-2,:,:]==0) | (terms_sorted.data[-1,:,:]==4) | (terms_sorted.data[-2,:,:]==4)]=0
    curl_utrd_transp_balances.data[(terms_sorted.data[-1,:,:]==5) | (terms_sorted.data[-2,:,:]==5)]=1
    curl_utrd_transp_balances.data[(terms_sorted.data[-1,:,:]==3) | (terms_sorted.data[-2,:,:]==3)]=2
    curl_utrd_transp_balances.data[(terms_sorted.data[-1,:,:]==2) | (terms_sorted.data[-2,:,:]==2)]=3
    curl_utrd_transp_balances.data[(terms_sorted.data[-1,:,:]==1) | (terms_sorted.data[-2,:,:]==1)]=4
    curl_utrd_transp_balances=xr.where((ztrd_pvo!=0) & (np.abs(ztrd_pvo)<1e6),curl_utrd_transp_balances,np.nan);

    return curl_utrd_transp_balances

