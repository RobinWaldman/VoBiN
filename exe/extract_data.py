"""
extract_data.py

Extracts grid and physical model data required for the computation of vorticity balance:
    - grid size and coordinates
    - masks
    - momentum trends, u, v, Coriolis parameter (to decompose Coriolis force), tauu, tauv (to decompose depth-integral stress)
"""

import numpy as np
import xarray as xr
import sys
import os
sys.path.append(os.path.abspath("../lib"))
from grid_methods import gridt_to_f

def extract_data(path_namelist):
    """
    Grid and physical data extraction
    path_namelist: path to the txt file containing the paths of all input files
    """

    # Save paths to data into dictionary
    with open(path_namelist) as f:
        dict_namelist_input = {name : path for line in f for (name, path) in [line.strip().split()]}

    # Extract grid variables
    meshmask=np.squeeze(xr.open_dataset(dict_namelist_input['meshmask']))
    meshmask['e3f']=gridt_to_f(meshmask.e3t_0,meshmask.tmask) # for integrating vorticity balance in the f-grid
    meshmask['umask_nan']=xr.where(meshmask.umask!=0,meshmask.umask,np.nan) # for masking locations where u=0
    meshmask['vmask_nan']=xr.where(meshmask.vmask!=0,meshmask.vmask,np.nan) # for masking locations where v=0
    meshmask['f_u']=2*2*np.pi/86400*np.sin(np.radians(meshmask.gphiu)) # for decomposing Coriolis vorticity
    meshmask['f_u_nan']=xr.where(meshmask.f_u!=0,meshmask.f_u,np.nan) # to divide by f in the transport vorticity equation
    meshmask['f_v']=2*2*np.pi/86400*np.sin(np.radians(meshmask.gphiv)) # for decomposing Coriolis vorticity
    meshmask['f_f']=2*2*np.pi/86400*np.sin(np.radians(meshmask.gphif)) # for computing Coriolis with EEN
    meshmask['bathy_u']=(meshmask.umask*meshmask.e3u_0).sum(dim='z') # for decomposing Coriolis depth-average vorticity
    meshmask['bathy_v']=(meshmask.vmask*meshmask.e3v_0).sum(dim='z') # for decomposing Coriolis depth-average vorticity
    meshmask['beta']=2*2*np.pi/86400/(6700*1e3)*np.cos(np.radians(meshmask.gphit)) # for expressing vorticity terms as beta transports
    meshmask=meshmask.rename({'z': 'lev'})

    # Extract momentum trends
    data = []
    for name in list(dict.keys(dict_namelist_input))[6:]:
        ds = xr.open_dataset(dict_namelist_input['utrd_dir']+dict_namelist_input[name])
        data.append(ds)
    utrd = np.squeeze(xr.merge(data,compat='override'))

    # extract horizontal velocities and zonal wind stress
    u=np.squeeze(xr.merge([xr.open_dataset(dict_namelist_input['uo']), xr.open_dataset(dict_namelist_input['vo'])],compat='override'))
    u['uo']=u['uo'].fillna(0); u['vo']=u['vo'].fillna(0)

    tau=np.squeeze(xr.merge([xr.open_dataset(dict_namelist_input['tauuo']), xr.open_dataset(dict_namelist_input['tauvo'])],compat='override'))

    return meshmask,utrd,u,tau
