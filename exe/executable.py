#!/usr/bin/python3.7

"""
executable.py

Executable script that launches:
    - the input data extraction
    - all operations required to get the vorticity balances

To run it, type python3 executable.py path_to_namelist.txt
in my case, path_to_namelist.txt=/home/waldmanr/Bureau/Model/VoBiN/VoBiN/input/namelist_input.txt

"""

import sys
from extract_data import extract_data
from save_data import save_data
from compute_vorticity import compute_vorticity

path_namelist=sys.argv[1]

### Momentum trend and grid variables extraction
meshmask,utrd,u,tau=extract_data(path_namelist)

### Computation of decomposed Coriolis trend, vorticity and depth integrals of momentum trends
utrd2,ztrd,ztrd2,ztrd_int,ztrd2_int,utrd2_int,utrd2_av,utrd2_transp,utrd_int,utrd_av,utrd_transp,curl_utrd_int,curl_utrd2_int,curl_utrd_av,curl_utrd2_av,curl_utrd_transp,curl_utrd2_transp,balances=compute_vorticity(meshmask,utrd,u,tau)

### Storage as netcdf files
save_data(meshmask,utrd,utrd2,ztrd,ztrd2,ztrd_int,ztrd2_int,utrd2_int,utrd2_av,utrd2_transp,utrd_int,utrd_av,utrd_transp,curl_utrd_int,curl_utrd2_int,curl_utrd_av,curl_utrd2_av,curl_utrd_transp,curl_utrd2_transp,balances)

