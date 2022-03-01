# VoBiN

url: https://github.com/RobinWaldman/VoBiN

Computes vorticity balances in NEMO

Input: NEMO's online momentum trends, u and v velocities, zonal and meridional wind stress, meshmask file containing grid, coordinate and mask variables

Output: depth-dependent and barotropic momentum balances; depth-dependent and barotropic vorticity balances, including the vorticity of the depth-integral momentum balance; the depth-integral vorticity balance; the vorticity of the depth-averaged momentum balance; the vorticity of the transport momentum balance.

Workflow defined in exe/executable.py and launched from exe/launch_executable.sh :
1. In exe/extract_data.py: extraction of input data from paths specified in input/namelist_input.txt
2. In lib/compute_vorticity.py: computation of relevant momentum and vorticity balances by making use of methods defined in lib/general_methods.py, lib/grid_methods.py and lib/pvo_methods.py
3. In exe/save_data.py: storage of relevant variables into netcdf files

Notes:
- Coriolis decomposition currently only supported for the EEN vorticity scheme
- Compatible with time-splitted momentum trends
