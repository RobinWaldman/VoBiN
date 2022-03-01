"""
save_data.py

"""

def save_data(meshmask,utrd,utrd2,ztrd,ztrd2,ztrd_int,ztrd2_int,utrd2_int,utrd2_av,utrd2_transp,utrd_int,utrd_av,utrd_transp,curl_utrd_int,curl_utrd2_int,curl_utrd_av,curl_utrd2_av,curl_utrd_transp,curl_utrd2_transp,balances):
    """
    Storage of the meshmask Dataset containing the coordinates, grid and mask variables, as well as all momentum and vorticity balances into netcdf files
    """
    
    names=['meshmask','utrd','utrd2','ztrd','ztrd2','ztrd_int','ztrd2_int','utrd2_int','utrd2_av','utrd2_transp','utrd_int','utrd_av','utrd_transp','curl_utrd_int','curl_utrd2_int','curl_utrd_av','curl_utrd2_av','curl_utrd_transp','curl_utrd2_transp','balances']
    for name in names:
        exec(name+'.to_netcdf("../output/'+name+'.nc")')

