"""
save_data.py

"""

def save_data(meshmask,utrd,utrd2,ztrd,ztrd2,ztrd_int,ztrd2_int,utrd2_int,utrd2_av,utrd2_transp,utrd_int,utrd_av,utrd_transp,curl_utrd_int,curl_utrd2_int,curl_utrd_av,curl_utrd2_av,curl_utrd_transp,curl_utrd2_transp,balances):
    """
    Grid and physical data extraction
    path_namelist: path to the txt file containing the paths of all input files
    """
    
    names=['meshmask','utrd','utrd2','ztrd','ztrd2','ztrd_int','ztrd2_int','utrd2_int','utrd2_av','utrd2_transp','utrd_int','utrd_av','utrd_transp','curl_utrd_int','curl_utrd2_int','curl_utrd_av','curl_utrd2_av','curl_utrd_transp','curl_utrd2_transp','balances']
    for name in names:
        exec(name+'.to_netcdf("../output/'+name+'.nc")')

