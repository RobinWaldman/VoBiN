"""
General methods
"""

import numpy as np

def norm(U,V):
    UV_norm=np.sqrt(U**2+V**2)
    return UV_norm

def log10(f):
    f_log10=np.log(np.abs(f))/np.log(10)
    return f_log10

