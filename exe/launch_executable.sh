#!$/bin/sh

# Go to executable directory:
path_exe='/home/waldmanr/Bureau/Model/VoBiN/exe'
cd ${path_exe}

# Launch diags from an input file specifying the grid and required model outputs:
ipython3 executable.py ../input/namelist_input.txt

# Opens the Jupyter notebook to visualize outputs:
#jupyter-notebook ../plot/figures.ipynb &

