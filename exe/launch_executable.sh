#!$/bin/sh

# Go to executable directory:
path_exe='/home/waldmanr/Bureau/Model/VoBiN/VoBiN/exe'
cd ${path_exe}

# Launch diags from an input file specifying the grid and output files:
ipython3 executable.py /home/waldmanr/Bureau/Model/VoBiN/VoBiN/input/namelist_input.txt 

# Opens the Jupyter notebook to visualize outputs:
#jupyter-notebook /home/waldmanr/Bureau/Model/VoBiN/VoBiN/nb/Plots.ipynb &

