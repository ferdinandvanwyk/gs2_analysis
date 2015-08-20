import os
import sys

in_file = sys.argv[1]

os.system('python local_heat_flux.py ' + in_file)
os.system('python radial_heat_flux.py ' + in_file)
os.system('python zf.py ' + in_file)
os.system('python phi_film.py ' + in_file)
os.system('python ntot_film.py ' + in_file)
os.system('python upar_film.py ' + in_file)
os.system('python tperp_film.py ' + in_file)
os.system('python tpar_film.py ' + in_file)
