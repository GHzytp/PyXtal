from pyxtal import pyxtal
from pyxtal.lattice import Lattice
struc = pyxtal(molecular=True)
sites = [{"4e": [0.77, 0.57, 0.53]}]
lat = Lattice.from_para(11.43, 6.49, 11.19, 90, 83.31, 90, ltype="monoclinic")
for i in range(200):
    struc.from_random(3, 14, ["aspirin"], [4], lattice=lat, sites=sites)
    print(i, struc.lattice)                                
