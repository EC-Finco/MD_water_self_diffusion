import process
import glob
import os
import numpy as np

os.chdir("logs-walls-spc/")
lognames = [i for i in glob.glob('*.lammps')]
i = 0
L = np.zeros(len(lognames))
for j in lognames:
    L[i] = process.thermo(j)
    i = i + 1
np.savetxt("sizes.txt", L)


