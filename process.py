import lammps_thermo as lmp
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d


def thermo(filename):
    n_str = filename[-9:-7]
    n = int(n_str)
    print(n)
    log_import = lmp.lammps_thermo.load(filename, 'Step', 'Loop')
    time = uniform_filter1d(log_import.data[10:, 0], 10) / 1000
    # temperature
    Temp = uniform_filter1d(log_import.data[10:, 1], 10)
    avgT = np.average(Temp)
    stdT = np.std(Temp)
    Tperc = Temp * 100 / avgT
    # potential energy
    PotEng = uniform_filter1d(log_import.data[10:, 2], 10)
    avgE = np.average(PotEng)
    stdE = np.std(PotEng)
    Eperc = PotEng * 100 / avgE
    # density
    L = uniform_filter1d(log_import.data[10:, 6], 10)
    avgL = np.average(L)
    Vol= L**3
    Dens = (n**3) * 18*1.6603 / Vol
    avgD = np.average(Dens)
    stdD = np.std(Dens)
    # plotting T e Pot energy
    plt.plot(time, Tperc, label="Temperature")
    plt.plot(time, Eperc, label="Potential Energy")
    plt.ylabel("Thermodynamic properties [% avg.]")
    plt.xlabel("Time [ps]")
    plt.legend()
    plt.savefig('log' + n_str + '.png', format='png')
    plt.show()
    print(avgT, "+-", stdT)
    print(avgE, "+-", stdE)
    print(avgD, "+-", stdD)
    return avgL
