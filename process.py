import lammps_thermo as lmp
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d


def thermo(filename):
    n_str = filename[-9:-7]
    n = int(n_str)
    print(n)
    log_import = lmp.lammps_thermo.load(filename, 'Step', 'Loop')
    time = log_import.data[10:, 0] / 1000
    # temperature
    Temp = uniform_filter1d(log_import.data[10:, 1], 1)
    avgT = np.average(Temp)
    stdT = np.std(Temp)
    T_perc = Temp * 100 / avgT
    # potential energy
    PotEng = uniform_filter1d(log_import.data[10:, 2], 1)
    avgE = np.average(PotEng)
    stdE = np.std(PotEng)
    E_perc = PotEng * 100 / avgE
    # density
    L = uniform_filter1d(log_import.data[10:, 6], 10)
    z = uniform_filter1d(log_import.data[10:, 8], 10)
    avgL = np.average(L)
    avgZ = np.average(z)
    Vol = z * L**2
    Dens = (n**3) * 18*1.6603 / Vol
    avgD = np.average(Dens)
    stdD = np.std(Dens)
    # plotting T e Pot energy
    plt.plot(time, T_perc, label="Temperature")
    plt.plot(time, E_perc, label="Potential Energy")
    plt.ylabel("Thermodynamic properties [% avg.]", fontsize=16)
    plt.xlabel("Time [ps]", fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig('log' + n_str + '.eps', format='png')
    plt.show()
    print(avgT, "+-", stdT)
    print(avgE, "+-", stdE)
    print(avgD, r"$\pm$", stdD)
    return avgZ

