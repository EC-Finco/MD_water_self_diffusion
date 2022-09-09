import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import statsmodels.api as smapi

os.chdir('xyz/')
# filenamesOO = [i for i in glob.glob('*OO.txt')]
# plt.figure()
# plt.xlabel('r (Å)', fontsize=20)
# plt.ylabel('g$_{O-O}$(r)', fontsize=20)
# for j in filenamesOO:
#     g = np.loadtxt(j)
#     radii = np.linspace(0.0, 3.1205*int(j[-8:-6])/2, 200)
#     plt.plot(radii, g, label=str(round(3.12*int(j[-8:-6]), 1)))
# plt.legend(title=r'Walls distance (Å)')
# plt.tight_layout()
# for suffix in 'png eps pdf'.split():
#     plt.savefig("plot gOO" + "." + suffix)
# plt.show()
#
# filenamesOH = [i for i in glob.glob('*OH.txt')]
#
# plt.figure()
# plt.xlabel('r (Å)', fontsize=20)
# plt.ylabel('g$_{O-H}$(r)', fontsize=20)
# for j in filenamesOH:
#     g = np.loadtxt(j)
#     radii = np.linspace(0.0, 3.1205*int(j[-8:-6])/2, 200)
#     plt.plot(radii, g, label=str(round(3.12*int(j[-8:-6]), 1)))
# plt.legend(title=r'Walls distance (Å)')
# plt.tight_layout()
# for suffix in 'png eps pdf'.split():
#     plt.savefig("plot gOH" + "." + suffix)
# plt.show()
#
# filenamesHH = [i for i in glob.glob('*HH.txt')]
#
# plt.figure()
# plt.xlabel('r (Å)', fontsize=20)
# plt.ylabel('g$_{H-H}$(r)', fontsize=20)
# for j in filenamesHH:
#     g = np.loadtxt(j)
#     radii = np.linspace(0.0, 3.1205*int(j[-8:-6])/2, 200)
#     plt.plot(radii, g, label=str(round(3.12*int(j[-8:-6]), 1)))
# plt.legend(title=r'Walls distance (Å)')
# plt.tight_layout()
# for suffix in 'png eps pdf'.split():
#     plt.savefig("plot gHH" + "." + suffix)
# plt.show()
#
# filenamesrho = [i for i in glob.glob('Rho*.txt')]
#
# plt.figure()
# plt.xlabel('z [norm.]', fontsize=20)
# plt.ylabel(r'$\rho$ [g cm$^{-3}$]', fontsize=20)
# ax = plt.gca()
# ax.set_ylim([0, 3])
# for j in filenamesrho:
#     g = np.loadtxt(j)
#     radii = np.linspace(0.0, 1, 200)
#     plt.plot(radii, g[:200], label=str(round(3.12*int(j[-6:-4]), 1)))
# plt.legend(title=r'Walls distance (Å)')
# plt.tight_layout()
# for suffix in 'png eps pdf'.split():
#     plt.savefig("plot rho" + "." + suffix)
# plt.show()
#
# filenameszmsd = [i for i in glob.glob('z-msd *.txt')]
# plt.figure()
# plt.xlabel('Time [ps]', fontsize=20)
# plt.ylabel(r'MSD$_{z}$ [Å$^{2}$]', fontsize=20)
# for j in filenameszmsd:
#     g = np.loadtxt(j, skiprows=1, usecols=1)
#     time = np.linspace(0.0, 100, len(g))
#     plt.plot(time, g, label=str(round(3.12*int(j[-6:-4]), 1)))
# plt.legend(title=r'Walls distance (Å)')
# plt.tight_layout()
# for suffix in 'png eps pdf'.split():
#     plt.savefig("plot msd" + "." + suffix)
# plt.show()

plt.figure()
plt.xlabel('1/L$_{z}$ [Å$^{-1}$]', fontsize=20)
plt.ylabel(r'D [Å$^{2}$ cm$^{-1}$]', fontsize=20)
x = np.loadtxt('diff.txt', skiprows=1, usecols=0)
y = np.loadtxt('diff.txt', skiprows=1, usecols=1)
x1 = smapi.add_constant(x)  # adding the intercept term
res_fit = smapi.OLS(y, x1).fit()
plt.plot(x, y, 'bo')
plt.plot(x1[:, 1], res_fit.fittedvalues, 'tab:orange')
plt.tight_layout()
for suffix in 'png eps pdf'.split():
    plt.savefig("plot diff" + "." + suffix)
plt.show()
print(res_fit.params, res_fit.bse, res_fit.rsquared_adj)