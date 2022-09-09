import pandas as pd
from numpy import sqrt, pi
import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import pickle


def volume(r):
    """ volume of a sphere of radius r located at height z """
    vol = 4.0 / 3.0 * pi * r ** 3
    return vol


def distance(a, b, x_lat, y_lat, z_lat):
    """ get displacement in each coordinate and wrap w.r.t. lattice parameter """
    dx = abs(a[0] - b[0])
    x = min(dx, abs(x_lat - dx))

    dy = abs(a[1] - b[1])
    y = min(dy, abs(y_lat - dy))

    dz = abs(a[2] - b[2])
    z = min(dz, abs(z_lat - dz))

    return sqrt(x ** 2 + y ** 2 + z ** 2)


class Trajectory:
    def __init__(self, filename, skip, resolution=200):
        """
        filename         : path to the trajectory file
        skip             : number of snapshots to be skipped between two configurations that are evaluated
                           (for example, if trajectory is 9000 steps long, and skip = 10, every tenth step
                           is evaluated, 900 steps in total; use skip = 1 to take every step of the MD)
        z_bot_interface  : average vertical coordinate for interface below water layer in Angstrom
        z_top_interface  : average vertical coordinate for interface above water layer in Angstrom
        interface_offset : distance between interface and region of water with bulk-like 'properties
        resolution       : number of points in the final radial distribution function """

        self.volume_per_h2o = None
        self.z_bins = None
        self.data_oxygen = None
        self.g_of_rOH = None
        self.radii = None
        self.g_of_rOO = None
        self.g_of_rHH = None
        self.rho_of_O = None
        with open(filename, 'r') as f:
            data = f.readlines()

        self.n_atoms = int(data[0].split()[0])
        self.n_steps_total = int(len(data) / (self.n_atoms + 2))
        self.atom_list = [line.split()[0] for line in data[2: self.n_atoms + 2]]
        self.skip = skip
        self.n_steps = self.n_steps_total // self.skip
        self.coordinates = np.zeros((self.n_steps, self.n_atoms, 3))
        for step in range(self.n_steps):
            coords = np.zeros((self.n_atoms, 3))
            i = step * self.skip * (self.n_atoms + 2)
            for j, line in enumerate(data[i + 2: i + self.n_atoms + 2]):
                coords[j, :] = [float(value) for value in line.split()[1:]]
            self.coordinates[step] = coords
        self.x_max = np.max(self.coordinates[:, :, 0])
        self.y_max = np.max(self.coordinates[:, :, 1])
        self.z_max = np.max(self.coordinates[:, :, 2])

        self.resolution = resolution
        self.data_oxygen = np.zeros((int(self.n_atoms/3), int(3), self.n_steps))
        self.data_hydrogen = np.zeros((int(2*self.n_atoms/3), int(3), self.n_steps))
        for step in range(self.n_steps):
            # print('{:4d} : {:4d}'.format(step, self.n_steps))
            """ isolate all liquid water molecules based on the position of the oxygen atoms """
            oxygen = np.zeros((int(self.n_atoms/3), 3))
            for i, atom in enumerate(self.coordinates[step]):
                if self.atom_list[i] == "1":
                    oxygen[int(i/3), :] = atom
            self.data_oxygen[:, :, step] = oxygen

            hydrogen = np.zeros((int(2*self.n_atoms/3), 3))
            for i, atom in enumerate(self.coordinates[step]):
                if self.atom_list[i] == "2":
                    hydrogen[int(i - np.floor(i/3))-1, :] = atom
            self.data_hydrogen[:, :, step] = hydrogen

        self.compute_volume_per_h2o()

    def compute_volume_per_h2o(self):
        """ calculates the average volume per water molecule from the subset of configurations """
        """ sum up all oxygen atoms belonging to water in the bulk region """
        n_h2o = self.n_atoms / 3
        """ divide the volume of the bulk region by the average number of molecules per step """
        self.volume_per_h2o = (self.x_max * self.y_max * self.z_max) / n_h2o

    def compute_radial_distribution(self):
        """ no reason to go above half of the smallest lattice parameter as mirror images start
        to be double-counted """
        r_cutoff = self.x_max / 2.0
        dr = r_cutoff / self.resolution
        volumes = np.zeros(self.resolution)

        self.radii = np.linspace(0.0, self.resolution * dr, self.resolution)
        self.g_of_rOO = np.zeros(self.resolution)
        self.g_of_rOH = np.zeros(self.resolution)
        self.g_of_rHH = np.zeros(self.resolution)
        for step in range(self.n_steps):
            """ loop over each pair of H2O molecules, calculate their distance, build a histogram 
            each pair is evaluated as two contributions to the distribution function """
            # compute g(r) for O-O and O-H
            for i, oxygen1 in enumerate(self.data_oxygen[:, :, step]):
                for j in range(self.resolution):
                    r1 = j * dr
                    r2 = r1 + dr
                    v1 = volume(r1)
                    v2 = volume(r2)
                    volumes[j] += v2 - v1

                for oxygen2 in self.data_oxygen[i + 1:, :, step]:
                    dist = distance(oxygen1, oxygen2, self.x_max, self.y_max, self.z_max)
                    index = int(dist / dr)
                    if 0 < index < self.resolution:
                        self.g_of_rOO[index] += 2.0

                hydrogen = np.delete(self.data_hydrogen, 2*i+1, axis=0)
                hydrogen = np.delete(hydrogen, 2*i, axis=0)
                for hydrogen_at in hydrogen[:, :, step]:
                    dist = distance(oxygen1, hydrogen_at, self.x_max, self.y_max, self.z_max)
                    index = int(dist / dr)
                    if 0 < index < self.resolution:
                        self.g_of_rOH[index] += 2.0
            # compute g(r) H-H
            for i, hydrogen1 in enumerate(self.data_hydrogen[:, :, step]):
                for j in range(self.resolution):
                    r1 = j * dr
                    r2 = r1 + dr
                    v1 = volume(r1)
                    v2 = volume(r2)
                    volumes[j] += v2 - v1
                if i % 2 != 0:
                    hydrogen2 = np.delete(self.data_hydrogen[:, :, step], i - 1, axis=0)
                else:
                    hydrogen2 = np.delete(self.data_hydrogen[:, :, step], i + 1, axis=0)
                for hydrogen_at in hydrogen2:
                    dist = distance(hydrogen1, hydrogen_at, self.x_max, self.y_max, self.z_max)
                    index = int(dist / dr)
                    if 0 < index < self.resolution:
                        self.g_of_rHH[index] += 2.0
            print('finished step', step)
        """ normalize by the volume of the spherical shell corresponding to each radius """
        for i, value in enumerate(self.g_of_rOO):
            self.g_of_rOO[i] = value * self.volume_per_h2o / volumes[i]
        for i, value in enumerate(self.g_of_rOH):
            self.g_of_rOH[i] = value * self.volume_per_h2o / (2*volumes[i])
        for i, value in enumerate(self.g_of_rHH):
            self.g_of_rHH[i] = value * self.volume_per_h2o / (4*volumes[i])

    def compute_profile_density(self):  # calculates profile density along z

        dz = self.z_max / self.resolution
        self.z_bins = np.linspace(0.0, self.z_max + dz, self.resolution + 1)
        self.rho_of_O = np.zeros(self.resolution + 1)
        for step in range(self.n_steps):
            for j, oxygen_coor in enumerate(self.data_oxygen[:, :, step]):
                index = int(oxygen_coor[2]/dz)
                self.rho_of_O[index] += 1.0  # counts the water molecules in each bins
            print('finished step', step)
        self.rho_of_O = self.rho_of_O * 180 / (self.x_max*self.y_max*dz*6.023*self.n_steps)
        # calculates density in g/cm3

    def compute_z_drift(self, n):
        z_pos = self.data_oxygen[:, 2, :]
        z_drift = np.zeros((int(self.n_atoms/3), int(self.n_steps)))
        z_drift_step = np.zeros(self.n_steps)
        for step in range(int(self.n_steps - 1)):
            step += 1
            z_drift[:, step] = (z_pos[:, step] - z_pos[:, step - 1])**2
            z_drift_step[step] = np.average(z_drift[:, step]) + z_drift_step[step - 1]
        z_diffusion = pd.DataFrame(z_drift_step)
        filename = str('z-msd' + n + '.txt')
        z_diffusion.to_csv(filename, sep='\t')
        plt.figure()
        plt.plot(z_drift_step)
        plt.show()

    def plot_g_r(self, filename):
        """ plots the radial distribution function
        if filename is specified, prints it to file as a pdf, eps and png """

        if not self.g_of_rOO.any():
            print('compute the radial distribution function first\n')
            return
        plt.figure()
        plt.xlabel('r (Å)', fontsize=20)
        plt.ylabel('g(r)', fontsize=20)
        plt.plot(self.radii, self.g_of_rOO, label='$O-O$')
        plt.plot(self.radii, self.g_of_rOH, label='$O-H$')
        plt.plot(self.radii, self.g_of_rHH, label='$H-H$')
        plt.legend()
        plt.tight_layout()
        for suffix in 'png eps pdf'.split():
            plt.savefig(filename + "." + suffix)
        plt.show()
        np.savetxt(filename + 'HH.txt', self.g_of_rHH)
        np.savetxt(filename + 'OH.txt', self.g_of_rOH)
        np.savetxt(filename + 'OO.txt', self.g_of_rOO)

    def plot_rho(self, filename):
        """ plots the radial distribution function
        if filename is specified, prints it to file as a pdf """

        if not self.rho_of_O.any():
            print('compute the profile density first\n')
            return
        plt.figure()
        plt.title('Density profile', fontsize=20)
        plt.xlabel('z (Å)', fontsize=20)
        plt.ylabel(r'$\rho$ [g cm$^{-3}$]', fontsize=20)
        plt.plot(self.z_bins, self.rho_of_O)
        plt.tight_layout()
        for suffix in 'png eps pdf'.split():
            plt.savefig(filename + "." + suffix)
        plt.show()
        np.savetxt(filename + '.txt', self.rho_of_O)


os.chdir("xyz/")
filenames = [i for i in glob.glob('*.xyz')]
H2O = None
for name in filenames:
    H2O = Trajectory(name, 1)
    num = name[11:-4]
    # H2O.compute_radial_distribution()
    # filename_rdf = 'RDF ' + num
    # H2O.plot_g_r(filename_rdf)
    # H2O.compute_profile_density()
    # filename_rho = 'Rho ' + num
    # H2O.plot_rho(filename_rho)
    H2O.compute_z_drift(num)
    name_dump = str('objs' + num + '.pkl')
    with open(name_dump, 'wb') as g:  # Python 3: open(..., 'wb')
        pickle.dump([H2O.rho_of_O, H2O.g_of_rOO], g)
