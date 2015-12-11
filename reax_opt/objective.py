import ctypes as ct
import numpy as np

import lmps_interact
import os
import shutil
import sys
import target
import unit

# minimum energy monomer configuration
# in the partridge-schwencke surface
PS_minimum = np.array([-1.556,
                       -0.111,
                        0.0,
                       -1.947,
                        0.764,
                        0.0,
                       -0.611,
                        0.049,
                        0.0])

class Objective(object):
    def __init__(self, tar, lmp_engine):
        """
        Compute objective function, given an engine instance
        and target configuration series.
        """
        self._tar     = tar
        self._lmp     = lmp_engine
        self._root    = os.getcwd()
        self.X2       = self.compute_X2()

    def compute_X2(self):
        X2 = None
        lmp = self._lmp
        N = lmp.N
        type = self._tar.series_type
        snaps = [pose[0] for pose in self._tar_poses]
        e_primary = np.array([lmp.get_V(coords.reshape((1,-1))[0]) \
                              for coords in snaps])

        if type == 'monomer' or type == 'hydronium':
            # '1-body' energy. we subtract the energy of the structure
            # minimized in the Partridge-Schwenke potential.
            PS_energy = lmp.get_V(PS_minimum)
            e_series_final = e_primary - PS_energy

        elif type == 'dimer':
            # Auxiliary engine for monomer calculations.
            # NOTE: Copy ff file we just wrote to the monomer_files dir
            os.chdir(os.path.join(self._root,'monomer_files'))
            shutil.copy(os.path.join(self._root,'forcefield','ffield_060614.reax_modified'),'.')
            lmp_m = lmps_interact.LAMMPS('in.water')
            os.chdir(self._root)
            e_m1 = np.array([lmp_m.get_V(coords[:3].reshape((1,-1))[0]) \
                             for coords in snaps])
            e_m2 = np.array([lmp_m.get_V(coords[3:].reshape((1,-1))[0]) \
                             for coords in snaps]) 
            # 2-body energy
            e_series_final = e_primary - (e_m1 + e_m2)

        else:
            # Monomer engine
            os.chdir(os.path.join(self._root,'monomer_files'))
            shutil.copy(os.path.join(self._root,'forcefield','ffield_060614.reax_modified'),'.')
            lmp_m = lmps_interact.LAMMPS('in.water')
            os.chdir(self._root)
            # Dimer engine
            os.chdir(os.path.join(self._root,'dimer_files'))
            shutil.copy(os.path.join(self._root,'forcefield','ffield_060614.reax_modified'),'.')
            lmp_d = lmps_interact.LAMMPS('in.water')
            os.chdir(self._root)

            e_m1 = np.array([lmp_m.get_V(coords[:3].reshape((1,-1))[0]) \
                             for coords in snaps])
            e_m2 = np.array([lmp_m.get_V(coords[3:6].reshape((1,-1))[0]) \
                             for coords in snaps])
            e_m3 = np.array([lmp_m.get_V(coords[6:].reshape((1,-1))[0]) \
                             for coords in snaps]) 

            def two_body_series(lmp_d, e_m1, e_m2, dimer_snaps):
                e_pair = np.array([lmp_d.get_V(coords.reshape((1,-1))[0]) \
                                   for coords in dimer_snaps])
                return e_pair - (e_m1 + e_m2)

            d_snaps_1_2 = [coords[:6] for coords in snaps] 
            d_snaps_2_3 = [coords[3:] for coords in snaps]
            d_snaps_1_3 = [np.vstack((coords[:3],coords[6:])) for coords in snaps]
            e_2b_1 = two_body_series(lmp_d, e_m1, e_m2, d_snaps_1_2)
            e_2b_2 = two_body_series(lmp_d, e_m2, e_m3, d_snaps_2_3)
            e_2b_3 = two_body_series(lmp_d, e_m1, e_m3, d_snaps_1_3)
            # 3-body energy
            e_series_final = e_primary - (e_2b_1 + e_2b_2 + e_2b_3 + e_m1 + e_m2 + e_m3)

        self.e_series = e_series_final
        diff = self.e_series - self._tar.energies
        weights = self.get_weights(self.e_series)
        X2 = sum(weights * (diff / np.std(self._tar.energies))**2) / float(len(self._tar.energies))
        return X2

    def get_weights(self, e_series):
        tar_energies = self._tar.energies

        if self._tar.series_type == 'trimer':
            delta_E = 37.5
        elif self._tar.series_type == 'dimer':
            delta_E = 25.0
        else:
            return np.ones(len(e_series))

        weights = []
        E_min = min(e_series)

        for e in e_series:
            weight = (delta_E / (e - E_min + delta_E))**2
            weights.append(weight)

        return np.array(weights)

