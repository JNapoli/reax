import ctypes as ct
import numpy as np

import lmps_interact
import os
import shutil
import sys
import target
import unit

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
        e_series = []

        if self._tar.series_type == 'monomer':
            for pose in self._tar.poses:
                e_series.append(lmp.get_V(pose[0].reshape((1,-1))))

        elif self._tar.series_type == 'dimer':
            # Auxiliary engine for monomer calculations.
            # NOTE: Copy ff file we just wrote to the monomer_files dir
            os.chdir(os.path.join(self._root,'monomer_files'))
            shutil.copy(os.path.join(self._root,'forcefield','ffield_060614.reax'),'.')
            lmp_mono = lmps_interact.LAMMPS('in.water')
            os.chdir(self._root)

            e_m1 = []
            e_m2 = []
            e_dimer = []

            for pose in self._tar.poses:
                coords = pose[0]
                h2o1_coords = coords[:3]
                h2o2_coords = coords[3:]
                # Energy we want is the 2-body energy, so compute
                # each monomer separately.
                e_m1.append(lmp_mono.get_V(h2o1_coords.reshape((1,-1))[0]))
                e_m2.append(lmp_mono.get_V(h2o2_coords.reshape((1,-1))[0]))
                # Dimer
                e_dimer.append(lmp.get_V(coords.reshape((1,-1))[0]))

            e_m1, e_m2 = np.array(e_m1), np.array(e_m2)
            e_dimer = np.array(e_dimer)
            e_2body  = (e_dimer - e_m1) - e_m2
            e_series = e_2body

        else:
            # Trimer case not yet implemented
            pass

        # Write out results for correlation plots.
        np.savetxt('fit_energies.dat', np.array(e_series))

        e_series = np.array(e_series)
        diff = e_series - self._tar.energies
        X2 = sum((diff / np.average(diff))**2) / len(diff)
        return X2
