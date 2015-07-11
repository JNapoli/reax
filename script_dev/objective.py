import ctypes as ct
import numpy as np

import lmps_interact
import os
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
        self._job_dir = os.getcwd()
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
            # Requires auxiliary engine
            # for monomer calculations
            os.chdir(os.path.join(os.getcwd(), 'monomer_files'))
            lmp_mono = lmps_interact.LAMMPS('in.water')

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

            e_m1 = np.array(e_m1)
            e_m2 = np.array(e_m2)
            e_dimer = np.array(e_dimer)
            e_2body  = e_dimer - e_m1
            e_2body -= e_m2
            e_series = e_2body
            os.chdir(self._job_dir)

        else:
            pass

        e_series = np.array(e_series)
        X2 = sum(((e_series - self._tar.energies) / unit.kT )**2) 

        return X2
