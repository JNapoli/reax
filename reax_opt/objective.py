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
                e_series.append(lmp.get_V(pose[0].reshape((1,-1))[0]) + 248.739362825)

        elif self._tar.series_type == 'dimer':
            # Auxiliary engine for monomer calculations.
            # NOTE: Copy ff file we just wrote to the monomer_files dir
            os.chdir(os.path.join(self._root,'monomer_files'))
            shutil.copy(os.path.join(self._root,'forcefield','ffield_060614.reax_modified'),'.')
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
            e_series = (e_dimer - e_m1) - e_m2

        else:
            # Monomer engine
            os.chdir(os.path.join(self._root,'monomer_files'))
            shutil.copy(os.path.join(self._root,'forcefield','ffield_060614.reax_modified'),'.')
            lmp_mono = lmps_interact.LAMMPS('in.water')
            os.chdir(self._root)
            # Dimer engine
            os.chdir(os.path.join(self._root,'dimer_files'))
            shutil.copy(os.path.join(self._root,'forcefield','ffield_060614.reax_modified'),'.')
            lmp_dimer = lmps_interact.LAMMPS('in.water')
            os.chdir(self._root)

            Es_3B = []
 
            def calc_two_body(dim_eng, dim_coords, E_tot_1_body):
                e_dimer  = dim_eng.get_V(dim_coords)
                return e_dimer - E_tot_1_body
            
            for pose in self._tar.poses:
                E_2_body_total = 0.0
                coords = pose[0]
                h2o_1_coords = coords[:3].reshape((1,-1))[0]
                h2o_2_coords = coords[3:6].reshape((1,-1))[0]
                h2o_3_coords = coords[6:].reshape((1,-1))[0]
                pair_1 = coords[:6].reshape((1,-1))[0]
                pair_2 = coords[3:].reshape((1,-1))[0]
                pair_3 = np.vstack((coords[:3],coords[6:])).reshape((1,-1))[0]
                Es_1_body = []
                Es_1_body.append(lmp_mono.get_V(h2o_1_coords))
                Es_1_body.append(lmp_mono.get_V(h2o_2_coords))
                Es_1_body.append(lmp_mono.get_V(h2o_3_coords))
                E_2_body_total += calc_two_body(lmp_dimer, pair_1, sum(Es_1_body[:2])) + \
                                  calc_two_body(lmp_dimer, pair_2, sum(Es_1_body[1:])) + \
                                  calc_two_body(lmp_dimer, pair_3, sum([Es_1_body[i] for i in [0,2]]))  
                E_trimer        = lmp.get_V(coords.reshape((1,-1))[0])
                E_3_body = (E_trimer - E_2_body_total) - sum(Es_1_body)
                Es_3B.append(E_3_body)

            e_series = Es_3B

        e_series = np.array(e_series)
        self.e_series = e_series
        diff = e_series - self._tar.energies
        X2 = sum((diff / np.average(diff))**2)
        return X2
