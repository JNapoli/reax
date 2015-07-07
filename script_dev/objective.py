import ctypes as ct
import numpy as np

import lmps_interact, os, target


def prepare_coords(x, pose_coords):
    """
    Function for filling the ctype array
    with values for each d.o.f.
    """
    curr = 0

    for atom in pose_coords:
        for dof in atom:
            x[curr] = dof
            curr += 1


class Objective:
    def __init__(self, target, lmp_engine):
        """
        Compute objective function, given an engine instance
        and target configuration series.
        """
        self._tar = target
        self._lmp = lmp_engine
        self._monomer_in = ''
        self._monomer_data = ''
        self.X2 = self.compute_X2()

    def compute_X2(self):

        X2 = None
        lmp = self._lmp
        N = lmp.N
        e_series = []

        if self._tar.series_type == 'monomer':
            # Loop over the target's poses
            # and extract eneries from lmp_engine
            x_tmp = (N * ct.c_double)()

            for pose in self._tar.poses:
                prepare_coords(x_tmp, pose[0])
                lmp.update(x_tmp)
                e_series.append(lmp.V)

        elif self._tar.series_type == 'dimer':
            # Requires auxiliary engine
            # for monomer calculations
            #lmp_mono = #instantiate here

            x_tmp_monomer = (9 * ct.c_double)()
            x_tmp_dimer = (N * ct.c_double)()
            e_m1 = []
            e_m2 = []
            e_d = []

            for pose in self._tar.poses:
                coords = pose[0]
                h2o1_coords = coords[:3]
                h2o2_coords = coords[3:]

                # Compute each monomer separately
                prepare_coords(x_tmp_monomer, h2o1_coords)
                lmp_mono.update(x_tmp_monomer)
                e_m1.append(lmp_mono.V)

                prepare_coords(x_tmp_monomer, h2o2_coords)
                lmp_mono.update(x_tmp_monomer)
                e_m2.append(lmp_mono.V)

                # Dimer
                prepare_coords(x_tmp_dimer, coords)
                lmp.update(x_tmp_dimer)
                e_d.append(lmp.V)

            e_2body  = e_d - e_m1
            e_2body -= e_m2
            e_series = e_2body

        else:
            pass

        e_series = np.array(e_series)
        # TODO: make X2 contribution coefficients more general
        X2 = sum(((e_series - target.energies) / unit.kT )**2) 

    return X2
