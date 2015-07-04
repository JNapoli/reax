import ctypes as ct
import numpy as np

import lmps_interact, os, target


def prepare_coords(x, pose):
    """
    Function for filling the ctype array
    with values for each d.o.f.
    """
    curr = 0

    for atom in pose[0]:
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
        self.X2 = compute_X2(target, lmp_engine)

    def compute_X2(self, target, lmp_engine):

        X2 = None
        lmp = lmp_engine
        N = lmp_engine.N
        e_series = []

        if target.series_type == 'monomer':
            # Loop over the target's poses
            # and extract eneries from lmp_engine
            x_tmp = (N * ct.c_double)()

            for pose in target.poses:
                prepare_coords(x_tmp, pose)
                lmp.update(x_tmp)
                e_series.append(lmp.V)

        elif target.series_type == 'dimer':
            # Requires auxiliary engine
            # for monomer calculations
            x_tmp_monomer = (3 * ct.c_double)()
            x_tmp_dimer = (N * ct.c_double)()
            e_m1 = []
            e_m2 = []
            e_d = []

            for pose in target.poses:
        else:
            pass

        e_series = np.array(e_series)
        # TODO: make X2 contribution coefficients more general
        X2 = sum(((e_series - target.energies) / 300.0 / units.kT)**2) 

    return X2
