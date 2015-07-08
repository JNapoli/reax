
import ctypes as ct
import numpy as np

import lammps


class LAMMPS(object):

    def __init__(self, fn_lmp):

        # Load all LAMMPS commands
        lammps_in = open(fn_lmp).read().splitlines()
        self._lammps_in = lammps_in

        # Neighbor lists checked at every step
        lammps_in.append('neigh_modify delay 0 every 1 check yes')

        # Instantiate
        lmp = lammps.lammps(cmdargs=['-screen','none','-log','none'])
        self._lmp = lmp

        # Command execution
        for cmd in self._lammps_in:
            lmp.command(cmd)

        # Gather some constants
        natoms = lmp.get_natoms()
        self.natoms = natoms
        N = 3 * natoms
        self.N = N

        # Ctypes array for LAMMPS position data
        self._x_c = (N * ct.c_double)()

    def get_V(self, x):

        # Our LAMMPS instance
        lmp = self._lmp

        # Update LAMMPS positions 
        self._x_c[:] = x
        lmp.scatter_atoms('x', 2, 3, self._x_c)

        # Update energy and forces in LAMMPS
        lmp.command('run 1 pre no post no')

        # Get energy and forces from LAMMPS
        V = float(lmp.extract_compute('thermo_pe', 0, 0))
        #F_c = lmp.gather_atoms('f', 1, 3)
        #self._dV_dx[:]  = F_c[:]
        #self._dV_dx[:] *= -1.0
        return V
