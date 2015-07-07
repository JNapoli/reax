from lammps import lammps

import ctypes as ct
import numpy as np

class Interaction(object):
    """Base class for all interactions, from Ondrej"""
    
    def __init__(self):
        pass

    def _update(x):
        raise NotImplementedError

    def V(self, x):
        if not np.equal(x, self._x).all():
            self._update(x)
        return self.V

    def dV_dx(self, x):
        if not np.equal(x, self._x).all():
            self._update(x)
        return self._dV_dx
    

class LAMMPS(Interaction):
    """
    Extract interactions from LAMMPS.

    Positions cached and energies and forces reevaluated if needed.
    """

    def __init__(self, fn_lmp, units_lmp='real'):

        # For now assume use of real units
        if units_lmp == 'real':
            self.unit_lmp_x = units.A
            self.unit_lmp_V = units.kcal_mol
            self.unit_lmp_F = units.kcal_mol / units.A
            self.unit_lmp_m = units.u
        else:
            raise RuntimeError('Use real units!')

        # Load all LAMMPS commands
        lammps_in = open(fn_lmp).read().splitlines()
        self._lammps_in = lammps_in

        # Neighbor lists checked at every step
        lammps_in.append('neigh_modify delay 0 every 1 check yes')

        # Set up calculation
        self.setup()
        # System d.o.f.
        N = self.N

        self._x = np.zeros(N)

        # Allocate energy and derivatives caches
        self.V = None
        self._dV_dx = np.zeros(N)

    def setup(self):

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

    def update(self, x):

        # Our LAMMPS instance
        lmp = self._lmp

        # Cache input positions
        self._x[:] = x

        # Update LAMMPS positions 
        self._x_c[:] = self._x / self.unit_lmp_x
        lmp.scatter_atoms('x', 2, 3, self._x_c)

        # Update energy and forces in LAMMPS
        lmp.command('run 1 pre no post no')

        # Get energy and forces from LAMMPS
        self.V = float(lmp.extract_compute('thermo_pe', 0, 0)) * self.unit_lmp_V
        F_c = lmp.gather_atoms('f', 1, 3)
        self._dV_dx[:] = F_c[:]
        self._dV_dx += - self.unit_lmp_F 
