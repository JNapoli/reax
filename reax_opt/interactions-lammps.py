class Interaction(object):
    """Base class for all interactions."""

    def __init__(self):

        pass

    def _update(x):

        raise NotImplementedError

    def V(self, x):
        """Get the potential energy for positions `x`, updating if needed."""

        if not np.equal(x, self._x).all():
            self._update(x)

        return self._V

    def dV_dx(self, x):
        """Get the potential energy derivatives for positions `x`, updating if needed."""

        if not np.equal(x, self._x).all():
            self._update(x)

        return self._dV_dx


class LAMMPS(Interaction):
    """Interactions taken from LAMMPS.

    This version works for classical MD and PIMD and uses a single local
    LAMMPS instance.

    Positions are cached and and energies and forces reevaluated
    only if needed.
    """

    def __init__(self, fn_lmp, P=1, units_lmp='electron'):
        """ """

        if units.name != 'atomic':
            raise _error.PyMDInputError('Use atomic units in PyMD when using LAMMPS.')

        if units_lmp == 'electron':
            self.unit_lmp_x = 1.0
            self.unit_lmp_V = 1.0
            self.unit_lmp_F = 1.0
            self.unit_lmp_m = units.u
        elif units_lmp == 'real':
            self.unit_lmp_x = units.A
            self.unit_lmp_V = units.kcal_mol
            self.unit_lmp_F = units.kcal_mol / units.A
            self.unit_lmp_m = units.u
        else:
            raise _error.PyMDInputError('Unsupported units for LAMMPS: %s' % units)

        # store P
        self.P = P

        # read and store LAMMPS commands
        lammps_in = open(fn_lmp).read().splitlines()
        self._lammps_in = lammps_in

        # make sure neighbor lists are checked every step
        # TODO: leave this up to user settings?
        lammps_in.append('neigh_modify delay 0 every 1 check yes')

        # set up LAMMPS instance
        self.setup()
        N = self.N

        # cached positions - to check the need to calculate again
        # this should be different from any atomic configuration
        if P == 1:
            self._x = np.zeros(N)
        else:
            self._x = np.zeros((N,P))

        # allocate energy and derivatives caches
        if P == 1:
            self._V = None
            self._dV_dx = np.zeros(N)
        else:
            self._V = np.empty(P)
            self._dV_dx = np.zeros((N,P))

    def setup(self):

        # create LAMMPS instance
        lmp = lammps.lammps(cmdargs=['-screen', 'none', '-log', 'none'])
        self._lmp = lmp

        # execute all the LAMMPS commands
        for command in self._lammps_in:
            lmp.command(command)

        # number of atoms and degrees of freedom
        natoms = lmp.get_natoms()
        self.natoms = natoms
        N = 3 * natoms
        self.N = N

        # ctypes array for positions for LAMMPS
        self._x_c = (N * ct.c_double)()

    def __getstate__(self):

        state = self.__dict__.copy()

        # can't pickle the LAMMPS instance or ctypes array
        state['_lmp'] = None
        state['_x_c'] = None

        return state

    def __setstate__(self, state):

        self.__dict__ = state

        self.setup()

    def _update(self, x):

        timer = _global.timer
        timer.start('LAMMPS update')

        # LAMMPS instance for convenience
        lmp = self._lmp

        # cache new input positions
        self._x[:] = x

        if self.P == 1:

            # update LAMMPS positions
            self._x_c[:] = self._x / self.unit_lmp_x
            lmp.scatter_atoms('x', 2, 3, self._x_c)

            # update energy and forces in LAMMPS
            timer.start('LAMMPS run')
            lmp.command('run 1 pre no post no')
            timer.stop('LAMMPS run')

            # get energy and forces from LAMMPS
            self._V = float(lmp.extract_compute('thermo_pe', 0, 0)) * self.unit_lmp_V
            F_c = lmp.gather_atoms('f', 1, 3)
            self._dV_dx[:] = F_c[:]

        else:

            for i in range(self.P):

                # update LAMMPS positions
                self._x_c[:] = self._x[:,i] / self.unit_lmp_x
                lmp.scatter_atoms('x', 2, 3, self._x_c)

                # update energy and forces in LAMMPS
                timer.start('LAMMPS run')
                lmp.command('run 1 pre no post no')
                timer.stop('LAMMPS run')

                # get energy and forces from LAMMPS
                self._V[i] = float(lmp.extract_compute('thermo_pe', 0, 0)) * self.unit_lmp_V
                F_c = lmp.gather_atoms('f', 1, 3)
                self._dV_dx[:,i] = F_c[:]

        self._dV_dx *= - self.unit_lmp_F

        timer.stop('LAMMPS update')

    def info(self):
        """ """

        # TODO

        raise NotImplementedError

    def get_masses(self):

        # convenience
        lmp  = self._lmp
        natoms = self.natoms

        # get masses of all DOFs
        types = lmp.gather_atoms('type', 0, 1)
        mass_types = lmp.extract_atom('mass', 2)
        mass_atoms = np.empty(self.natoms)
        for i in range(self.natoms):
            mass_atoms[i] = mass_types[types[i]]
        m = mass_atoms.repeat(3) * self.unit_lmp_m

        return m

    def get_names_pos(self, fn_ini):
        """Optionally use initial positions and atom names from a file.

        If `None`, fake atom name are used and initial positions are taken
        from the interactions provider.
        """

        if fn_ini is None:
            names = self.natoms * ['X']
            x_c = self._lmp.gather_atoms('x', 1, 3)
            x0 = np.empty(self.N)
            for i in range(self.N):
                x0[i] = x_c[i] * self.unit_lmp_x
        else:
            fn, ext = os.path.splitext(fn_ini)
            if ext == '.xyz':
                comment, names, pos = io.read_XYZ_frame(open(fn_ini))
                x0 = pos.reshape(-1) * units.A
            elif ext == '.gro':
                results = io.read_GRO_frame(open(fn_ini))
                pos = results[5]
                x0 = pos.reshape(-1) * units.nm
                names = results[3]
            else:
                raise _error.PyMDInputError('Unsupported extension of reference file.')

        return names, x0

