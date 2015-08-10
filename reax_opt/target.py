import numpy as np
import sys

class Target(object):
    def __init__(self, fn_train):
        """
        Target initialized by loading all poses and their
        corresponding energies from file fn_train
        """
        self.poses, self.energies, self.series_type = self._load_poses(fn_train)
        np.savetxt('tgt-energies.dat',self.energies)

    def _load_poses(self, fn_train):

        poses = []

        # TODO: Function for parsing data directly from file
        with open(fn_train,'r') as f:
            raw_dat = [line.strip() for line in f.readlines()]
        
        len_chunk = int(raw_dat[0]) + 2
        start = 0   

        while True:
            if start == len(raw_dat): break
            lines = raw_dat[start:start+len_chunk]
            e_in_kcal_mol = float(lines[1])
            coord_lines = lines[2:]
            h2o_atom_coords = np.array([tuple(float(i) for i in line.split()[1:]) \
                                        for line in coord_lines])
            poses.append((h2o_atom_coords, e_in_kcal_mol))
            start += len_chunk
        
        energies = np.array([elem[1] for elem in poses])

        if len(poses[0][0]) == 3:
            s_type = 'monomer'
        elif len(poses[0][0]) == 6:
            s_type = 'dimer'
        elif len(poses[0][0]) == 9:
            s_type = 'trimer'
        elif len(poses[0][0]) == 7:
            s_type = 'hydronium'

        return poses, energies, s_type

