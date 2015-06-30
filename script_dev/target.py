from itertools import islice
import sys

class Target:
    def __init__(self, fn_train):
        self.poses = self.load_poses(fn_train)

    def load_poses(self, fn_train):
        poses = []
        with open(fn_train,'r') as f:
            raw_dat = [line.strip() for line in f.readlines()]
        len_chunk = int(raw_dat[0]) + 2
        start = 0    
        while True:
            if start == len(raw_dat): break
            lines = raw_dat[start:start+len_chunk]
            e_in_Ha = float(lines[1])
            coord_lines = lines[2:]
            h2o_atom_coords = [tuple(line.split()[1:]) for line in coord_lines]
            poses.append((h2o_atom_coords, e_in_Ha))
            start += len_chunk
        return poses
