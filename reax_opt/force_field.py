import os
import sys


def ff_write(fn_ff, param_list):
    """
    Utility function for writing reax force
    field file.
    """
    abs_fn_ff = os.path.join(os.getcwd(),'forcefield',fn_ff + \
            '_template')

    with open(abs_fn_ff, 'r') as f:
        ff = ''.join(f.readlines())

    ff = ff.format(*param_list)

    fn_ff = os.path.join(os.getcwd(),'forcefield',fn_ff)
    with open(fn_ff, 'w') as f:
        f.write(ff)
        

def ff_read(fn_ff):
    """
    Utility function for reading a reax
    force field file and extracting
    its parameters.
    """
    pass

