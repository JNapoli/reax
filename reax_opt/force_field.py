import sys


def ff_write(fn_ff, param_list):
    """
    Utility function for writing reax force
    field file.
    """
    with open(fn_ff + '_template', 'r') as f:
        ff = ''.join(f.readlines())

    ff = ff.format(*param_list)

    with open(fn_ff, 'w') as f:
        f.write(ff)
        

def ff_read(fn_ff):
    """
    Utility function for reading a reax
    force field file and extracting
    its parameters.
    """
    pass

