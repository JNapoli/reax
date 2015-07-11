#!/usr/bin/env python

import argparse
import force_field
import lmps_interact
import objective
import os
import target
import time


def evaluate_objective(params, i, delta, h, tar, lmp):
    params[i] += delta * h
    force_field.ff_write('ffield_060614.reax', params)
    obj = objective.Objective(tar, lmp)
    return obj.X2


def main():
    t0 = time.time()

    fn_target = 'ts-ccpol.xyz'
    tar = target.Target(os.path.join(os.getcwd(), '..', 'paesani-dimers', fn_target)) 

    fn_lmps = 'in.water'
    lmp = lmps_interact.LAMMPS(fn_lmps)

    params = [0.0283, 1.2885, 10.919, 0.9215]
    grads  = [0.0] * len(params)

    for i, p in enumerate(params):
        h = 0.1 * p
        X2_1, X2_2 = (evaluate_objective(params, i, delta, h, tar, lmp) for delta in
                [-1,1])
        grad = (X2_2 - X2_1) / 2.0 / h
        grads[i] = grad
        
    print 'Gradients: ' + str(grads)

    print ''
    print 'Calculation took % .3f seconds.' % (time.time() - t0)

if __name__ == '__main__':
    main()
