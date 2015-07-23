#!/usr/bin/env python

import argparse
import force_field
import lmps_interact
import objective
import os
import sys
import target
import time
import numpy as np


def evaluate_objective(params, i, delta, h, tar, lmp):
    params[i] += delta * h
    force_field.ff_write('ffield_060614.reax', params)
    obj = objective.Objective(tar, lmp)
    return obj.X2


def get_X2(params, tar, lmp):
    force_field.ff_write('ffield_060614.reax', params)
    obj = objective.Objective(tar, lmp)
    return obj.X2


def perturb_params(param_list):
    """
    Slightly perturb each parameter, sampling 
    from a normal distribution.
    """
    perturbed = []

    for p in param_list:
        delta = np.random.normal(loc=0.0, scale=abs(p))
        while abs(delta) / abs(p) > 0.2:
            delta = np.random.normal(loc=0.0, scale=abs(p))
        updated_p = p + delta
        perturbed.append(updated_p)
    return perturbed


def main():
    t0 = time.time()

    fn_target = 'ts-ccpol.xyz'
    tar = target.Target(os.path.join(os.getcwd(), '..', 'paesani-dimers', fn_target)) 

    fn_lmps = 'in.water'
    lmp = lmps_interact.LAMMPS(fn_lmps)

    X2_series = []
    param_series = []
    #TODO: load parameter values from file
    params = [0.0283, 1.2885, 10.919, 0.9215]
    beta = 1/300.0
    iters = 0

    while True:
        # Evaluate X2 with current params
        if not iters:
            X2 = get_X2(params, tar, lmp)
            X2_series.append(X2)
            param_series = [[p] for p in params]
            iters += 1
            continue
        params_temp = perturb_params(params)
        X2_temp = get_X2(params_temp, tar, lmp)
        if X2_temp < X2_series[-1]:
            # Accept move
            # Record X2 and param steps
            X2_series.append(X2_temp)
            for i in range(len(params)):
                param_series[i].append(params_temp[i])
            params = params_temp
        else:
            pass
            # Accept move with probability given by 
            # Monte Carlo criterion
        if not iters % 10:
            # Dump X2 and param histories
            np.savetxt('param_series_dump.dat', np.array(param_series))
            np.savetxt('X2_series_dump.dat', np.array(X2_series))
            print 'Step % 04d, Objective: % .3f' % (iters,
                    X2_series[-1])
        iters += 1

    #grads  = [0.0] * len(params)
    #for i, p in enumerate(params):
    #    h = 0.2 * p
    #    X2_1, X2_2 = (evaluate_objective(params, i, delta, h, tar, lmp) for delta in
    #            [-1,1])
    #    grad = (X2_2 - X2_1) / 2.0 / h
    #    grads[i] = grad
    #print 'Gradients: ' + str(grads)
    #print ''
    #print 'Calculation took % .3f seconds.' % (time.time() - t0)

if __name__ == '__main__':
    main()
