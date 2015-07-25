#!/usr/bin/env python

import math
import os
import random
import sys
import time
import numpy as np
import reax_opt as reax


def instantiate_lammps(fn_in):
    root = os.getcwd()
    os.chdir(os.path.join(root,'forcefield'))
    lmp = reax.lmps_interact.LAMMPS(fn_in)
    os.chdir(root)
    return lmp


def get_X2(params, tar):
    reax.force_field.ff_write('ffield_060614.reax',params)
    lmp = instantiate_lammps('in.water')
    obj = reax.objective.Objective(tar,lmp)
    return obj.X2


def perturb_params(param_list):
    """
    Slightly perturb each parameter, sampling 
    from a normal distribution.
    """
    perturbed = []
    for p in param_list:
        delta = np.random.normal(loc=0.0, scale=abs(p))
        while abs(delta) / abs(p) > 0.1:
            delta = np.random.normal(loc=0.0, scale=abs(p))
        updated_p = p + delta
        perturbed.append(updated_p)
    return perturbed


def main():
    abs_fn_target = os.path.join(os.getcwd(),'targets','ts-ccpol.xyz')
    tar = reax.target.Target(abs_fn_target) 

    X2_series = []
    param_series = []
    acceptance = []
    #TODO: load parameter values from file
    params = [0.0283, 1.2885, 10.919, 0.9215]
    mbeta = -1.0/1.0
    accepted = 0
    trials = 0
    initialized = False

    while True:
        if not initialized:
            # Calculate initial objective function.
            X2 = get_X2(params,tar)
            X2_series.append(X2)
            param_series = [[p] for p in params]
            initialized = True
            continue
        params_temp = perturb_params(params)
        X2_temp = get_X2(params_temp,tar)
        if X2_temp < X2_series[-1]:
            # Accept move. Record X2 and param steps.
            X2_series.append(X2_temp)
            for i in range(len(params)):
                param_series[i].append(params_temp[i])
            params = params_temp
            accepted += 1
        else:
            if random.random() < math.exp(mbeta * (X2_temp - X2_series[-1])):
                X2_series.append(X2_temp)
                for i in range(len(params)):
                    param_series[i].append(params_temp[i])
                params = params_temp
                accepted += 1
        trials += 1
        acceptance.append(float(accepted) / float(trials))
        if not trials % 10:
            # Dump X2 and param histories
            np.savetxt('param_series_dump.dat', np.array(param_series))
            np.savetxt('X2_series_dump.dat', np.array(X2_series))
            np.savetxt('acceptance_rate.dat', np.array(acceptance))
            print 'Step % 04d, Objective: % .3f' % (trials,
                    X2_series[-1])

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
