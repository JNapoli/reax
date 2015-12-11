#!/usr/bin/env python

import math
import os
import random
import sys
import time
import numpy as np
import reax_opt as reax

save = 'dimer-large-spacing'

def instantiate_lammps(fn_in):
    root = os.getcwd()
    os.chdir(os.path.join(root,'forcefield'))
    lmp = reax.lmps_interact.LAMMPS(fn_in)
    os.chdir(root)
    return lmp

def get_X2(params, tar, first=False):
    reax.force_field.ff_write('ffield_060614.reax',params)
    lmp = instantiate_lammps('in.water')
    obj = reax.objective.Objective(tar,lmp)
    if first:
        e_series = obj.e_series
        np.savetxt('ff-'+save+'-shifted.dat',e_series)
        sys.exit()
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

def evaluate_objective(params, i, delta, h, tar):
    params[i] += delta * h
    reax.force_field.ff_write('ffield_060614.reax',params)
    lmp = instantiate_lammps('in.water')
    obj = reax.objective.Objective(tar,lmp)
    return obj.X2

def main():
    t0 = time.time()
    abs_fn_target = os.path.join(os.getcwd(),'targets','dimer-large-spacing.xyz')
    tar = reax.target.Target(abs_fn_target) 
    tar_energies = tar.energies
    np.savetxt('tgt-'+save+'.dat',tar_energies)

    X2_series = []
    param_series = []
    acceptance = []
    #TODO: load parameter values from file
    #params = [0.0283, 1.2885, 10.919, 0.9215]
    mbeta = -1.0/10.0
    accepted = 0
    trials = 0
    initialized = False
    params = np.loadtxt('parameters.dat')
    #params = [ 50.0000,
    #            9.5469,
    #           26.5405,
    #            1.7224,
    #            6.8702,
    #           60.4850,
    #            1.0588,
    #            4.6000,
    #           12.1176,
    #           13.3056,
    #          -70.5044,
    #            0.0000,
    #           10.0000,
    #            2.8793,
    #           33.8667,
    #            6.0891,
    #            1.0563,
    #            2.0384,
    #            6.1431,
    #            6.9290,
    #            0.3989,
    #            3.9954,
    #           -2.4837,
    #            5.7796,
    #           10.0000,
    #            1.9487,
    #           -1.2327,
    #            2.1645,
    #            1.5591,
    #            0.1000,
    #            2.1365,
    #            0.6991,
    #           50.0000,
    #            1.8512,
    #            0.5000,
    #           20.0000,
    #            5.0000,
    #            0.0000,
    #            2.6962]
    X2_init = get_X2(params, tar, first=True)
    grads_forward  = [0.0] * len(params)
    grads_backward = [0.0] * len(params)
    for i, p in enumerate(params):
        print ''
        print 'PARAMETER #' + str(i+1)
        print ''
        h = 0.2 * abs(p)
        X2_neg_1, X2_0, X2_1 = (evaluate_objective(params, i, delta, h, tar) for delta in [-1, 0, 1])
        grad_forward  = (X2_1 - X2_0) / h
        grad_backward = (X2_neg_1 - X2_0) / h
        grads_forward[i] = grad_forward
        grads_backward[i] = grad_backward
    np.savetxt('grads_forward.dat', grads_forward)
    np.savetxt('grads_backward.dat', grads_backward)
    print 'Gradients: ' + str(grads)
    print ''
    print 'Calculation took ' + str((time.time()-t0)/60.0) + ' minutes.'
    sys.exit()

    while True:
        if not initialized:
            # Calculate initial objective function.
            X2 = get_X2(params,tar,first=True)
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
            else:
                # Objective function and params stay the same
                X2_series.append(X2_series[-1])
                for i in range(len(params)):
                    param_series[i].append(param_series[i][-1])
        trials += 1
        acceptance.append(float(accepted) / float(trials))
        if not trials % 10:
            # Dump X2 and param histories
            np.savetxt('param_series_dump.dat', np.array(param_series))
            np.savetxt('X2_series_dump.dat', np.array(X2_series))
            np.savetxt('acceptance_rate.dat', np.array(acceptance))
        if not trials % 250:
            mbeta *= 2

if __name__ == '__main__':
    main()
