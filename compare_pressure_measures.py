import numpy as np
import multiprocessing as mp
from functools import partial
from itertools import repeat
import libs.stress_dep_lib as lib #library for simulation routines
import libs.data as data
from structure.global_constants import *
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth
import sys,os
from optparse import OptionParser
import stress_measures as sm

def cell_pressure(tissue,i,pressure_type):
        """calculates the pressure p_i on a single cell i according to the formula p_i = sum_j mag(F^rep_ij.u_ij)/l_ij
        where F^rep_ij is the repulsive force between i and j, i.e. F^rep_ij=F_ij if Fij is positive, 0 otherwise;
        u_ij is the unit vector between the i and j cell centres and l_ij is the length of the edge between cells i and j"""
        edge_lengths = tissue.mesh.edge_lengths(i)
        if pressure_type == "virial":
            return virial_pressure(tissue,i)
        elif pressure_type == "repulsive":
            forces = tissue.Force.force_ij(tissue,i)
            forces[forces<0] = 0
        elif pressure_type == "magnitude":
            forces = np.fabs(tissue.Force.force_ij(tissue,i))
        elif pressure_type == "full":
            forces = tissue.Force.force_ij(tissue,i)
        return sum(forces/edge_lengths)    

def virial_pressure(tissue,i):
    area = tissue.mesh.areas[i]
    distances = 0.5*tissue.mesh.distances[i]
    forces = tissue.Force.force_ij(tissue,i)
    return 0.5*sum(forces*distances)/area
    
def tissue_pressure(tissue,pressure_type):
    return np.array([cell_pressure(tissue,i,pressure_type) for i in range(len(tissue))])

def full_traceback(func):
    import traceback, functools
    @functools.wraps(func)
    def full_traceback_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = "{}\n\nOriginal {}".format(e, traceback.format_exc())
            raise type(e)(msg)
    return full_traceback_wrapper

def run_single(i,T_D,seed=None,return_value="h"):
    rand = np.random.RandomState(seed)
    history = lib.run_simulation(simulation,l,timestep,timend,rand, init_time=init_time,save_areas=True,T_D=T_D, N_limit=N_limit,progress_on=True)
    if return_value == "h": 
        return history
    else: return None

@full_traceback
def run_single_for_parallel(args):
    return run_single(*args)

def multiple_runs(repeats,parameters,return_value="h"):  
    #import ipdb; ipdb.set_trace()
    num_pvals = len(parameters)
    pool = mp.Pool(mp.cpu_count(),maxtasksperchild=1000)  
    args_ = [(i,T_D,None,return_value) for T_D in parameters for i in range(repeats)]
    histories = [f for f in pool.imap(run_single_for_parallel,args_)]
    pool.close()
    pool.join()
    return histories    

def save_histories_data(T_D_vals,histories,outdir):
    for T_D,histories_by_TD in zip(T_D_vals,histories):
        for i,history in enumerate(histories_by_TD):
            if T_D is None: T_D = np.inf
            save_data(history,outdir+'lambda_%.0f/'%T_D,i)
			

def save_data(history,outdir,index): 
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    pop_size = [len(tissue) for tissue in history]
    np.savetxt(outdir+'/pop_size_%d'%index,pop_size,fmt='%3d')
    for stress_type in ("virial","full","repulsive"):
	stress = [sm.tissue_pressure(tissue,stress_type) for tissue in history]
        save_stress(stress,stress_type,outdir,index)
        
def save_stress(stress_data,stress_type,outdir,index):
    with open(outdir+'/pressure_%s_%d'%(stress_type,index),'w') as f:
        for tissue in stress_data:
            for cell_stress in tissue:
                f.write('%5e    '%cell_stress)
            f.write('\n')		

l = 10 # population size N=l*l
timend = 20. # simulation time (hours)
timestep = 1.  # time intervals to save simulation history
init_time=10
N_limit=1000

simulation = lib.simulation_no_stress_dependence
T_D_vals = (None,40.,30.,20.,15.,10.)
repeats = 3
histories = multiple_runs(repeats,T_D_vals,return_value="h")
histories = np.reshape(histories,(len(T_D_vals),repeats,len(histories[0])))
save_histories_data(T_D_vals,histories,'stress_measure_compare/')



