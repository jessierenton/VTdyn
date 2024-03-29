import numpy as np
import multiprocessing as mp
from functools import partial
from itertools import repeat
import libs.stress_dep_lib as lib #library for simulation routines
import libs.data as data
from structure.global_constants import *
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth
import sys
from optparse import OptionParser

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

def run_single(i,stress_threshold,T_D,outdir=None,seed=None,return_value="h"):
    rand = np.random.RandomState(seed)
    history = lib.run_simulation(simulation,l,timestep,timend,rand,progress_on=options.progress_on,til_fix=options.til_fix,
                init_time=init_time,save_areas=options.save_areas,store_dead=True,save_events=options.save_events,T_D=T_D,ancestors=options.ancestors,
                stress_threshold=stress_threshold,N_limit=N_limit)
    if outdir is not None:
        data.save_stress(history,outdir,index=i)
        data.save_N_cell(history,outdir,index=i)
        data.save_info(history,outdir,index=i,**{"stress threshold":(stress_threshold,"%.1f"),
            "cell cycle": ("delayed uniform (low=%d, high=%d)"%(T_other,T_other+T_G1*2),"%s"), 
            "cell death": ("poisson (mean=%.1f)"%T_D,"%s")})
    if return_value == "h": return history

@full_traceback
def run_single_for_parallel(args):
    return run_single(*args)

def multiple_runs(repeats,parameters,outdir=None,return_value=None):  
    num_pvals = len(parameters)
    if outdir is not None and outdir[-1]!='/': outdir+='/'
    pool = mp.Pool(mp.cpu_count(),maxtasksperchild=1000)     
    args = [(i,stress_threshold,T_D,outdir+"/Td%.1f_pc%.1f"%(T_D,stress_threshold),None,return_value)
                if outdir is not None else (i,stress_threshold,T_D,None,return_value)
                for stress_threshold,T_D in parameters for i in range(repeats)]
    histories = [f for f in pool.imap(run_single_for_parallel,args)]
    pool.close()
    pool.join()
    return histories    

parser = OptionParser()
parser.set_defaults(save_events=False,progress_on=False,til_fix=False,seed=None,ancestors=False,outdir=None,
                    repeats=None,parameter_file=None,stress_threshold=10.,T_D=T_D,save_areas=False)
parser.add_option("-e","--events", action="store_true", dest="save_events",
                    help="store tissue every time a birth/death event occurs")
parser.add_option("-p","--progress", action="store_true", dest="progress_on",
                    help="show progress bar")
parser.add_option("-f","--fixation", action="store_true",dest="til_fix", 
                    help="stop simulation after mutants have fixed or died out")
parser.add_option("-s","--seed", type="int", dest="seed", metavar="SEED",
                    help="supply seed for random state")                   
parser.add_option("-a","--ancestors", action="store_true", dest="ancestors",
                    help="track ancestry of cells")  
parser.add_option("-d","--data", type="str", dest="outdir",metavar="DIRECTORY",
                    help="save data for simulation in given directory")                
parser.add_option("-t","--threshold-pressure",type="float",dest="stress_threshold",metavar="THRESHOLD",
                    help="define the stress threshold above which proliferation does not occur")
parser.add_option("-T",dest="T_D",metavar="TIME", type="float",
					help="specify the mean age of apoptosis or no apoptosis if -1")
parser.add_option("-m","--multiple", type="int", dest="repeats",metavar="REPEATS",
                    help="run given number of simulations in parallel")
parser.add_option("-P","--parameters",type="str", dest="parameter_file",metavar="FILE",
                    help="provide parameter file for running multiple simulations")
parser.add_option("-A", "--areas", action="store_true",dest="save_areas",
                    help="save areas of cells in simulations")
(options,args) = parser.parse_args()

"""run a single voronoi tessellation model simulation"""

l = 10 # population size N=l*l
timend = float(args[0]) # simulation time (hours)
timestep = 1.  # time intervals to save simulation history
init_time=10.
N_limit=500
if options.T_D<0: options.T_D=None
simulation = lib.simulation_stress_dependent
if options.repeats is None: 
    history = run_single(0,options.stress_threshold,options.T_D,outdir=options.outdir,seed=options.seed,return_value="h")
else: 
    if options.parameter_file is not None: parameters = np.loadtxt(options.parameter_file)
    else: parameters = [(options.stress_threshold,options.T_D)]
    multiple_runs(options.repeats,parameters,outdir=options.outdir)

