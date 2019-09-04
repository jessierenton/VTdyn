import numpy as np
# from pathos.multiprocessing import cpu_count
# from pathos.pools import ParallelPool as Pool
from multiprocessing import Pool,cpu_count
import libs.contact_inhibition_lib as lib #library for simulation routines
import libs.data as data
from structure.global_constants import *
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth
import sys,os
import itertools 

L = 8 # population size N=l*l
TIMEND = 1000. # simulation time (hours)
MAX_POP_SIZE = 500
TIMESTEP = 1. # time intervals to save simulation history
G_TO_S_RATE,S_TO_DIV_RATE = 0.5,0.1
INIT_TIME = None
DATA_SAVE_ROUTINES = (data.save_N_cell,data.save_cell_histories,data.save_cycle_phases,data.save_cell_density,
                        data.save_tension_area_product)
DATA_SAVE_FIELDS = ["pop_size","division_history","extrusion_history","cycle_phases","density","energy",
                    "cell_seperation"]

for d in DATA_SAVE_FIELDS:
    if d not in data.FIELDS_DICT:
        raise ValueError("not all data types are correct")

PARENTDIR = "CIP_data_fixed_cc_rates_fixed_size_sweep"
# if not os.path.exists(PARENTDIR): # if the folder doesn't exist create it
#      os.makedirs(PARENTDIR)
with open(PARENTDIR+'/info',"w") as f:
    f.write('G_to_S_rate = %.3f\nS_to_div_rate\n = %.3f'%(G_TO_S_RATE,S_TO_DIV_RATE))
    f.write('initial cells = %3d'%(L*L))
simulation = lib.simulation_contact_inhibition  #simulation routine imported from lib

def run_single_unpack(args):
    return run_single(*args)

def run_single(i,threshold,death_rate,domain_size_multiplier):
    """run a single voronoi tessellation model simulation"""
    CIP_parameters = {'threshold':threshold}
    rates = (death_rate,G_TO_S_RATE,S_TO_DIV_RATE)
    rand = np.random.RandomState()
    history = lib.run_simulation(simulation,L,TIMESTEP,TIMEND,rand,progress_on=True,
                init_time=INIT_TIME,til_fix=False,save_areas=False,cycle_phase=True,
                return_events=False,store_dead=True,N_limit=MAX_POP_SIZE,
                domain_size_multiplier=domain_size_multiplier,
                CIP_parameters=CIP_parameters,rates=rates)
    outdir = PARENTDIR+'/DRate%.3f_Thresh%02.1f'%(death_rate,threshold)
    # save_data(history,outdir,i)
    data.save_as_json(history,outdir,DATA_SAVE_FIELDS,{"threshold":threshold,"death_rate":death_rate,
                    "width":history[0].mesh.geometry.width,"height":history[0].mesh.geometry.height,
                    "domain_multiplier":domain_size_multiplier,"G_rate":G_TO_S_RATE,"S_rate":S_TO_DIV_RATE},i)

def save_data(history,outdir,index,data_save_routines=DATA_SAVE_ROUTINES):
    for save in DATA_SAVE_ROUTINES:
        save(history,outdir,index)
    
def run_parallel(threshold_vals,death_rate_vals,dsm_vals,idx):
    pool = Pool(cpu_count()-1,maxtasksperchild=1000)
    args = list(zip())
    args = [(idx,threshold,death_rate,domain_size_multiplier) 
                for threshold,death_rate,domain_size_multiplier in itertools.product(threshold_vals,death_rate_vals,dsm_vals)]
    pool.map(run_single_unpack,args)
threshold_vals = np.arange(-10,60,5)
death_rate_vals = np.arange(0.01,0.085,0.005)
domain_size_multiplier = 1.5
dsm_vals=[domain_size_multiplier]
idx = int(sys.argv[1])
run_parallel(threshold_vals,death_rate_vals,dsm_vals,idx)           