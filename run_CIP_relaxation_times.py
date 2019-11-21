import numpy as np
import libs.contact_inhibition_lib as lib #library for simulation routines
import libs.data as data
import libs.plot as vplt #plotting library
from structure.global_constants import *
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth
import matplotlib.pyplot as plt
import os

"""run a single voronoi tessellation model simulation"""

OUTDIR = "CIP_cell_division_relaxation_time2/"

l = 10 # population size N=l*l
timend = 30. # simulation time (hours)
timestep = 1.0 # time intervals to save simulation history

rand = np.random.RandomState()

simulation = lib.simulation_contact_inhibition_area_dependent  #simulation routine imported from lib
threshold_area_fraction=1.0
DEATH_RATE = 1./12
rates = (DEATH_RATE,DEATH_RATE/0.4) #death_rate,division_rate
domain_size_multiplier=0.980940

eta,mu,dt=1.,-250,0.001
T_m_init=0.1

def get_relaxation_data(T_m_vals,T_m_init,eta,mu,dt,relaxtime):
    history = lib.run_simulation(simulation,l,timestep,timend,rand,progress_on=True,
                init_time=None,til_fix=False,save_areas=True,cycle_phase=None,eta=eta,mu=mu,dt=dt,T_m=T_m_init,
                return_events=False,save_cell_histories=True,domain_size_multiplier=domain_size_multiplier,
                rates=rates,threshold_area_fraction=threshold_area_fraction)

    tissue = lib.run_return_final_tissue(lib.simulation_no_division(history[-1],dt,200,rand,eta),200) 
    division_ready = lib.check_area_threshold(tissue.mesh,threshold_area_fraction)
    mother = rand.choice(division_ready)
    tissue.add_daughter_cells(mother,rand)
    tissue.remove(mother,True)
    tissue.update(dt)
    init_tissues = [tissue.copy() for T_m in T_m_vals]
    for T_m,tissue in zip(T_m_vals,init_tissues):
        tissue.Force = BasicSpringForceNoGrowth(mu,T_m)
    histories = [lib.run(lib.simulation_no_division(tissue,dt,int(relaxtime/dt),rand,eta),int(relaxtime/dt),1) for tissue in init_tissues]
    for T_m,history in zip(T_m_vals,histories):
        cell1,cell2 = len(history[0])-2,len(history[0])-1
        sibling_distance = get_sibling_distance(history,cell1,cell2)
        mean_area = np.array([np.mean(tissue.mesh.areas[-2:]) for tissue in history])
        time = np.arange(0,relaxtime,dt)
        data = np.vstack((time,sibling_distance,mean_area))
        try: np.savetxt(OUTDIR+"T_m=%.3f.txt"%T_m,data)
        except IOError:
            os.makedirs(OUTDIR)
            np.savetxt(OUTDIR+"T_m=%.3f.txt"%T_m,data)

def narg(tissue,i,j):                                       
    try: return np.where(tissue.mesh.neighbours[i]==j)[0][0]
    except IndexError: return np.nan

def get_sibling_distance(history,cell1,cell2):
    return np.array([tissue.mesh.distances[cell1][narg(tissue,cell1,cell2)] if narg(tissue,cell1,cell2)<100 else np.nan for tissue in history])

relaxtime = 2.0
T_m_vals=[0.001,0.01,0.1,0.25,0.5,1.0,2.0]
   
get_relaxation_data(T_m_vals,T_m_init,eta,mu,dt,relaxtime)