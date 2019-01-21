import sys
import numpy as np
import itertools
import structure
from structure.global_constants import *
from structure.cell import Tissue, BasicSpringForceNoGrowth, MutantSpringForce
import structure.initialisation as init


def print_progress(step,N_steps):
    sys.stdout.write("\r %.2f %%"%(step*100/N_steps))
    sys.stdout.flush() 

def simulation_no_division(tissue,dt,N_steps,rand):
    step = 0.
    while True:
        N= len(tissue)
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt))
        tissue.update(dt)
        print_progress(step,N_steps)  
        yield tissue
        
def run(tissue_original,simulation,N_step,skip):
    return [tissue_original.copy()]+[tissue.copy() for tissue in itertools.islice(simulation,skip-1,N_step,skip)]


    
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------SIZE-DEPENDENCE-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def simulation_size_dependent(tissue,dt,N_steps,stepsize,rand):
    step = 0.
    while True:
        if len(np.where(tissue.properties['ancestor']!=tissue.properties['ancestor'][0])[0])==1 and step%stepsize==1: break
        N= len(tissue)
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt))
        ready = np.where(tissue.mesh.areas>=DIV_AREA)[0]
        for mother in ready:
            tissue.add_daughter_cells(mother,rand)
            tissue.properties['ancestor'] = np.append(tissue.properties['ancestor'],[tissue.properties['ancestor'][mother]]*2)
        tissue.remove(ready)
        if rand.rand() < (1./T_D)*N*dt:
            tissue.remove(rand.randint(N))
        tissue.update(dt)
        print_progress(step,N_steps)
        yield tissue

def run_simulation_size_dependent(N,timestep,timend,rand):
    ages = rand.rand(N*N)*(T_G1+T_other)
    multiplier = RHO+GROWTH_RATE*0.5*(T_G1+T_other)
    force = BasicSpringForce()
    tissue = init.init_tissue_torus_with_multiplier(N,N,0.01,force,rand,multiplier,ages,save_areas=True)
    tissue.properties['ancestor'] = np.arange(N*N)
    history = run(tissue,simulation_size_dependent(tissue,dt,timend/dt,timestep/dt,rand=rand),timend/dt,timestep/dt)
    return history

