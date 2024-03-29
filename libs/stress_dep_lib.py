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

def run_save_events(tissue_original,simulation,N_step):
    return [tissue_original.copy()]+[tissue.copy() for tissue in itertools.islice(simulation,N_step)]

def run_save_final(simulation,N_step):
    return next(itertools.islice(simulation,N_step,None))


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------POISSON-BIRTH-DEATH----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def death_function_poisson(n,rand,T_D=T_D):
    return rand.exponential(T_D,n)

def cycle_function_uniform(n,rand,T_G1=T_G1,T_other=T_other):
    return rand.rand(n)*T_G1*2 + T_other

def simulation_no_stress_dependence(tissue,dt,N_steps,stepsize,rand,til_fix=False,progress_on=False,store_dead=False,save_events=False,T_D=T_D,N_limit=np.inf,**kwargs):  
    step = 0.
    complete = False
    properties = tissue.properties
    mesh = tissue.mesh
    while not til_fix or not complete:
        if progress_on: 
            print_progress(step,N_steps)
            step += 1
        N=len(tissue)
        if N <10 or N>=N_limit: 
            break
        mesh.move_all(tissue.dr(dt))
        births = np.where(properties['cycle_length']<=tissue.age)[0]
        if len(births)>0:
            for mother in births:
                daughter_properties = {'cycle_length':cycle_function_uniform(2,rand)}
                if T_D is not None: daughter_properties['age_of_death'] = death_function_poisson(2,rand,T_D=T_D)
                tissue.add_daughter_cells(mother,rand,daughter_properties)
            tissue.remove(births,True)
        if T_D is not None:        	
            deaths = np.where(properties['age_of_death']<=tissue.age)[0]
            if len(deaths)>0:
            	tissue.remove(deaths,False)
        tissue.update(dt)
        if til_fix: complete = (1 not in tissue.properties['type'] or 0 not in tissue.properties['type']) and step%stepsize==0 
        if not save_events or (len(births)!=0 or len(deaths)!=0): 
            yield tissue 

def simulation_constant_pop_size(tissue,dt,N_steps,stepsize,rand,til_fix=False,progress_on=False,store_dead=False,save_events=False,T_D=T_D,**kwargs):
    step = 0.
    complete = False
    properties = tissue.properties
    mesh = tissue.mesh
    while not til_fix or not complete:
        if progress_on: 
            print_progress(step,N_steps)
            step += 1
        N=len(tissue)
        mesh.move_all(tissue.dr(dt))
        births = np.where(properties['cycle_length']<=tissue.age)[0]
        if len(births)>0:
            for mother in births:
                tissue.add_daughter_cells(mother,rand,{'cycle_length':cycle_function_uniform(2,rand)})
            tissue.remove(births,True)
            deaths = rand.randint(0,N-1,len(births))
            tissue.remove(deaths,False)
        tissue.update(dt)
        if til_fix: complete = (1 not in tissue.properties['type'] or 0 not in tissue.properties['type']) and step%stepsize==0 
        if not save_events or (len(births)!=0 or len(deaths)!=0): 
            yield tissue 

def simulation_stress_dependent(tissue,dt,N_steps,stepsize,rand,til_fix=False,progress_on=False,store_dead=False,save_events=False,T_D=T_D,stress_threshold=np.inf,N_limit=np.inf,**kwargs):
    step = 0.
    complete = False
    properties = tissue.properties
    mesh = tissue.mesh
    while not til_fix or not complete:
        if progress_on: 
            print_progress(step,N_steps)
            step += 1
        N=len(tissue)
        if N <=16 or N>=N_limit: 
            break
        mesh.move_all(tissue.dr(dt))
        births = np.where(properties['cycle_length']<=tissue.age)[0]
        births = [mother for mother in births if tissue.cell_stress(mother) < stress_threshold]
        if len(births)>0:
            for mother in births:
                daughter_properties = {'cycle_length':cycle_function_uniform(2,rand)}
                if T_D is not None: daughter_properties['age_of_death'] = death_function_poisson(2,rand,T_D=T_D)
                tissue.add_daughter_cells(mother,rand,daughter_properties)
            tissue.remove(births,True)
        if T_D is not None:        	
            deaths = np.where(properties['age_of_death']<=tissue.age)[0]
            if len(deaths)>0:
            	tissue.remove(deaths,False)
        tissue.update(dt)
        if til_fix: complete = (1 not in tissue.properties['type'] or 0 not in tissue.properties['type']) and step%stepsize==0 
        if not save_events or (len(births)!=0 or len(deaths)!=0): 
            yield tissue 
        else: yield None

def run_simulation(simulation,N,timestep,timend,rand,init_time=10.,til_fix=False,mutant_num=1,ancestors=None,save_areas=False,store_dead=False,tissue=None,force=None,save_events=False,T_D=T_D,stress_threshold=np.inf,N_limit=np.inf,**kwargs):
    if tissue is None:
        if force is None: force = BasicSpringForceNoGrowth()
        tissue = init.init_tissue_torus(N,N,0.01,force,rand,save_areas=save_areas,store_dead=store_dead)
        tissue.properties['cycle_length'] = cycle_function_uniform(N*N,rand)
        tissue.age = tissue.properties['cycle_length']*rand.rand(N*N) #initialise cell ages at random point in cell cycle
        if init_time is not None: 
            tissue = run_save_final(simulation_constant_pop_size(tissue,dt,init_time/dt,timestep/dt,rand,til_fix=False,store_dead=store_dead,T_D=T_D,**kwargs),init_time/dt)
            tissue.reset(reset_age=False)
        if mutant_num is not None:
            tissue.properties['type'] = np.zeros(N*N,dtype=int)
            tissue.properties['type'][rand.choice(N*N,size=mutant_num,replace=False)]=1
        if ancestors is not None: tissue.properties['ancestor'] = np.arange(N*N,dtype=int)
        if T_D is not None: tissue.properties['age_of_death'] = death_function_poisson(N*N,rand,T_D=T_D)+tissue.age #add age of death to initial cell age 
    if save_events: history = run_save_events(tissue, simulation(tissue,dt,timend/dt,timestep/dt,rand,til_fix=til_fix,store_dead=store_dead,save_events=save_events,T_D=T_D,stress_threshold=stress_threshold,N_limit=N_limit,**kwargs),timend/dt)
    else: history = run(tissue, simulation(tissue,dt,timend/dt,timestep/dt,rand,til_fix=til_fix,store_dead=store_dead,T_D=T_D,stress_threshold=stress_threshold,N_limit=N_limit,**kwargs),timend/dt,timestep/dt)
    return history
