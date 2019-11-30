import sys
import numpy as np
import itertools
import structure
from structure.global_constants import *
from structure.cell import Tissue, BasicSpringForceNoGrowth, MutantSpringForce
import structure.initialisation as init
from structure.global_constants import MU,T_M,ETA

def print_progress(step,N_steps):
    sys.stdout.write("\r %.2f %%"%(step*100/N_steps))
    sys.stdout.flush() 

def simulation_no_division(tissue,dt,N_steps,rand,eta=ETA):
    step = 0.
    while True:
        N= len(tissue)
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt,eta))
        tissue.update(dt)
        print_progress(step,N_steps)  
        yield tissue

def run(simulation,N_step,skip):
    return [tissue.copy() for tissue in itertools.islice(simulation,0,N_step,skip)]

def run_return_events(simulation,N_step):
    return [tissue.copy() for tissue in itertools.islice(simulation,N_step) if tissue is not None]

def run_return_final_tissue(simulation,N_step):
    return next(itertools.islice(simulation,N_step,None))

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------PRISONER'S-DILEMMA----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def prisoners_dilemma_averaged(cell_type,neighbour_types,b,c):
    """calculate average payoff for single cell"""
    return -c*cell_type+b*np.sum(neighbour_types)/len(neighbour_types)

def prisoners_dilemma_accumulated(cell_type,neighbour_types,b,c):
    """calculate accumulated payoff for single cell"""
    return -c*cell_type*len(neighbour_types)+b*np.sum(neighbour_types)

def get_fitness(cell_type,neighbour_types,DELTA,game,game_constants):
    """calculate fitness of single cell"""
    return 1+DELTA*game(cell_type,neighbour_types,*game_constants)

def recalculate_fitnesses(neighbours_by_cell,types,DELTA,game,game_constants):
    """calculate fitnesses of all cells"""
    return np.array([get_fitness(types[cell],types[neighbours],DELTA,game,game_constants) for cell,neighbours in enumerate(neighbours_by_cell)])

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------POISSON-BIRTH-DEATH----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def step_function(val,threshold):
    return np.heaviside(val-threshold,1)

def G_to_S_transition(properties,age,tension_area_product,G_to_S_rate,dt,CIP_function,CIP_parameters,rand):
    cycle_phases = properties['cycle_phase']
    num_G_cells = sum(1-cycle_phases)
    energies = np.array([tension_area_product(i) if phase==0 else np.inf
                            for i,phase in enumerate(cycle_phases)])
    transitions = rand.rand(len(energies))<G_to_S_rate*dt*CIP_function(energies,**CIP_parameters)
    if not np.any(transitions):
        return False
    else:
        cycle_phases[transitions]=1
        properties['transition_age'][transitions]=age[transitions]        
        return True

def simulation_contact_inhibition_energy_checkpoint_2_stage(tissue,dt,N_steps,stepsize,rand,rates,CIP_parameters=None,CIP_function=None,til_fix=False,progress_on=False,return_events=False,stress_threshold=np.inf,N_limit=np.inf,**kwargs):
    yield tissue # start with initial tissue 
    step = 1.
    complete = False
    properties = tissue.properties
    mesh = tissue.mesh
    death_rate,G_to_S_rate,S_to_div_rate = rates
    if CIP_function is None: 
        CIP_function = step_function
    while not til_fix or not complete:
        event_occurred = False
        if progress_on: 
            print_progress(step,N_steps)
            step += 1
        N=len(tissue)
        if N <=16 or N>=N_limit: 
            break
        mesh.move_all(tissue.dr(dt))
        event_occurred = G_to_S_transition(properties,tissue.age,tissue.tension_area_product,G_to_S_rate,dt,CIP_function,CIP_parameters,rand)
        #cell division
        num_S_cells = sum(properties['cycle_phase'])
        if rand.rand() < num_S_cells*S_to_div_rate*dt:
            event_occurred = True
            mother = rand.choice(N,p=properties['cycle_phase']/float(num_S_cells))
            tissue.add_daughter_cells(mother,rand,{'cycle_phase':(0,0),'transition_age':(-1,-1)})
            tissue.remove(mother,True)
        #cell_death
        N = len(tissue)
        if death_rate is not None:  
            if rand.rand() < N*death_rate*dt:
                tissue.remove(rand.randint(N),False)   
                event_occurred = True   	
        tissue.update(dt)
        if til_fix: 
            complete = (1 not in tissue.properties['type'] or 0 not in tissue.properties['type']) and step%stepsize==0
        if not return_events or event_occurred: 
            yield tissue
        else: yield

def check_area_threshold(mesh,threshold_area_fraction):
    return np.where(mesh.areas > threshold_area_fraction*A0)[0]

def check_separation_threshold(mesh,threshold_separation_fraction):
    return np.where(mesh.areas > threshold_area_fraction*A0)[0]
    
def simulation_contact_inhibition_area_dependent(tissue,dt,N_steps,stepsize,rand,rates,threshold_area_fraction=0.,til_fix=False,progress_on=False,return_events=False,N_limit=np.inf,eta=ETA,DELTA=None,game=None,game_constants=None,**kwargs):
    yield tissue # start with initial tissue 
    step = 0.
    complete = False
    properties = tissue.properties
    mesh = tissue.mesh
    death_rate,division_rate = rates
    while not til_fix or not complete:
        event_occurred = False
        if progress_on: 
            print_progress(step,N_steps)
            step += 1
        N=len(tissue)
        if N <=16 or N>=N_limit: 
            break
        mesh.move_all(tissue.dr(dt,eta))
        #cell division     
        division_ready = check_area_threshold(mesh,threshold_area_fraction)
        if rand.rand() < len(division_ready)*division_rate*dt:
            if game is None:
                mother = rand.choice(division_ready)
            else:
                fitnesses = np.array([get_fitness(properties['type'][cell],properties['type'][mesh.neighbours[cell]],DELTA,game,game_constants) 
                                for cell in division_ready])
                mother = rand.choice(division_ready,p=fitnesses/sum(fitnesses))
            tissue.add_daughter_cells(mother,rand)
            tissue.remove(mother,True)
            event_occurred = True  
        #cell_death
        N = len(tissue)
        if death_rate is not None:  
            if rand.rand() < N*death_rate*dt:
                tissue.remove(rand.randint(N),False)   
                event_occurred = True   	
        tissue.update(dt)
        if til_fix: 
            complete = (1 not in tissue.properties['type'] or 0 not in tissue.properties['type']) and step%stepsize==0
        if not return_events or event_occurred: 
            yield tissue
        else: yield

def run_simulation(simulation,N,timestep,timend,rand,init_time=10.,til_fix=False,progress_on=False,mutant_num=1,ancestors=None,mu=MU,T_m=T_M,eta=ETA,dt=dt,DELTA=None,game=None,game_constants=None,
        cycle_phase=None,save_areas=False,save_cell_histories=False,tissue=None,force=None,return_events=False,N_limit=np.inf,domain_size_multiplier=1.,**kwargs):
    if tissue is None:
        if force is None: force = BasicSpringForceNoGrowth(mu,T_m)
        tissue = init.init_tissue_torus_with_multiplier(N,N,0.01,force,rand,domain_size_multiplier,save_areas=save_areas,save_cell_histories=save_cell_histories)
        if cycle_phase is not None:
            tissue.properties["cycle_phase"] = np.zeros(N*N,dtype=int)
            tissue.properties["transition_age"] = -np.ones(N*N,dtype=float)
        if init_time is not None: 
            tissue = run_return_final_tissue(simulation(tissue,dt,init_time/dt,timestep/dt,rand,til_fix=False,eta=ETA,progress_on=progress_on,**kwargs),init_time/dt)
            tissue.reset(reset_age=False)
        if mutant_num is not None:
            tissue.properties['type'] = np.zeros(len(tissue),dtype=int)
            tissue.properties['type'][rand.choice(len(tissue),size=mutant_num,replace=False)]=1
        if ancestors is not None: tissue.properties['ancestor'] = np.arange(len(tissue),dtype=int)
    if return_events: history = run_return_events(simulation(tissue,dt,timend/dt,timestep/dt,rand,til_fix=til_fix,progress_on=progress_on,return_events=return_events,N_limit=N_limit,DELTA=DELTA,game=game,game_constants=game_constants,**kwargs),timend/dt)
    else: history = run(simulation(tissue,dt,timend/dt,timestep/dt,rand,til_fix=til_fix,progress_on=progress_on,return_events=return_events,N_limit=N_limit,eta=ETA,DELTA=DELTA,game=game,game_constants=game_constants,**kwargs),timend/dt,timestep/dt)
    return history
