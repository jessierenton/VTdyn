import sys
import numpy as np
import itertools
import structure
from structure.global_constants import T_D,dt
from structure.cell import Tissue, BasicSpringForceNoGrowth
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
        update_progress(step/N_steps)  
        yield tissue
        
def run(tissue_original,simulation,N_step,skip):
    return [tissue_original.copy()]+[tissue.copy() for tissue in itertools.islice(simulation,skip-1,N_step,skip)]

def run_save_events(tissue_original,simulation,N_step):
    return [tissue_original.copy()]+[tissue.copy() for tissue in itertools.islice(simulation,N_step) if tissue is not None]

def run_save_final(simulation,N_step):
    return next(itertools.islice(simulation,N_step,None))
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------POISSON-CONSTANT-POP-SIZE-AND-FITNESS------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def prisoners_dilemma_averaged(cell_type,neighbour_types,b,c):
    return -c*cell_type+b*np.sum(neighbour_types)/len(neighbour_types)

def prisoners_dilemma_accumulated(cell_type,neighbour_types,b,c):
    return -c*cell_type*len(neighbour_types)+b*np.sum(neighbour_types)

def get_fitness(cell_type,neighbour_types,DELTA,game,game_params):
    return 1+DELTA*game(cell_type,neighbour_types,*game_params)

def recalculate_fitnesses(neighbours_by_cell,types,DELTA,game,game_params):
    return 1+DELTA*np.array([game(types[cell],types[neighbours],*game_params) for cell,neighbours in enumerate(neighbours_by_cell)])
    
def simple_fitness(types,DELTA,r):
    return 1+DELTA*r*types

def linear_density_dep(densities,gradient,critical_density,y_int):
    return gradient*densities/critical_density+y_int

def step_density_dep(densities,A,critical_density,invert):
    if invert: return 2*A*(densities<critical_density)
    else: return 2*A*(densities>critical_density)
    
def no_density_dep(densities,A):
    return A*np.ones(len(densities))

def simulation_local_density_dep(tissue,dt,N_steps,stepsize,rand,til_fix=False,progress_on=False,store_dead=False,save_events=False,**kwargs):
    step = 0.
    complete = False
    properties = tissue.properties
    mesh = tissue.mesh
    DELTA = kwargs['DELTA']
    game = kwargs['game']
    game_params = kwargs['game_params']
    birth_dd_func,death_dd_func = kwargs['birth_dd_func'],kwargs['death_dd_func']
    birth_dd_params,death_dd_params = kwargs['birth_dd_params'],kwargs['death_dd_params']
    death_rate = (1./T_D)
    try: birth_rate = kwargs['birth_to_death_rate_ratio']*death_rate
    except: birth_rate = death_rate
    while not til_fix or not complete:
        if progress_on: 
            print_progress(step,N_steps)
            step += 1
        N= len(tissue)
        mesh.move_all(tissue.dr(dt))
        if game == 'simple': fitnesses = simple_fitness(properties['type'],DELTA,game_params)
        else: fitnesses = recalculate_fitnesses(mesh.neighbours,properties['type'],DELTA,game,game_params)
        densities = mesh.local_density()
        births = np.where(rand.rand(N) < fitnesses*birth_rate*birth_dd_func(densities,*birth_dd_params)*dt)[0]
        deaths = np.where(rand.rand(N) < death_rate*death_dd_func(densities,*death_dd_params)*dt)[0]
        if len(births)!=0 or len(deaths)!=0:
            births_and_deaths(tissue,births,deaths,rand,store_dead)      
        tissue.update(dt)
        if til_fix: complete = (1 not in tissue.properties['type'] or 0 not in tissue.properties['type']) and step%stepsize==0 
        if not save_events or (len(births)!=0 or len(deaths)!=0): yield tissue 
        else: yield None       

def births_and_deaths(tissue,births,deaths,rand,store_dead=False):
    if len(births)>0: tissue.add_many_daughter_cells(births,rand)
    if store_dead: mother_bool = np.array([True]*len(births)+[False]*len(deaths))
    else: mother_bool=None
    tissue.remove(np.append(births,deaths),mother_bool)

# def births_and_deaths(tissue,births,deaths,rand):
#     if len(births)==1 and len(deaths)==0: #if only a birth
#         single_birth(tissue,births[0],rand)
#     elif len(births)==0 and len(deaths)==1: #if only a death
#         single_death(tissue,deaths[0])
#     elif len(births)==len(deaths)==1: #if one birth and one death
#         if births[0]<deaths[0]:
#             single_death(tissue,deaths[0])
#             single_birth(tissue,births[0],rand)
#         elif births[0]>deaths[0]:
#             single_birth(tissue,births[0],rand)
#             single_death(tissue,deaths[0])
#     else:
#         multiple_births_and_deaths(tissue,births,deaths,rand)
#
# def single_birth(tissue,mother,rand):
#     tissue.add_daughter_cells(mother,rand)
#     tissue.properties['type'] = np.append(tissue.properties['type'],[tissue.properties['type'][mother]]*2)
#     tissue.remove(mother)
#
# def single_death(tissue,dead):
#     tissue.remove(dead)
#
# def multiple_births_and_deaths(tissue,births,deaths,rand):
#     order = [cell for cell in np.append(births,deaths) if cell not in set(births).intersection(deaths)]
#     order.sort(reverse=True)
#     for cell in order:
#         if cell in births:
#             single_birth(tissue,cell,rand)
#         else:
#             single_death(tissue,cell)
    
    
def run_simulation(simulation,N,timestep,timend,rand,init_time=10.,til_fix=False,mutant_num=1,save_areas=False,store_dead=False,tissue=None,force=None,save_events=False,**kwargs):
    if tissue is None:
        if force is None: force = BasicSpringForceNoGrowth()
        tissue = init.init_tissue_torus(N,N,0.01,force,rand,save_areas=save_areas,store_dead=store_dead)
        tissue.properties['type'] = np.zeros(N*N,dtype=int)
        tissue.age = np.zeros(N*N,dtype=float)
        if init_time is not None: 
            tissue = run_save_final(tissue, simulation(tissue,dt,init_time/dt,timestep/dt,rand,til_fix=False,store_dead=store_dead,**kwargs),init_time/dt)
            tissue.reset()
        tissue.properties['type'][rand.choice(N*N,size=mutant_num,replace=False)]=1
    if save_events: history = run_save_events(tissue, simulation(tissue,dt,timend/dt,timestep/dt,rand,til_fix=til_fix,store_dead=store_dead,save_events=save_events,**kwargs),timend/dt)
    else: history = run(tissue, simulation(tissue,dt,timend/dt,timestep/dt,rand,til_fix=til_fix,store_dead=store_dead,**kwargs),timend/dt,timestep/dt)
    return history