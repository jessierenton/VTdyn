import sys
import numpy as np
import itertools
import structure
from structure.global_constants import T_D,dt,ETA,MU
from structure.cell import Tissue, BasicSpringForceNoGrowth
import structure.initialisation as init

def print_progress(step,N_steps):
    sys.stdout.write("\r %.2f %%"%(step*100./N_steps))
    sys.stdout.flush() 

def run(simulation,N_step,skip):
    """run a given simulation for N_step iterations
    returns list of tissue objects at intervals given by skip"""
    return [tissue.copy() for tissue in itertools.islice(simulation,0,N_step,skip)]

def run_generator(simulation,N_step,skip):
    """generator for running a given simulation for N_step iterations
    returns generator for of tissue objects at intervals given by skip"""
    return itertools.islice(simulation,0,N_step,skip)

def run_return_events(simulation,N_step):
    return [tissue.copy() for tissue in itertools.islice(simulation,N_step) if tissue is not None]

def run_return_final_tissue(simulation,N_step):
    return next(itertools.islice(simulation,N_step,None))

def run_til_fix(simulation,N_step,skip,include_fixed=True):
    return [tissue.copy() for tissue in generate_til_fix(simulation,N_step,skip,include_fixed=include_fixed)]
        
def fixed(tissue):
    try:
        return (1 not in tissue.properties['type'] or 0 not in tissue.properties['type'])
    except KeyError:
        return np.all(tissue.properties['ancestor']==tissue.properties['ancestor'][0])
    

def generate_til_fix(simulation,N_step,skip,include_fixed=True):
    for tissue in itertools.islice(simulation,0,N_step,skip):
        if not fixed(tissue):
            yield tissue
        else:
            if include_fixed:
                yield tissue
            break

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------POISSON-CONSTANT-POP-SIZE-AND-FITNESS------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def prisoners_dilemma_averaged(cell_type,neighbour_types,b,c):
    """calculate average payoff for single cell"""
    return -c*cell_type+b*np.sum(neighbour_types)/len(neighbour_types)

def prisoners_dilemma_accumulated(cell_type,neighbour_types,b,c):
    """calculate accumulated payoff for single cell"""
    return -c*cell_type*len(neighbour_types)+b*np.sum(neighbour_types)

def N_person_prisoners_dilemma(cell_type,neighbour_types,b,c):
    return -c*cell_type + b*(np.sum(neighbour_types)+cell_type)/(len(neighbour_types)+1)

def volunteers_dilemma(cell_type,neighbour_types,b,c,M):
    return -c*cell_type +b*((np.sum(neighbour_types)+cell_type)>=M)

def benefit_function_game(cell_type,neighbour_types,benefit_function,benefit_function_params,b,c):
    return -c*cell_type + b*benefit_function(np.sum(neighbour_types)+cell_type,len(neighbour_types)+1,*benefit_function_params)

def logistic_benefit(j,Ng,s,k):
    return (logistic_function(j,Ng,s,k)-logistic_function(0,Ng,s,k))/(logistic_function(Ng,Ng,s,k)-logistic_function(0,Ng,s,k))

def logistic_function(j,Ng,s,k):
    return 1./(1.+np.exp(s*(k-j)/Ng))

def get_fitness(cell_type,neighbour_types,DELTA,game,game_constants):
    """calculate fitness of single cell"""
    return 1+DELTA*game(cell_type,neighbour_types,*game_constants)

def recalculate_fitnesses(neighbours_by_cell,types,DELTA,game,game_constants):
    """calculate fitnesses of all cells"""
    return np.array([get_fitness(types[cell],types[neighbours],DELTA,game,game_constants) 
                        for cell,neighbours in enumerate(neighbours_by_cell)])

def choose_birth_and_death(tissue,rand,DELTA,game,game_constants,update):
    dead_cell = rand.randint(len(tissue))
    if update == 'death_birth':
        parent = choose_parent_death_birth(tissue,rand,DELTA,game,game_constants,update,dead_cell)
    elif update == 'decoupled':
        parent = choose_parent_decoupled(tissue,rand,DELTA,game,game_constants)
    return parent,dead_cell

def choose_parent_death_birth(tissue,rand,DELTA,game,game_constants,dead_cell):
    dead_cell_neighbours = tissue.mesh.neighbours[dead_cell]
    if game is None:
        return rand.choice(dead_cell_neighbours)
    else:
        neighbours_by_cell = [tissue.mesh.neighbours[dcn] for dcn in dead_cell_neighbours]
        fitnesses = np.array([get_fitness(tissue.properties['type'][cell],tissue.properties['type'][neighbours],DELTA,game,game_constants) 
                            for cell,neighbours in zip(dead_cell_neighbours,neighbours_by_cell)])
        return rand.choice(dead_cell_neighbours,p=fitnesses/sum(fitnesses))

def choose_parent_decoupled(tissue,rand,DELTA,game,game_constants):
    if game is None:
        return rand.randint(len(tissue))
    else:
        fitnesses = recalculate_fitnesses(tissue.mesh.neighbours,tissue.properties['type'],DELTA,game,game_constants)
        return np.where(rand.multinomial(1,fitnesses/sum(fitnesses))==1)[0][0]

def _simulation(tissue,dt,N_steps,stepsize,rand,DELTA,game,game_constants,update,eta=ETA,progress_on=False):
    step = 0.
    yield tissue
    while True:
        if progress_on: print_progress(step,N_steps)
        N= len(tissue)
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt))
        if rand.rand() < (1./T_D)*N*dt:
            parent,dead_cell = choose_birth_and_death(tissue,rand,DELTA,game,game_constants,update)
            tissue.add_daughter_cells(parent,rand)
            tissue.remove(parent)
            tissue.remove(dead_cell) #kill random cell
        tissue.update(dt)
        yield tissue
        
def simulation_decoupled_update(tissue,dt,N_steps,stepsize,rand,DELTA,game,game_constants,eta=ETA,progress_on=False):
    """simulation loop for decoupled update rule"""
    update = 'decoupled'
    return _simulation(tissue,dt,N_steps,stepsize,rand,DELTA,game,game_constants,update,eta,progress_on)

def simulation_death_birth(tissue,dt,N_steps,stepsize,rand,DELTA,game,game_constants,eta=ETA,progress_on=False):
    """simulation loop for death-birth update rule"""
    update = 'death_birth'
    return _simulation(tissue,dt,N_steps,stepsize,rand,DELTA,game,game_constants,update,eta,progress_on)

def simulation_no_division(tissue,dt,N_steps,rand,eta=ETA):
    """run tissue simulation with no death or division"""
    step = 0.
    while True:
        N= len(tissue)
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt,eta))
        tissue.update(dt)
        yield tissue

def initialise_tissue(simulation,N,dt,timend,timestep,rand,mu=MU,save_areas=False,save_cell_histories=False):  
    """initialise tissue and run simulation until timend returning final state"""              
    tissue = init.init_tissue_torus(N,N,0.01,BasicSpringForceNoGrowth(mu),rand,save_areas=save_areas,save_cell_histories=save_cell_histories)
    tissue.age = np.zeros(N*N,dtype=float)
    if timend !=0: tissue = run_return_final_tissue(simulation(tissue,dt,timend/dt,timestep/dt,rand,None,None,None,eta=ETA),timend/dt)
    tissue.time=0.
    return tissue

def run_simulation(simulation,N,timestep,timend,rand,DELTA,game,game_constants,init_time=None,mu=MU,eta=ETA,dt=dt,til_fix=True,generator=False,save_areas=False,
                tissue=None,mutant_num=1,save_cell_histories=False,progress_on=False,**kwargs):
    """initialise tissue with NxN cells and run given simulation with given game and constants.
            starts with single cooperator
            ends at time=timend OR if til_fix=True when population all cooperators (type=1) or defectors (2)
        returns history: list of tissue objects at time intervals given by timestep
            """
    if tissue is None:
        tissue = initialise_tissue(simulation,N,dt,init_time,timestep,rand,mu=mu,save_areas=save_areas,save_cell_histories=save_cell_histories)
    if mutant_num > 0:
        tissue.properties['type']=np.zeros(N*N,dtype=int)
        tissue.properties['type'][rand.choice(N*N,size=mutant_num,replace=False)]=1
    else:
        tissue.properties['ancestor']=np.arange(N*N)
    if til_fix:
        include_fix = not (til_fix=='exclude_final')
        if generator:
            history = generate_til_fix(simulation(tissue,dt,timend/dt,timestep/dt,rand,DELTA,game,game_constants,eta=eta,progress_on=progress_on,**kwargs),timend/dt,timestep/dt,include_fix)
        else:
            history = run_til_fix(simulation(tissue,dt,timend/dt,timestep/dt,rand,DELTA,game,game_constants,eta=eta,progress_on=progress_on,**kwargs),timend/dt,timestep/dt)
    else:
        history = run(simulation(tissue,dt,timend/dt,timestep/dt,rand,DELTA,game,game_constants,eta=eta,progress_on=progress_on,**kwargs),timend/dt,timestep/dt)
    return history