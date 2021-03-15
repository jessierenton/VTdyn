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
 
def run(simulation,N_step,skip):
    """run a given simulation for N_step iterations
    returns list of tissue objects at intervals given by skip"""
    return [tissue.copy() for tissue in itertools.islice(simulation,0,N_step,skip)]

def run_generator(simulation,N_step,skip):
    """generator for running a given simulation for N_step iterations
    returns generator of tissue objects at intervals given by skip"""
    return itertools.islice(simulation,0,N_step,skip)

def run_return_events(simulation,N_step):
    """run given simulation for N_step iterations
    returns list of tissue objects containing all tissues immediately after an update event occured"""
    return [tissue.copy() for tissue in itertools.islice(simulation,N_step) if tissue is not None]

def run_return_final_tissue(simulation,N_step):
    """run given simulation for N_step iterations
    returns final tissue object"""
    return next(itertools.islice(simulation,N_step,None))

def run_til_fix(simulation,N_step,skip,include_fixed=True):
    """run a given simulation until fixation or for N_step iterations (whichever is shorter)
    returns list of tissue objects at intervals given by skip (includes final fixed tissue if include_fixed is True)"""
    return [tissue.copy() for tissue in generate_til_fix(simulation,N_step,skip,include_fixed=include_fixed)]
    
def run_til_fix_return_events(simulation,N_step,skip,include_fixed=True):
    """run a given simulation until fixation or for N_step iterations (whichever is shorter)
    returns list of tissue objects containing all tissues immediately after an update event occurred (includes final fixed tissue if include_fixed is True)"""
    return [tissue.copy() for tissue in generate_til_fix(simulation,N_step,include_fixed=include_fixed) if tissue is not None]
        
def fixed(tissue):
    """returns True if tissue has reached fixation"""
    if tissue is None:
        return False
    try:
        return (1 not in tissue.properties['type'] or 0 not in tissue.properties['type'])
    except KeyError:
        return np.all(tissue.properties['ancestor']==tissue.properties['ancestor'][0])
    

def generate_til_fix(simulation,N_step,skip=1,include_fixed=True):
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

def get_fitness(cell_type,neighbour_types,DELTA,game,game_constants):
    """calculate fitness of single cell"""
    return 1+DELTA*game(cell_type,neighbour_types,*game_constants)

def recalculate_fitnesses(neighbours_by_cell,types,DELTA,game,game_constants):
    """calculate fitnesses of all cells"""
    return np.array([get_fitness(types[cell],types[neighbours],DELTA,game,game_constants) for cell,neighbours in enumerate(neighbours_by_cell)])

# def simulation_with_mutation_ancestor_tracking(tissue,dt,N_steps,stepsize,rand,DELTA,game,constants,initial=False):
#     """simulation loop for decoupled update rule with mutation. tracks ancestors in tissue.properties['ancestor']"""
#     mutation_rate = constants[0]
#     game_constants = constants[1:]
#     step = 0.
#     complete = False
#     while initial or not complete:
#         N= len(tissue)
#         properties = tissue.properties
#         mesh = tissue.mesh
#         step += 1
#         mesh.move_all(tissue.dr(dt))
#         if rand.rand() < (1./T_D)*N*dt:
#             fitnesses = recalculate_fitnesses(tissue.mesh.neighbours,properties['type'],DELTA,game,game_constants)
#             mother = np.where(rand.multinomial(1,fitnesses/sum(fitnesses))==1)[0][0]
#             tissue.add_daughter_cells(mother,rand)
#             r = rand.rand()
#             if r < mutation_rate**2: properties['type'] = np.append(properties['type'],rand.randint(0,2,2))
#             elif r < mutation_rate: properties['type'] = np.append(properties['type'],[properties['type'][mother],rand.randint(2)])
#             else: properties['type'] = np.append(properties['type'],[properties['type'][mother]]*2)
#             properties['ancestor'] = np.append(properties['ancestor'],[properties['ancestor'][mother]]*2)
#             tissue.remove(mother)
#             tissue.remove(rand.randint(N)) #kill random cell
#         tissue.update(dt)
#         complete = (1 not in tissue.properties['type'] or 0 not in tissue.properties['type']) and step%stepsize==0
#         yield tissue
#
# def simulation_with_mutation(tissue,dt,N_steps,stepsize,rand,DELTA,game,constants,initial=False):
#     """simulation loop for decoupled update rule with mutation"""
#     mutation_rate = constants[0]
#     game_constants = constants[1:]
#     step = 0.
#     complete = False
#     while initial or not complete:
#         N= len(tissue)
#         properties = tissue.properties
#         mesh = tissue.mesh
#         step += 1
#         mesh.move_all(tissue.dr(dt))
#         if rand.rand() < (1./T_D)*N*dt:
#             fitnesses = recalculate_fitnesses(tissue.mesh.neighbours,properties['type'],DELTA,game,game_constants)
#             mother = np.where(rand.multinomial(1,fitnesses/sum(fitnesses))==1)[0][0]
#             tissue.add_daughter_cells(mother,rand)
#             r = rand.rand()
#             if r < mutation_rate**2: properties['type'] = np.append(properties['type'],rand.randint(0,2,2))
#             elif r < mutation_rate: properties['type'] = np.append(properties['type'],[properties['type'][mother],rand.randint(2)])
#             else: properties['type'] = np.append(properties['type'],[properties['type'][mother]]*2)
#             tissue.remove(mother)
#             tissue.remove(rand.randint(N)) #kill random cell
#         tissue.update(dt)
#         complete = (1 not in tissue.properties['type'] or 0 not in tissue.properties['type']) and step%stepsize==0
#         yield tissue


def simulation_decoupled_update(tissue,dt,N_steps,stepsize,rand,DELTA,game,game_constants,progress_on=False):
    """simulation loop for decoupled update rule"""
    step = 0.
    yield tissue
    while True:
        if progress_on: print_progress(step,N_steps)
        N= len(tissue)
        properties = tissue.properties
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt))
        if rand.rand() < (1./T_D)*N*dt:
            fitnesses = recalculate_fitnesses(tissue.mesh.neighbours,properties['type'],DELTA,game,game_constants)
            mother = np.where(rand.multinomial(1,fitnesses/sum(fitnesses))==1)[0][0]   
            tissue.add_daughter_cells(mother,rand)
            tissue.remove(mother)
            tissue.remove(rand.randint(N)) #kill random cell
        tissue.update(dt)
        
        yield tissue

def simulation_death_birth(tissue,dt,N_steps,stepsize,rand,DELTA,game,game_constants,progress_on=False):
    """simulation loop for death-birth update rule"""
    step = 0.
    yield tissue
    while True:
        if progress_on: print_progress(step,N_steps)
        N= len(tissue)
        properties = tissue.properties
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt))
        if rand.rand() < (1./T_D)*N*dt:
            dead_cell = rand.randint(N)
            dead_cell_neighbours = tissue.mesh.neighbours[dead_cell]
            neighbours_by_cell = [tissue.mesh.neighbours[dcn] for dcn in dead_cell_neighbours]
            fitnesses = np.array([get_fitness(tissue.properties['type'][cell],tissue.properties['type'][neighbours],DELTA,game,game_constants) for cell,neighbours in zip(dead_cell_neighbours,neighbours_by_cell)])
            mother = rand.choice(dead_cell_neighbours,p=fitnesses/sum(fitnesses))
            tissue.add_daughter_cells(mother,rand)
            tissue.remove(mother)
            tissue.remove(dead_cell) #kill random cell
        tissue.update(dt)
        yield tissue

def simulation_no_division(tissue,dt,N_steps,rand):
    """run tissue simulation with no death or division"""
    step = 0.
    while True:
        N= len(tissue)
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt))
        tissue.update(dt)
        yield tissue
        
# def simulation_death_birth_radius(tissue,dt,N_steps,stepsize,rand,DELTA,game,game_constants,division_radius,initial=False):
#     """simulation loop for death-birth update rule with interactions occurring within a given radius rather than between neighbours"""
#     step = 0.
#     complete = False
#     while initial or not complete:
#         N= len(tissue)
#         properties = tissue.properties
#         mesh = tissue.mesh
#         step += 1
#         mesh.move_all(tissue.dr(dt))
#         if rand.rand() < (1./T_D)*N*dt:
#             dead_cell = rand.randint(N)
#             cell_in_range = np.where((mesh.get_distances(dead_cell)<division_radius))[0]
#             cell_in_range = cell_in_range[cell_in_range!=dead_cell]
#             neighbours_by_cell = [tissue.mesh.neighbours[cell] for cell in cell_in_range]
#             fitnesses = np.array([get_fitness(tissue.properties['type'][cell],tissue.properties['type'][neighbours],DELTA,game,game_constants)
#                                 for cell,neighbours in zip(cell_in_range,neighbours_by_cell)])
#             mother = rand.choice(cell_in_range,p=fitnesses/sum(fitnesses))
#             tissue.add_daughter_cells(mother,rand)
#             properties['type'] = np.append(properties['type'],[properties['type'][mother]]*2)
#             tissue.remove(mother)
#             tissue.remove(dead_cell) #kill random cell
#         tissue.update(dt)
#         complete = (1 not in tissue.properties['type'] or 0 not in tissue.properties['type']) and step%stepsize==0
#         yield tissue

    
def run_simulation(simulation,N,timestep,timend,rand,DELTA,game,constants,init_time=None,til_fix=True,save_areas=False,
                    tissue=None,mutant_num=1,save_cell_histories=False,progress_on=False):
    """initialise tissue with NxN cells and run given simulation with given game and constants.
            starts with single cooperator
            ends at time=timend OR if til_fix=True when population all cooperators (type=1) or defectors (2)
        returns history: list of tissue objects at time intervals given by timestep
            """
    if tissue is None:
        tissue = init.init_tissue_torus(N,N,0.01,BasicSpringForceNoGrowth(),rand,save_areas=save_areas,save_cell_histories=save_cell_histories)
    tissue.properties['type'] = np.zeros(N*N,dtype=int)
    tissue.age = np.zeros(N*N,dtype=float)
    if init_time is not None:    
        tissue = run_return_final_tissue(simulation(tissue,dt,init_time/dt,timestep/dt,rand,DELTA,game,constants),init_time/dt)
        tissue.reset()
    tissue.properties['ancestors']= np.arange(100,dtype=int)
    tissue.properties['type'][rand.choice(N*N,size=mutant_num,replace=False)]=1
    if til_fix:
        history = run_til_fix(simulation(tissue,dt,timend/dt,timestep/dt,rand,DELTA,game,constants,progress_on=progress_on),timend/dt,timestep/dt)
    else:
        history = run(simulation(tissue,dt,timend/dt,timestep/dt,rand,DELTA,game,constants,progress_on=progress_on),timend/dt,timestep/dt)
    return history
