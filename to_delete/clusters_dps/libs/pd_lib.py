import sys
import numpy as np
import itertools
import structure
from structure.global_constants import *
from structure.cell import Tissue, BasicSpringForceNoGrowth, MutantSpringForce
import structure.initialisation as init


def update_progress(progress):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1:.4f}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
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

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------POISSON-CONSTANT-POP-SIZE-AND-FITNESS------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def prisoners_dilemma_averaged(cell_type,neighbour_types,b,c):
    return -c*cell_type+b*np.sum(neighbour_types)/len(neighbour_types)

def prisoners_dilemma_accumulated(cell_type,neighbour_types,b,c):
    return -c*cell_type*len(neighbour_types)+b*np.sum(neighbour_types)

def get_fitness(cell_type,neighbour_types,DELTA,game,game_constants):
    return 1+DELTA*game(cell_type,neighbour_types,*game_constants)

def recalculate_fitnesses(neighbours_by_cell,types,DELTA,game,game_constants):
    return np.array([get_fitness(types[cell],types[neighbours],DELTA,game,game_constants) for cell,neighbours in enumerate(neighbours_by_cell)])


def simulation_initialise_tissue_with_cluster(tissue,dt,min_steps,rand,cluster_size,main_type):
    step = 0.
    N= len(tissue)
    tissue.properties['ancestor']=np.arange(N,dtype=int)
    max_cluster_size = 1
    max_cluster_index = 0
    while True:
        properties = tissue.properties
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt))
        if rand.rand() < (1./T_D)*N*dt:
            mother = rand.randint(N)
            tissue.add_daughter_cells(mother,rand)
            properties['ancestor'] = np.append(properties['ancestor'],[properties['ancestor'][mother]]*2)
            tissue.remove(mother)
            tissue.remove(rand.randint(N)) #kill random cell
        tissue.update(dt)
        # update_progress(step/N_steps)
        if step > min_steps:
            if cluster_size == 1: 
                tissue.properties['type'] = np.full(N,main_type,dtype=int)
                tissue.properties['type'][rand.randint(N)] = 1-main_type
                return tissue
            cluster_index = np.where(np.bincount(properties['ancestor'])==cluster_size)[0]
            if len(cluster_index)!=0:
                tissue.properties['type'] = np.full(N,main_type,dtype=int)
                tissue.properties['type'][np.where(tissue.properties.pop('ancestor')==rand.choice(cluster_index))[0]] = 1-main_type
                return tissue

def simulation_with_mutation_ancestor_tracking(tissue,dt,N_steps,stepsize,rand,DELTA,game,constants,initial=False):
    mutation_rate = constants[0]
    game_constants = constants[1:]
    step = 0.
    complete = False
    while initial or not complete:
        N= len(tissue)
        properties = tissue.properties
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt))
        if rand.rand() < (1./T_D)*N*dt:
            fitnesses = recalculate_fitnesses(tissue.mesh.neighbours,properties['type'],DELTA,game,game_constants)
            mother = np.where(np.random.multinomial(1,fitnesses/sum(fitnesses))==1)[0][0]   
            tissue.add_daughter_cells(mother,rand)
            r = rand.rand()
            if r < mutation_rate**2: properties['type'] = np.append(properties['type'],rand.randint(0,2,2))
            elif r < mutation_rate: properties['type'] = np.append(properties['type'],[properties['type'][mother],rand.randint(2)])
            else: properties['type'] = np.append(properties['type'],[properties['type'][mother]]*2)
            properties['ancestor'] = np.append(properties['ancestor'],[properties['ancestor'][mother]]*2)
            tissue.remove(mother)
            tissue.remove(rand.randint(N)) #kill random cell
        tissue.update(dt)
        update_progress(step/N_steps)
        complete = (1 not in tissue.properties['type'] or 0 not in tissue.properties['type']) and step%stepsize==0  
        yield tissue

def simulation_with_mutation(tissue,dt,N_steps,stepsize,rand,DELTA,game,constants,initial=False):
    mutation_rate = constants[0]
    game_constants = constants[1:]
    step = 0.
    complete = False
    while initial or not complete:
        N= len(tissue)
        properties = tissue.properties
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt))
        if rand.rand() < (1./T_D)*N*dt:
            fitnesses = recalculate_fitnesses(tissue.mesh.neighbours,properties['type'],DELTA,game,game_constants)
            mother = np.where(np.random.multinomial(1,fitnesses/sum(fitnesses))==1)[0][0]   
            tissue.add_daughter_cells(mother,rand)
            r = rand.rand()
            if r < mutation_rate**2: properties['type'] = np.append(properties['type'],rand.randint(0,2,2))
            elif r < mutation_rate: properties['type'] = np.append(properties['type'],[properties['type'][mother],rand.randint(2)])
            else: properties['type'] = np.append(properties['type'],[properties['type'][mother]]*2)
            tissue.remove(mother)
            tissue.remove(rand.randint(N)) #kill random cell
        tissue.update(dt)
        update_progress(step/N_steps)
        complete = (1 not in tissue.properties['type'] or 0 not in tissue.properties['type']) and step%stepsize==0  
        yield tissue


def simulation_decoupled_update(tissue,dt,N_steps,stepsize,rand,DELTA,game,game_constants,initial=False):
    step = 0.
    complete = False
    while initial or not complete:
        N= len(tissue)
        properties = tissue.properties
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt))
        if rand.rand() < (1./T_D)*N*dt:
            fitnesses = recalculate_fitnesses(tissue.mesh.neighbours,properties['type'],DELTA,game,game_constants)
            mother = np.where(np.random.multinomial(1,fitnesses/sum(fitnesses))==1)[0][0]   
            tissue.add_daughter_cells(mother,rand)
            properties['type'] = np.append(properties['type'],[properties['type'][mother]]*2)
            tissue.remove(mother)
            tissue.remove(rand.randint(N)) #kill random cell
        tissue.update(dt)
        # update_progress(step/N_steps)
        complete = (1 not in tissue.properties['type'] or 0 not in tissue.properties['type']) and step%stepsize==0  
        yield tissue

def simulation_death_birth(tissue,dt,N_steps,stepsize,rand,DELTA,game,game_constants,initial=False):
    step = 0.
    complete = False
    while initial or not complete:
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
            properties['type'] = np.append(properties['type'],[properties['type'][mother]]*2)
            tissue.remove(mother)
            tissue.remove(dead_cell) #kill random cell
        tissue.update(dt)
        # update_progress(step/N_steps)
        complete = (1 not in tissue.properties['type'] or 0 not in tissue.properties['type']) and step%stepsize==0  
        yield tissue
        
def simulation_death_birth_radius(tissue,dt,N_steps,stepsize,rand,DELTA,game,game_constants,division_radius,initial=False):        
    step = 0.
    complete = False
    while initial or not complete:
        N= len(tissue)
        properties = tissue.properties
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt))
        if rand.rand() < (1./T_D)*N*dt:
            dead_cell = rand.randint(N)
            cell_in_range = np.where((mesh.get_distances(dead_cell)<division_radius))[0]
            cell_in_range = cell_in_range[cell_in_range!=dead_cell]
            neighbours_by_cell = [tissue.mesh.neighbours[cell] for cell in cell_in_range]
            fitnesses = np.array([get_fitness(tissue.properties['type'][cell],tissue.properties['type'][neighbours],DELTA,game,game_constants) 
                                for cell,neighbours in zip(cell_in_range,neighbours_by_cell)])
            mother = rand.choice(cell_in_range,p=fitnesses/sum(fitnesses))
            tissue.add_daughter_cells(mother,rand)
            properties['type'] = np.append(properties['type'],[properties['type'][mother]]*2)
            tissue.remove(mother)
            tissue.remove(dead_cell) #kill random cell
        tissue.update(dt)
        # update_progress(step/N_steps)
        complete = (1 not in tissue.properties['type'] or 0 not in tissue.properties['type']) and step%stepsize==0  
        yield tissue

def initialise_tissue_ancestors(N,dt,timend,timestep,rand,DELTA,game,constants):                
    tissue = init.init_tissue_torus(N,N,0.01,BasicSpringForceNoGrowth(),rand,save_areas=False)
    tissue.properties['type'] = np.zeros(N*N,dtype=int)
    tissue.properties['ancestor'] = np.arange(N*N)
    tissue.age = np.zeros(N*N,dtype=float)
    tissue = run(tissue,simulation_with_mutation_ancestor_tracking(tissue,dt,timend/dt,timestep/dt,rand,DELTA,game,constants,True),timend/dt,timestep/dt)[-1]
    return tissue
    
def run_simulation(simulation,N,timestep,timend,rand,DELTA,game,constants,til_fix=True,save_areas=False,tissue=None):
    """prisoners dilemma with decoupled birth and death"""
    if tissue is None:
        tissue = init.init_tissue_torus(N,N,0.01,BasicSpringForceNoGrowth(),rand,save_areas=False)
        tissue.properties['type'] = np.zeros(N*N,dtype=int)
        tissue.age = np.zeros(N*N,dtype=float)
        tissue = run(tissue, simulation(tissue,dt,10./dt,timestep/dt,rand,DELTA,game,constants,True),10./dt,timestep/dt)[-1]
        tissue.properties['type'][rand.randint(N*N,size=1)]=1
    history = run(tissue, simulation(tissue,dt,timend/dt,timestep/dt,rand,DELTA,game,constants,~til_fix),timend/dt,timestep/dt)
    return history


def run_simulation_decoupled_update(N,timestep,timend,rand,DELTA,game,game_constants,save_areas=False):
    """prisoners dilemma with decoupled birth and death"""
    tissue = init.init_tissue_torus(N,N,0.01,BasicSpringForceNoGrowth(),rand,save_areas=False)
    tissue.properties['type'] = np.zeros(N*N,dtype=int)
    tissue.age = np.zeros(N*N,dtype=float)
    tissue = run(tissue, simulation_decoupled_update(tissue,dt,10./dt,timestep/dt,rand,DELTA,game,game_constants,True),10./dt,timestep/dt)[-1]
    tissue.properties['type'][rand.randint(N*N,size=1)]=1
    history = run(tissue, simulation_decoupled_update(tissue,dt,timend/dt,timestep/dt,rand,DELTA,game,game_constants),timend/dt,timestep/dt)
    return history

def run_simulation_death_birth(N,timestep,timend,rand,DELTA,game,game_constants,save_areas=False):
    """prisoners dilemma with death-birth update rule"""
    tissue = init.init_tissue_torus(N,N,0.01,BasicSpringForceNoGrowth(),rand,save_areas=False)
    tissue.properties['type'] = np.zeros(N*N,dtype=int)
    tissue.age = np.zeros(N*N,dtype=float)
    tissue = run(tissue, simulation_poisson_death_birth(tissue,dt,10./dt,timestep/dt,rand,DELTA,game,game_constants,True),10./dt,timestep/dt)[-1]
    tissue.properties['type'][rand.randint(N*N,size=1)]=1
    history = run(tissue, simulation_death_birth(tissue,dt,timend/dt,timestep/dt,rand,DELTA,game,game_constants),timend/dt,timestep/dt)
    return history
    
def run_simulation_death_birth_radius(N,timestep,timend,rand,DELTA,game,game_constants,division_radius,save_areas=False):
    """prisoners dilemma with birth events restricted to a given radius of death"""
    tissue = init.init_tissue_torus(N,N,0.01,BasicSpringForceNoGrowth(),rand,save_areas=False)
    tissue.properties['type'] = np.zeros(N*N,dtype=int)
    tissue.age = np.zeros(N*N,dtype=float)
    tissue = run(tissue, simulation_death_birth_radius(tissue,dt,10./dt,timestep/dt,rand,DELTA,game,game_constants,division_radius,True),10./dt,timestep/dt)[-1]
    tissue.properties['type'][rand.randint(N*N,size=1)]=1
    history = run(tissue,simulation_death_birth_radius(tissue,dt,timend/dt,timestep/dt,rand,DELTA,game,game_constants,division_radius),timend/dt,timestep/dt)
    return history