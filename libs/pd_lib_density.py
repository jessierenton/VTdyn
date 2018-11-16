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

def get_payoff(cell_type,neighbour_types,DELTA,game,game_constants):
    return 1+DELTA*game(cell_type,neighbour_types,*game_constants)

def recalculate_payoffs(neighbours_by_cell,types,DELTA,game,game_constants):
    return np.array([game(types[cell],types[neighbours],*game_constants) for cell,neighbours in enumerate(neighbours_by_cell)])

def simulation_pd_density_dep(tissue,dt,N_steps,stepsize,rand,DELTA,OMEGA,game,game_constants,initial=False):
    step = 0.
    complete = False
    while initial or not complete:
        N= len(tissue)
        properties = tissue.properties
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt))
        if rand.rand() < (1./T_D)*N*dt:
            payoffs = recalculate_payoffs(mesh.neighbours,properties['type'],DELTA,game,game_constants)
            fitnesses = DELTA*payoffs + OMEGA*mesh.areas
            mother = np.where(np.random.multinomial(1,fitnesses/sum(fitnesses))==1)[0][0]   
            tissue.add_daughter_cells(mother,rand)
            properties['type'] = np.append(properties['type'],[properties['type'][mother]]*2)
            tissue.remove(mother)
            tissue.remove(rand.randint(N)) #kill random cell
        tissue.update(dt)
        update_progress(step/N_steps)
        complete = (1 not in tissue.properties['type'] or 0 not in tissue.properties['type']) and step%stepsize==0  
        yield tissue

def run_simulation_pd_density_dep(N,timestep,timend,rand,DELTA,OMEGA,game,game_constants,save_areas=True):
    """prisoners dilemma with decoupled birth and death"""
    tissue = init.init_tissue_torus(N,N,0.01,BasicSpringForceNoGrowth(),rand,save_areas=save_areas)
    tissue.properties['type'] = np.zeros(N*N,dtype=int)
    tissue.age = np.zeros(N*N,dtype=float)
    tissue = run(tissue, simulation_pd_density_dep(tissue,dt,10./dt,timestep/dt,rand,DELTA,OMEGA,game,game_constants,True),10./dt,timestep/dt)[-1]
    tissue.properties['type'][rand.randint(N*N,size=1)]=1
    history = run(tissue, simulation_pd_density_dep(tissue,dt,timend/dt,timestep/dt,rand,DELTA,OMEGA,game,game_constants),timend/dt,timestep/dt)
    return history