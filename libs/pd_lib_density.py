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

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------POISSON-CONSTANT-POP-SIZE-AND-FITNESS------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def prisoners_dilemma_averaged(cell_type,neighbour_types,b,c):
    return -c*cell_type+b*np.sum(neighbour_types)/len(neighbour_types)

def prisoners_dilemma_accumulated(cell_type,neighbour_types,b,c):
    return -c*cell_type*len(neighbour_types)+b*np.sum(neighbour_types)

def get_fitness(cell_type,neighbour_types,DELTA,game,game_constants):
    return 1+DELTA*game(cell_type,neighbour_types,*game_constants)

def recalculate_fitnesses(neighbours_by_cell,types,DELTA,game,game_constants):
    return 1+DELTA*np.array([game(types[cell],types[neighbours],*game_constants) for cell,neighbours in enumerate(neighbours_by_cell)])

def simulation_pd_density_dep(tissue,dt,N_steps,stepsize,rand,DELTA,OMEGA,game,game_constants,til_fix=False):
    step = 0.
    complete = False
    while not til_fix or not complete:
        # print_progress(step,N_steps)
        N= len(tissue)
        properties = tissue.properties
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt))
        if rand.rand() < (1./T_D)*N*dt:
            fitnesses = recalculate_fitnesses(mesh.neighbours,properties['type'],DELTA,game,game_constants)
            mother = np.where(np.random.multinomial(1,fitnesses/sum(fitnesses))==1)[0][0]   
            densities = mesh.local_density()
            prob_dying = 1 - OMEGA + OMEGA*densities
            prob_dying[mother]=0
            prob_dying = prob_dying/sum(prob_dying)
            dead = np.where(np.random.multinomial(1,prob_dying)==1)[0][0]
            tissue.add_daughter_cells(mother,rand)
            properties['type'] = np.append(properties['type'],[properties['type'][mother]]*2)
            tissue.remove((mother,dead))
        tissue.update(dt)
        complete = (1 not in tissue.properties['type'] or 0 not in tissue.properties['type']) and step%stepsize==0  
        yield tissue

def simulation_pd_density_dep_data(tissue,dt,N_steps,stepsize,rand,DELTA,OMEGA,game,game_constants,til_fix=False):
    step = 0.
    complete = False
    with open('dd/event_data','a') as f:
        f.write('event#  mother_cid  mother_mid  dead_cid  dead_mid \n')
        event = 0
    while not til_fix or not complete:
        # print_progress(step,N_steps)
        N= len(tissue)
        properties = tissue.properties
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt))
        if rand.rand() < (1./T_D)*N*dt:
            fitnesses = recalculate_fitnesses(mesh.neighbours,properties['type'],DELTA,game,game_constants)
            mother = np.where(np.random.multinomial(1,fitnesses/sum(fitnesses))==1)[0][0]   
            densities = mesh.local_density()
            prob_dying = 1 - OMEGA + OMEGA*densities
            prob_dying[mother]=0
            prob_dying = prob_dying/sum(prob_dying)
            dead = np.where(np.random.multinomial(1,prob_dying)==1)[0][0]
            with open('dd/event_data','a') as f:
                f.write('%4d    %2d          %2d          %2d        %2d \n'%(event,mother,tissue.cell_ids[mother],dead,tissue.cell_ids[dead]))
            with open('dd/neighbours/%02d'%event,'w') as f:
                for cell_neighbours in mesh.neighbours:
                    for cell in cell_neighbours:
                        f.write('%2d '%cell)
                    f.write('\n')
            with open('dd/densities/%02d'%event,'w') as f:
                for d in densities:
                    f.write('%2f \n'%d)
            tissue.add_daughter_cells(mother,rand)
            properties['type'] = np.append(properties['type'],[properties['type'][mother]]*2)
            tissue.remove((mother,dead))
            event += 1
        tissue.update(dt)
        complete = (1 not in tissue.properties['type'] or 0 not in tissue.properties['type']) and step%stepsize==0  
        yield tissue



def run_simulation(simulation,N,timestep,timend,rand,DELTA,OMEGA,game,game_constants,til_fix=False,mutant_num=1):
    tissue = init.init_tissue_torus(N,N,0.01,BasicSpringForceNoGrowth(),rand,save_areas=False)
    tissue.properties['type'] = np.zeros(N*N,dtype=int)
    tissue.age = np.zeros(N*N,dtype=float)
    tissue = run(tissue, simulation(tissue,dt,10./dt,timestep/dt,rand,DELTA,OMEGA,game,game_constants,False),10./dt,1./dt)[-1]
    tissue.reset()
    tissue.properties['type'][rand.choice(N*N,size=mutant_num,replace=False)]=1
    history = run(tissue, simulation(tissue,dt,timend/dt,timestep/dt,rand,DELTA,OMEGA,game,game_constants,til_fix=til_fix),timend/dt,timestep/dt)
    return history