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

def simulation_neutral_bd_coupled(tissue,dt,N_steps,stepsize,rand):
    """simulation loop for neutral process tracking ancestor ids"""
    complete=False
    step = 0.
    print dt
    while True:
        N= len(tissue)
        properties = tissue.properties
        mesh = tissue.mesh
        mesh.move_all(tissue.dr(dt))
        if rand.rand() < (1./T_D)*N*dt:
            mother = rand.randint(N)
            tissue.add_daughter_cells(mother,rand)
            tissue.remove(mother)
            tissue.remove(rand.randint(N)) #kill random cell
        tissue.update(dt)
        print_progress(step,N_steps)
        step += 1 
        yield tissue
        
def simulation_pd_global_density_dep(tissue,dt,N_steps,stepsize,rand,params,DELTA,game,game_constants,til_fix=False):
    KAPPA = params['KAPPA']
    N0 = params['N0']
    LAMBDA = (1./T_D)
    MU = LAMBDA/(1.+KAPPA)
    step = 0.
    complete = False
    while not til_fix or not complete:
        print_progress(step,N_steps)
        N= len(tissue)
        properties = tissue.properties
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt))
        do_birth = rand.rand() < LAMBDA*N*dt
        do_death = rand.rand() < MU*N*(1+KAPPA*N/N0)*dt
        if do_birth:
            fitnesses = recalculate_fitnesses(mesh.neighbours,properties['type'],DELTA,game,game_constants)
            mother = np.where(np.random.multinomial(1,fitnesses/sum(fitnesses))==1)[0][0]   
            tissue.add_daughter_cells(mother,rand)
            properties['type'] = np.append(properties['type'],[properties['type'][mother]]*2)
            tissue.remove(mother)
            N+=1
        if do_death:
            tissue.remove(rand.randint(N))
        tissue.update(dt)
        complete = (1 not in tissue.properties['type'] or 0 not in tissue.properties['type']) and step%stepsize==0  
        yield tissue

def simulation_pd_global_density_dep2(tissue,dt,N_steps,stepsize,rand,params,DELTA,game,game_constants,til_fix=False):
    N0 = params['N0']
    LAMBDA = (1./T_D)
    MU = LAMBDA
    step = 0.
    complete = False
    while not til_fix or not complete:
        print_progress(step,N_steps)
        N= len(tissue)
        properties = tissue.properties
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt))
        do_birth = rand.rand() < LAMBDA*N*dt
        do_death = rand.rand() < MU*(N/N0)*N*dt
        if do_birth:
            fitnesses = recalculate_fitnesses(mesh.neighbours,properties['type'],DELTA,game,game_constants)
            mother = np.where(np.random.multinomial(1,fitnesses/sum(fitnesses))==1)[0][0]   
            tissue.add_daughter_cells(mother,rand)
            properties['type'] = np.append(properties['type'],[properties['type'][mother]]*2)
            tissue.remove(mother)
            N+=1
        if do_death:
            tissue.remove(rand.randint(N))
        tissue.update(dt)
        complete = (1 not in tissue.properties['type'] or 0 not in tissue.properties['type']) and step%stepsize==0  
        yield tissue

def simulation_pd_local_density_threshold(tissue,dt,N_steps,stepsize,rand,params,DELTA,game,game_constants,til_fix=False):
    R = params['density_radius'] #desity-dependence radius
    density_threshold_birth = params['density_threshold_birth']
    density_threshold_death = params['density_threshold_death']
    MU = (1./T_D)
    LAMBDA = params['birth-to-death_ratio']*MU
    step = 0.
    complete = False
    mesh = tissue.mesh
    while not til_fix or not complete:
        print_progress(step,N_steps)
        N= len(tissue)
        properties = tissue.properties
        step += 1
        mesh.move_all(tissue.dr(dt))
        do_death = rand.rand() < MU*N*dt
        do_birth = rand.rand() < LAMBDA*N*dt
        if do_birth:
            fitnesses = recalculate_fitnesses(mesh.neighbours,properties['type'],DELTA,game,game_constants)
            mother = np.where(np.random.multinomial(1,fitnesses/sum(fitnesses))==1)[0][0]   
            if mesh.cell_local_density(R,mother)<density_threshold_birth:
                tissue.mother_ids.append(mother)
                tissue.add_daughter_cells(mother,rand)
                properties['type'] = np.append(properties['type'],[properties['type'][mother]]*2)
                tissue.remove(tissue.cell_ids[mother])
                N+=1
        if do_death:
            dead =rand.randint(N)
            if mesh.cell_local_density(R,dead)>density_threshold_death: 
                tissue.dead_ids.append(tissue.remove[dead])
                tissue.remove(rand.randint(N))
        tissue.update(dt)
        complete = (1 not in tissue.properties['type'] or 0 not in tissue.properties['type']) and step%stepsize==0  
        yield tissue
# def simulation_pd_density_dep(tissue,dt,N_steps,stepsize,rand,DELTA,OMEGA,game,game_constants,til_fix=False):
#     step = 0.
#     complete = False
#     while not til_fix or not complete:
#         print_progress(step,N_steps)
#         N= len(tissue)
#         properties = tissue.properties
#         mesh = tissue.mesh
#         step += 1
#         mesh.move_all(tissue.dr(dt))
#         if rand.rand() < (1./T_D)*N*dt:
#             fitnesses = recalculate_fitnesses(mesh.neighbours,properties['type'],DELTA,game,game_constants)
#             mother = np.where(np.random.multinomial(1,fitnesses/sum(fitnesses))==1)[0][0]
#             densities = mesh.local_density()
#             prob_dying = 1 - OMEGA + OMEGA*densities
#             prob_dying[mother]=0
#             prob_dying = prob_dying/sum(prob_dying)
#             dead = np.where(np.random.multinomial(1,prob_dying)==1)[0][0]
#             tissue.add_daughter_cells(mother,rand)
#             properties['type'] = np.append(properties['type'],[properties['type'][mother]]*2)
#             tissue.remove((mother,dead))
#         tissue.update(dt)
#         complete = (1 not in tissue.properties['type'] or 0 not in tissue.properties['type']) and step%stepsize==0
#         yield tissue


def run_simulation(simulation,N,timestep,timend,rand,params,DELTA,game,game_constants,init_time=10.,til_fix=False,mutant_num=1):
    tissue = init.init_tissue_torus(N,N,0.01,BasicSpringForceNoGrowth(),rand,save_areas=False)
    tissue.properties['type'] = np.zeros(N*N,dtype=int)
    tissue.age = np.zeros(N*N,dtype=float)
    if init_time is not None:
        tissue = run(tissue, simulation(tissue,dt,init_time/dt,timestep/dt,rand,DELTA,KAPPA,N0,game,game_constants,False),10./dt,1./dt)[-1]
        tissue.reset()
    tissue.properties['type'][rand.choice(N*N,size=mutant_num,replace=False)]=1
    history = run(tissue, simulation(tissue,dt,timend/dt,timestep/dt,rand,params,DELTA,game,game_constants,til_fix=til_fix),timend/dt,timestep/dt)
    return history
    
def calc_pref_ld(N,timestep,timend,rand):
    tissue = init.init_tissue_torus(N,N,0.01,BasicSpringForceNoGrowth(),rand,save_areas=False)
    history = run(tissue, simulation_neutral_bd_coupled(tissue,dt,timend/dt,timestep/dt,rand),timend/dt,timestep/dt)[int(10/timestep):]
    densities = [tissue.mesh.local_density() for tissue in history]
    np.savetxt('local_densities',densities)
    
    