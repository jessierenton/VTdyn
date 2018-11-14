import numpy as np
from structure.global_constants import *
import os
import sys
import itertools
import structure
from structure.global_constants import *
from structure.cell import Tissue, BasicSpringForceNoGrowth
import structure.initialisation as init
from multiprocessing import Process
from functools import partial

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

def run(simulation,N_step,skip):  
    return np.array([sigmas for sigmas in itertools.islice(simulation,skip-1,N_step,skip)])

def write_run(simulation,N_step,skip,fname):
    with open(fname,'w',1) as f:  
        f.write('#     sigma             num             denom             sigma_weighted           num            denom            N_mutants \n')
        for results in itertools.islice(simulation,skip-1,N_step,skip):
            f.write('%e       %e       %e       %e     %e      %e        %d \n'
                    %(results[0]/results[1],results[0],results[1],results[2]/results[3],
                       results[2],results[3],results[4]))
        

def simulation_initialise_tissue_with_cluster(tissue,dt,N_steps,stepsize,rand,size,main_type):
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
        max_cluster_size = np.max(np.bincount(properties['ancestor']))
        max_cluster_index = np.argmax(np.bincount(properties['ancestor']))
        if max_cluster_size == size:
            tissue.properties['type'] = np.full(N,main_type,dtype=int)
            tissue.properties['type'][np.where(tissue.properties.pop('ancestor')==max_cluster_index)[0]] = 1-main_type
            return tissue
        

def simulation_initialise_tissue(tissue,dt,N_steps,stepsize,rand,mutant_number,main_type):
    step = 0.
    complete = False
    while step<N_steps:
        N= len(tissue)
        properties = tissue.properties
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt))
        if rand.rand() < (1./T_D)*N*dt:
            mother = rand.randint(N)
            tissue.add_daughter_cells(mother,rand)
            tissue.remove(mother)
            tissue.remove(rand.randint(N)) #kill random cell
        tissue.update(dt)
        # update_progress(step/N_steps)
    tissue.properties['type']=np.full(N,main_type,dtype=int)
    tissue.properties['type'][rand.choice(N,mutant_number,replace=False)] = 1-main_type    
    return tissue

def simulation_neutral_with_mutation_yield_sigma(tissue,dt,N_steps,stepsize,rand,mutation_rate,initial=False):
    num_denoms = np.zeros(4,dtype=float) #sigma numerator, sigma denominator, w-sigma numerator, w-sigma denominator
    step = 0.
    complete = False
    while initial or not complete:
        N= len(tissue)
        properties = tissue.properties
        mesh = tissue.mesh
        step += 1
        mesh.move_all(tissue.dr(dt))
        if rand.rand() < (1./T_D)*N*dt:
            mother = rand.randint(N)
            tissue.add_daughter_cells(mother,rand)
            r = rand.rand()
            if r < mutation_rate**2: properties['type'] = np.append(properties['type'],rand.randint(0,2,2))
            elif r < mutation_rate: properties['type'] = np.append(properties['type'],[properties['type'][mother],rand.randint(2)])
            else: properties['type'] = np.append(properties['type'],[properties['type'][mother]]*2)
            tissue.remove(mother)
            tissue.remove(rand.randint(N)) #kill random cell
        tissue.update(dt)
        if step%100==0: update_progress(step/N_steps)
        complete = (1 not in tissue.properties['type'] or 0 not in tissue.properties['type']) and step%stepsize==0
        interactions = calc_interactions(tissue)
        # numerator/denominator for calculating sigma and weighted sigma
        num_denoms += interactions[:4]*interactions[4]
        yield np.append(num_denoms,N-interactions[4])


def initialise_tissue(N,timestep,timend,rand,mutant_number,cluster=False,main_type=0):
    tissue = init.init_tissue_torus(N,N,0.01,BasicSpringForceNoGrowth(),rand,save_areas=False)
    tissue.age = np.zeros(N*N,dtype=float)
    if cluster: tissue = simulation_initialise_tissue_with_cluster(tissue,dt,timend/dt,timestep/dt,rand,mutant_number,main_type)
    else: tissue = simulation_initialise_tissue(tissue,dt,timend/dt,timestep/dt,rand,mutant_number,main_type)
    
    return tissue

def calc_interactions(state):
    types,neighbours = state.properties['type'],state.mesh.neighbours
    I_CC,I_CD,W_CC,W_CD,N_D = 0,0,0.,0.,0
    for ctype,cell_neighbours in zip(types,neighbours):
        if ctype == 1:
            Cneigh,neigh = float(sum(types[cell_neighbours])),float(len(cell_neighbours))
            I_CC += Cneigh
            I_CD += neigh - Cneigh
            W_CC += Cneigh/neigh
            W_CD += (neigh-Cneigh)/neigh
    N_D = sum(1-types)
    return np.array((I_CC,I_CD,W_CC,W_CD,N_D))
    
def save_all_interactions(history,outdir,i):
    ints = np.array([calc_interactions(state) for state in history])
    np.savetxt('%s/interactions_%d'%(outdir,i),ints,delimiter = '    ',header = 'I_CC    I_CD    W_CC    W_CD    N_D')
    
        
outdir = 'sigma_calc'
if not os.path.exists(outdir): # if the outdir doesn't exist create it
    os.makedirs(outdir)

N = 10
timend = float(sys.argv[1])
timestep = 1000.

def sim_run(mutant_number,cluster,main_type,i):
    fname = '%s/%d%d_%02d_%02d'%(outdir,cluster,main_type,mutant_number,i)
    rand = np.random.RandomState()
    mutation_rate = 1e-3
    tissue = initialise_tissue(N,timestep,1,rand,mutant_number,cluster)
    history = write_run(simulation_neutral_with_mutation_yield_sigma(tissue,dt,timend/dt,timestep/dt,rand,mutation_rate,True),timend/dt,timestep/dt,fname)   
    # np.savetxt('%s/init_%d_i_%d'%(outdir,mutant_number,i),history,delimiter='    ',header='#    sigma        sigma_weighted    N_mutants')

# if __name__ == '__main__':
#     mutant_numbers = [1,20,50]*6
#     clusters = [False,True,True]*6
#     main_types = [0,0,0,1,1,1]*3
#     # mutant_numbers = [5,50,5]
# #     clusters = [True,False,True]
# #     main_types = [0,0,1]
#     procs = []
#
#     for index, (mutant_number,cluster,main_type) in enumerate(zip(mutant_numbers,clusters,main_types)):
#             proc = Process(target=sim_run,args=(mutant_number,cluster,main_type,index))
#             procs.append(proc)
#             proc.start()
#
#     for proc in procs:
#             proc.join()


tissue= initialise_tissue(N,timestep,timend,np.random.RandomState(),1,cluster=False,main_type=1)