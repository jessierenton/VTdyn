import numpy as np

def load_nh(i):
    with open('neighbours/%02d'%i,'r') as f:
        a = f.readlines()
    for i,line in enumerate(a):
        a[i] = np.array(line.split(),dtype=int)  
    return a

def prob_neighbour():
    count = compare_prob = daughter_count = 0
    for i,mother in enumerate(mother_mid[:events]):
        mother_neighbours = neighbour_history[i][mother]
        compare_prob +=len(mother_neighbours)/100.
        dead = dead_mid[i]
        next_dead = dead_mid[i+1]
        if next_dead >=98: daughter_count +=1
        if next_dead > mother: next_dead -=1
        if next_dead > dead: next_dead -=1
        if next_dead in mother_neighbours: count +=1
    return count/float(events+1),compare_prob/(events+1), daughter_count/float(events+1)
    

def density_at_death(dead_mid,density_history):
    return [density[id] for density,id in zip(density_history,dead_mid)]
       
    

events = 811
neighbour_history = [load_nh(i) for i in range(events+1)]
density_history = [np.loadtxt('densities/%02d'%i) for i in range(events+1)]
event_data = np.loadtxt('event_data',dtype=int)
mother_cid = event_data[:,2]
mother_mid = event_data[:,1]
dead_cid = event_data[:,4]
dead_mid = event_data[:,3]

print prob_neighbour()
densities_at_death = density_at_death(dead_mid,density_history)
mean_local_density_time = np.mean(density_history,axis=1)