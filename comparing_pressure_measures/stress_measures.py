import numpy as np 

def cell_pressure(tissue,i,pressure_type):
        """calculates the pressure p_i on a single cell i according to the formula p_i = sum_j mag(F^rep_ij.u_ij)/l_ij
        where F^rep_ij is the repulsive force between i and j, i.e. F^rep_ij=F_ij if Fij is positive, 0 otherwise;
        u_ij is the unit vector between the i and j cell centres and l_ij is the length of the edge between cells i and j"""
        edge_lengths = tissue.mesh.edge_lengths(i)
        if pressure_type == "virial":
            return virial_pressure(tissue,i)
        else:
            if pressure_type == "repulsive":
                forces = tissue.Force.force_ij(tissue,i)
                forces[forces<0] = 0
            elif pressure_type == "magnitude":
                forces = np.fabs(tissue.Force.force_ij(tissue,i))
            elif pressure_type == "full":
                forces = tissue.Force.force_ij(tissue,i)
            return sum(forces/edge_lengths)    

def virial_pressure(tissue,i):
    area = tissue.mesh.areas[i]
    distances = 0.5*tissue.mesh.distances[i]
    forces = tissue.Force.force_ij(tissue,i)
    return 0.5*sum(forces*distances)/area
    
def tissue_pressure(tissue,pressure_type):
    return np.array([cell_pressure(tissue,i,pressure_type) for i in range(len(tissue))])
