import numpy as np
import matplotlib.pyplot as plt

def get_adj_matrix(neighbours): #get adjacency matrix from a list of neighbours by vertex
    n = len(neighbours)
    adj_mat = np.zeros((n,n),dtype=float)
    for cell,cell_neighbours in enumerate(neighbours):
        adj_mat[cell][cell_neighbours] = 1
    return adj_mat

def get_stationary_distr(adj_mat):
    return np.sum(adj_mat,axis=1)/np.sum(adj_mat)

def get_prob_mat(adj_mat):
    prob_mat = adj_mat/np.sum(adj_mat,axis=1,keepdims=True)
    if True in np.isnan(prob_mat):
        raise Exception('Graph not connected')
    return prob_mat
    
def build_M(p_mat):
    n = len(p_mat)
    M_block = p_mat - np.identity(n)*2
    return np.block([[M_block if i == j else np.zeros((n,n)) for j in range(n)] for i in range(n)])
    
def build_L(p_mat):
    n = len(p_mat)
    def block(i,j):
        return np.stack([p_mat[i] if l==j else np.zeros(n) for l in range(n)])
    return np.block([[block(i,j) for j in range(n)] for i in range(n)])


def find_tau_matrix(adj_mat):
    n = len(adj_mat)
    b = np.ones(n*n)*-2
    p_mat = get_prob_mat(adj_mat)
    G = build_M(p_mat) + build_L(p_mat)
    for i in range(n*n):
        if i%n==i/n:
            b[i] = 0
            for j in range(n*n):
                if i == j: G[i,j] = 1
                else: G[i,j] = 0
    tau = np.reshape(np.linalg.solve(G,b),(n,n))
    return tau

def find_tau_ii_plus(tau_mat,p_mat):
    return np.array([1 + np.dot(p_vec,tau_vec) for p_vec,tau_vec in zip(tau_mat,p_mat)])
    
def get_tau_1(pi_vec,tau_plus_vec):
    return np.dot(pi_vec,tau_plus_vec) - 1.

def get_tau_2(pi_vec,tau_plus_vec):
    return np.dot(pi_vec,tau_plus_vec) - 2.

def get_tau_3(pi_vec,tau_plus_vec,p_mat):
    p_ii_2_vec = np.array([np.dot(p,pT) for p,pT in zip(p_mat,p_mat.T)])
    return np.sum(pi_vec*tau_plus_vec*(1+p_ii_2_vec)) - 3.

def find_taus(adj_mat):
    p_mat = get_prob_mat(adj_mat)
    tau_plus_vec = find_tau_ii_plus(find_tau_matrix(adj_mat),p_mat)
    pi_vec = get_stationary_distr(adj_mat)
    
    tau_1 = get_tau_1(pi_vec,tau_plus_vec)
    tau_2 = get_tau_2(pi_vec,tau_plus_vec)
    tau_3 = get_tau_3(pi_vec,tau_plus_vec,p_mat)
    
    return tau_1,tau_2,tau_3

def find_critical_ratio(tau_1,tau_2,tau_3):  
    return tau_2/(tau_3-tau_1)

def plot_fix_prob_vs_b(N,DELTA,c,tau_1,tau_2,tau_3):
    b_vals = np.arange(91)/10.+1
    rho_C = 1./N + DELTA/(2*N)*(-c*tau_2+b_vals*(tau_3-tau_1))
    plt.figure()
    plt.plot(b_vals,rho_C)
    plt.show()
    

if __name__ == '__main__':
    adj_mat = np.array([[0,1,0,0,1],[1,0,1,0,0],[0,1,0,1,0],[0,0,1,0,1],[1,0,0,1,0]],dtype=float)
    # adj_mat = np.ones(100)-np.identity(100)
    from igraph import Graph
    neighbours = Graph.Ring(5).get_adjlist()
    adj_mat = get_adj_matrix(neighbours)
    taus = find_taus(adj_mat)
    print find_critical_ratio(*taus)
    plot_fix_prob_vs_b(len(adj_mat),0.025,1.,*taus)
    
    
    
