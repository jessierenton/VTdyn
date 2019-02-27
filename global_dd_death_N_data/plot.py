import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

KAPPA_vals = [0.1,1.0,10.,100.,1000.]
repeats = 5 

def load_data(KAPPA_vals):
    data = {}
    for KAPPA in KAPPA_vals:
        data[KAPPA] = np.array([np.loadtxt('data/K_%.1f_%d'%(KAPPA,i)) for i in range(repeats)])
    return data        

def plot_means_vs_time(KAPPA_vals):
    N_data = load_data(KAPPA_vals)
    fig = plt.figure()
    for KAPPA in KAPPA_vals:
        plt.plot(np.mean(N_data[KAPPA],axis=0)[::10],label=r'$\kappa = %.1f$'%KAPPA)
    plt.xlabel('Time (hours)')
    plt.ylabel('Population size, N')
    plt.legend(loc='best')
    plt.savefig('meanN_over_time')
    plt.show()
    
def plot_time_av_mean_with_var(KAPPA_vals):
    N_data = load_data(KAPPA_vals)
    fig = plt.figure()
    mean_N = [np.mean(N_data[KAPPA],axis=1) for KAPPA in KAPPA_vals]
    time_av = [np.mean(mN) for mN in mean_N]
    stdev = np.array([np.std(mN) for mN in mean_N])
    plt.errorbar(KAPPA_vals,time_av,stdev/2)
    plt.xlabel(r'$\kappa$')
    plt.xscale('log')
    plt.ylabel('Time-averaged population size, N')
    plt.savefig('time_av_N')
    plt.show()

def plot_KAPPA(KAPPA):
    N_data = load_data((KAPPA,))
    plt.plot(N_data[KAPPA].T[::100])
    plt.xlabel('Time (hours)')
    plt.ylabel('Population size, N')
    plt.show()
    
plot_means_vs_time(KAPPA_vals)
plot_time_av_mean_with_var(KAPPA_vals)
# plot_KAPPA(1000)
# N_data = load_data(KAPPA_vals)
# Time_av = {KAPPA: np.mean(N_vals,axis=1) for KAPPA,N_vals in N_data.items()}

