import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

sns.set_style("white")

def load_file(filename):
    with open(filename,'r') as f:
        stress = [[float(s) for s in line.split()] for line in f]
    return stress
    
def import_data(T_d):
    folder = "Td%.1f_pcinf/"%T_d
    pop_files = [folder+f for f in os.listdir(folder) if f[0]=="N"]
    stress_files = [folder+f for f in os.listdir(folder) if f[0]=="s"]
    pop_size = [np.loadtxt(f) for f in pop_files]
    stress = [load_file(f) for f in stress_files]
    return np.array(pop_size),stress

def import_all_data():
    pop_size_all = []
    stress_all = []
    T_d_list = []
    folders = [f for f in os.listdir(".") if f[:2]=="Td"]
    for folder in folders:
        T_d = float(folder[2:6])
        T_d_list.append(T_d)
        pop_size,stress = import_data(T_d)
        pop_size_all.append(pop_size)
        stress_all.append(stress)
    return T_d_list,pop_size_all,stress_all

def plot_means(data_list,T_d_list,ylabel,fit=False,skip=1):
    fig=plt.figure()
    for T_d,data in zip(T_d_list,data_list):
        time = np.arange(len(data))/2.
        plt.plot(time[::skip],data[::skip],label=r'$T_d=%.1f$'%(T_d))
        plt.legend(loc='best')
        plt.xlabel('Time (hours)')
        plt.ylabel(ylabel)            
    plt.show()
    
def plot_single_runs(data,T_d,ylabel,num=None,skip=1):
    fig=plt.figure()
    runs = data[:num]
    for run in data:
        time = np.arange(len(run))/2.
        plt.plot(time[::skip],run[::skip])
    plt.title(r'$T_d=%.1f$'%(T_d))
    plt.xlabel('Time (hours)')
    plt.ylabel(ylabel)

def plot_single_runs_x4(data,T_d_list,T_d_vals,ylabel,num=None,skip=1):
    data = [d for d,T_d in zip(data,T_d_list) if T_d in T_d_vals]
    fig,axs=plt.subplots(2,2)
    for i,(ax,runs,T_d) in enumerate(zip(axs.flat,data,T_d_vals)):
        runs = runs[:num]
        for run in runs:
            time = np.arange(len(run))/2.
            ax.plot(time[::skip],run[::skip])
            ax.set_title(r'$T_d=%.1f$'%(T_d))
    axs[1,0].set_xlabel('Time (hours)')
    axs[0,0].set_xlabel('Time (hours)')
    axs[0,0].set_ylabel(ylabel)
    axs[0,1].set_ylabel(ylabel)

def plot_single_runs_x6(data,T_d_list,T_d_vals,ylabel,num=None,skip=1):
    data = [d for d,T_d in zip(data,T_d_list) if T_d in T_d_vals]
    fig,axs=plt.subplots(3,2)
    for i,(ax,runs,T_d) in enumerate(zip(axs.flat,data,T_d_vals)):
        runs = runs[:num]
        for run in runs:
            time = np.arange(len(run))/2.
            ax.plot(time[::skip],run[::skip])
            ax.set_title(r'$T_d=%.1f$'%(T_d))
    axs[2,1].set_xlabel('Time (hours)')
    axs[2,0].set_xlabel('Time (hours)')
    axs[0,0].set_ylabel(ylabel)
    axs[1,0].set_ylabel(ylabel)
    axs[2,0].set_ylabel(ylabel)
    for i in (0,1):
        for j in (0,1):
            axs[i,j].set_xticks([])
    for j in (0,1,2):
        axs[j,1].set_yticks([])
    

def mean_various_sim_length(data_list,max_length=None):
    means = []
    for data in data_list:
        max_timestep = min(min(len(d) for d in data),max_length)
        data = np.array([d[:max_timestep] for d in data])
        means.append(np.mean(data,axis=0))
    return np.array(means)
    
    
def plot_means_T_d(T_d,data_dict,ylabel,fit=False,skip=1):
    data_dict = {T_d:data_dict[T_d]}
    plot_means(data_dict,ylabel,fit=False,skip=1)
    
def plot_means_p_c(p_c,data_dict,ylabel,fit=False,skip=1):
    data_dict = {T_d:{p_c:pop[p_c]} for T_d,pop in pop_size_mean.iteritems() if p_c in pop}
    plot_means(data_dict,ylabel,fit=False,skip=1)

MAX_TIMESTEP=600

T_d_list,pop_size_all,stress_all = import_all_data()
pop_size_mean = mean_various_sim_length(pop_size_all,MAX_TIMESTEP)   
stress_pop_mean = np.array([[[np.mean(stress_by_cell) for stress_by_cell in stress_t] for stress_t in stress] for stress in stress_all])
stress_mean = mean_various_sim_length(stress_pop_mean,MAX_TIMESTEP)   

# plot_means(pop_size_mean,T_d_list,'Population size',skip=2)
# plot_single_runs_x6(pop_size_all,T_d_list,(17.5,18.0,18.5,19,19.5,20.0),'Population size',skip=5)

plot_means(stress_mean,T_d_list,'Mean stress',skip=2)
    
        
    