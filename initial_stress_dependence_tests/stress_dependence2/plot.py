import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from operator import itemgetter
from scipy.stats import linregress

sns.set_style("white")
current_palette = sns.color_palette()

def load_file(filename):
    with open(filename,'r') as f:
        stress = [[float(s) for s in line.split()] for line in f]
    return stress
    
def import_data(folder,T_d,p_c):
    if folder[-1]!='/': folder+='/'
    pop_files = [folder+f for f in os.listdir(folder) if f[0]=="N"]
    stress_files = [folder+f for f in os.listdir(folder) if f[0]=="s"]
    pop_size = [np.loadtxt(f) for f in pop_files]
    stress = [load_file(f) for f in stress_files]
    return np.array(pop_size),stress

def import_all_data(readdir,sort=False):
    pop_size_all = []
    stress_all = []
    T_d_list = []
    p_c_list = []
    folders = [readdir+'/'+f for f in os.listdir(readdir) if f[:2]=="Td"]
    L=len(readdir)
    for folder in folders:
        T_d = float(folder[L+3:L+7])
        p_c = float(folder[L+10:])
        T_d_list.append(T_d)
        p_c_list.append(p_c)
        pop_size,stress = import_data(folder,T_d,p_c)
        pop_size_all.append(pop_size)
        stress_all.append(stress)
    if sort is not None:
        if sort == 'p_c': 
            sort_by = p_c_list[:]
            p_c_list.sort()
            T_d_list = [t for i,t in sorted(zip(sort_by,T_d_list))]
        elif sort == 'T_d': 
            sort_by = T_d_list[:]
            T_d_list.sort()
            p_c_list = [p for i,p in sorted(zip(sort_by,p_c_list))]
        pop_size_all = [p for i,p in sorted(zip(p_c_list,pop_size_all))]
        stress_all = [s for i,s in sorted(zip(p_c_list,stress_all))]
    return T_d_list,p_c_list,pop_size_all,stress_all

def plot_means(data_list,T_d_list,p_c_list,ylabel,fit=False,skip=1,timestep=1.):
    fig=plt.figure()
    for i,(T_d,p_c,data) in enumerate(zip(T_d_list,p_c_list,data_list)):
        time = np.arange(len(data))*timestep
        if np.all(np.array(T_d_list)==T_d):
            label = r'$p_c=%.1f$'%p_c
            title = r'$T_d=%.1f$'%(T_d)
        elif np.all(np.array(p_c_list)==p_c):
            label = r'$T_d=%.1f$'%(T_d)
            title = r'$p_c=%.1f$'%p_c
        else: 
            label = r'$T_d=%.1f, p_c=%.1f$'%(T_d,p_c)
            title = None
        plt.plot(time[::skip],data[::skip],label=label,color=current_palette[i])
    plt.legend(loc='best')
    plt.xlabel('Time (hours)')
    plt.ylabel(ylabel)  
    if title is not None: plt.title(title)          
    plt.show()
    
def plot_single_runs(data,T_d,p_c,ylabel,num=None,skip=1,bestfit=False,timestep=1.):
    fig=plt.figure()
    runs = data[:num]
    if bestfit: mean_gradient = 0
    for i,run in enumerate(runs):
        time = np.arange(len(run))*timestep
        plt.plot(time[::skip],run[::skip],color=current_palette[i])
        if bestfit:
            m,c = linregress(time,run)[:2]
            print 'T_d=%.1f, p_c=%.1f: m=%.3e, c=%.2f'%(T_d,p_c,m,c)
            plt.plot(time,m*time+c,color='black')
            mean_gradient += m
    if bestfit: 
        mean_gradient/=len(runs)
        print 'mean gradient = %.3e'%mean_gradient
    plt.title(r'$T_d=%.1f,p_c=%.1f$'%(T_d,p_c))
    plt.xlabel('Time (hours)')
    plt.ylabel(ylabel)
    plt.show()

def plot_single_runs_x4(data,T_d_list,T_d_vals,p_c_list,p_c_vals,ylabel,num=None,skip=1,timestep=1.):
    data = [d for d,T_d,p_c in sorted(zip(data,T_d_list,p_c_list),key=itemgetter(1),reverse=True) 
                if T_d in T_d_vals and p_c_vals[T_d_vals.index(T_d)]==p_c]
    fig,axs=plt.subplots(2,2)
    for i,(ax,runs,T_d) in enumerate(zip(axs.flat,data,T_d_vals)):
        runs = runs[:num]
        for run in runs:
            time = np.arange(len(run))*timestep
            ax.plot(time[::skip],run[::skip])
            ax.set_title(r'$T_d=%.1f$'%(T_d))
    axs[1,0].set_xlabel('Time (hours)')
    axs[0,0].set_xlabel('Time (hours)')
    axs[0,0].set_ylabel(ylabel)
    axs[0,1].set_ylabel(ylabel)

def plot_single_runs_x6(data,T_d_list,T_d_vals,ylabel,num=None,skip=1,timestep=1.):
    data = [d for d,T_d in zip(data,T_d_list) if T_d in T_d_vals]
    fig,axs=plt.subplots(3,2)
    for i,(ax,runs,T_d) in enumerate(zip(axs.flat,data,T_d_vals)):
        runs = runs[:num]
        for run in runs:
            time = np.arange(len(run))*timestep
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

def read_timestep_from_info(info_file):
    with open(info_file,'r') as f:
        timestep = float(f.readline().split()[2][:-1])
    return timestep
    
MAX_TIMESTEP=None
timestep = read_timestep_from_info('./data/Td20.0_pc4.5/info_000')

T_d_list,p_c_list,pop_size_all,stress_all = import_all_data('data',sort='p_c')
pop_size_mean = mean_various_sim_length(pop_size_all,MAX_TIMESTEP)   
stress_pop_mean = np.array([[[np.mean(stress_by_cell) for stress_by_cell in stress_t] for stress_t in stress] for stress in stress_all])
stress_mean = mean_various_sim_length(stress_pop_mean,MAX_TIMESTEP)   

plot_means(pop_size_mean,T_d_list,p_c_list,'Population size',skip=1,timestep=timestep)
# plot_means(stress_mean,T_d_list,p_c_list,'Mean stress',skip=1)
plot_single_runs(pop_size_all[p_c_list.index(4.5)],20.,4.5,'Population size',skip=10,bestfit=False,num=3)
        
    