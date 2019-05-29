import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

sns.set_style("white")

def load_file(filename):
    with open(filename,'r') as f:
        stress = [[float(s) for s in line.split()] for line in f]
    return stress
    
def import_data(T_d,p_c,repeats):
    folder = "Td%.1f_pc%.1f"%(T_d,p_c)
    try: pop_size = [np.loadtxt(folder+'/N_cell_%03d'%i) for i in range(repeats)]
    except IOError: 
        print 'incomplete: T_d=%.1f, p_c=%d'%(T_d,p_c)
        return None,None
    stress = [load_file(folder+'/stress_%03d'%i) for i in range(repeats)]
    return np.array(pop_size),stress

def import_all_data(repeats):
    pop_size_all = {}
    stress_all = {}
    folders = [f for f in os.listdir(".") if f[:2]=="Td"]
    for folder in folders:
        T_d = float(folder[2:6])
        p_c = float(folder[9:])
        pop_size,stress = import_data(T_d,p_c,repeats)
        if pop_size is not None:
            if T_d not in pop_size_all: pop_size_all[T_d] = {}
            if T_d not in stress_all: stress_all[T_d] = {}
            pop_size_all[T_d][p_c] = pop_size
            stress_all[T_d][p_c] = stress
    return pop_size_all,stress_all

def plot_means(data_dict,ylabel,fit=False,skip=1):
    fig=plt.figure()
    time = np.arange(0,500.5,0.5)
    for T_d, data_by_p_c in data_dict.iteritems():
        for p_c, d in data_by_p_c.iteritems():
            plt.plot(time[::skip],d[::skip],label=r'$T_d=%d, p_c=%d$'%(T_d,p_c))
            plt.legend(loc='best')
            plt.xlabel('Time (hours)')
            plt.ylabel(ylabel)            
    plt.show()
    
def plot_single_runs(data,ylabel,T_d,p_c,num=None,skip=1):
    fig=plt.figure()
    time = np.arange(0,500.5,0.5)
    runs = data[T_d][p_c][:num]
    for run in runs:
        plt.plot(time[::skip],run[::skip])
        plt.title(r'$T_d=%d, p_c=%d$'%(T_d,p_c))
        plt.xlabel('Time (hours)')
        plt.ylabel(ylabel)

def plot_means_T_d(T_d,data_dict,ylabel,fit=False,skip=1):
    data_dict = {T_d:data_dict[T_d]}
    plot_means(data_dict,ylabel,fit=False,skip=1)
    
def plot_means_p_c(p_c,data_dict,ylabel,fit=False,skip=1):
    data_dict = {T_d:{p_c:pop[p_c]} for T_d,pop in pop_size_mean.iteritems() if p_c in pop}
    plot_means(data_dict,ylabel,fit=False,skip=1)

repeats = 10
pop_size_all,stress_all = import_all_data(repeats)
pop_size_mean = {T_d:{p_c:np.mean(pop_size,axis=0) for p_c,pop_size in pop_size_by_p_c.iteritems()} 
                    for T_d, pop_size_by_p_c in pop_size_all.iteritems()}          
stress_pop_mean = {T_d:{p_c:np.array([[np.mean(stress_by_cell) for stress_by_cell in stress_run] for stress_run in stress])
                    for p_c,stress in stress_by_p_c.iteritems()}
                    for T_d, stress_by_p_c in stress_all.iteritems()}
stress_mean = {T_d:{p_c:np.mean(stress,axis=0) for p_c,stress in stress_by_p_c.iteritems()}
                    for T_d,stress_by_p_c in stress_pop_mean.iteritems()}

stress_time_averaged = {T_d:{p_c:(np.mean(stress),np.std(stress)) for p_c,stress in stress_by_p_c.iteritems()} for T_d,stress_by_p_c in stress_mean.iteritems()}
pop_size_time_average = {T_d:{p_c:(np.mean(pop_size),np.std(pop_size)) for p_c,pop_size in pop_size_by_p_c.iteritems()} for T_d,pop_size_by_p_c in pop_size_mean.iteritems()}

plot_means(pop_size_mean,'Population size',skip=2)
plot_means_p_c(10,pop_size_mean,'Population size',skip=2)
plot_means_p_c(5,pop_size_mean,'Population size',skip=2)
# plot_single_runs(pop_size_all, 'Population_size',18,20,num=5,skip=10)
# plot_single_runs(pop_size_all, 'Population size',19,10,num=5,skip=10)
# plot_single_runs(pop_size_all, 'Population size',18,15,num=5,skip=10)
#
#dictionaries store data: first key is T_d value, second is p_c


    
        
    