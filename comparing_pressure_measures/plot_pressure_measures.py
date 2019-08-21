import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style('white')
palette = sns.color_palette()

def import_from_file(fname):
    with open(fname,'r') as f:
        data = [np.array(line.split(),dtype=float)
                        for line in f.readlines()]
    return data

def import_with_loadtxt(folder,file_pref,T_D_vals,repeats,dtype):
    return {T_D:np.array([np.loadtxt(folder+'T_D_%.0f/%s_%03d'%(T_D,file_pref,r),dtype=dtype) for r in range(repeats)]) for T_D in T_D_vals} 

def import_pressure(folder,T_D_vals,pressure_type,repeats):
    return {T_D:[import_from_file(folder+'T_D_%.0f/pressure_%s_%03d'%(T_D,pressure_type,r)) for r in range(repeats)]
                    for T_D in T_D_vals} 

def import_all_data(folder,T_D_vals,repeats):
    pop_size = import_with_loadtxt(folder,"N_cell",T_D_vals,repeats,int)
    n_dist = import_with_loadtxt(folder,"neigh_distr",T_D_vals,repeats,int)
    a_mean = import_with_loadtxt(folder,"area_mean",T_D_vals,repeats,float)
    p_v = import_pressure(folder,T_D_vals,"virial",repeats)
    p_r = import_pressure(folder,T_D_vals,"repulsive",repeats)
    p_m = import_pressure(folder,T_D_vals,"magnitude",repeats) 
    p_f = import_pressure(folder,T_D_vals,"full",repeats) 

    return pop_size,n_dist,a_mean,p_v,p_r,p_m,p_f

def data_means_by_pop_size(pop_size,data,T_D_vals):
    max_N = int(max((max(N_vals) for N_by_T_D in pop_size.itervalues() for N_vals in N_by_T_D)))
    min_N = int(min((min(N_vals) for N_by_T_D in pop_size.itervalues() for N_vals in N_by_T_D)))
    N_vals = np.arange(min_N,max_N+1,dtype=int)
    data_by_N = [[] for N in N_vals]
    for T_D in T_D_vals:
        for N_history,data_history in zip(pop_size[T_D],data[T_D]):
            for N,d in zip(N_history,data_history):
                data_by_N[N-min_N].append(np.mean(d))
    delete_id = [i for i,s in enumerate(data_by_N) if len(s)==0]
    N_vals = np.delete(N_vals,delete_id)
    mean_data_by_N = [np.mean(s) for i,s in enumerate(data_by_N) if i not in delete_id]
    return N_vals,np.array(mean_data_by_N)

def plot_mean_stresses_by_pop_size(pop_size,pressures,T_D_vals,fname=None,legend=False,ylims=None):
    fig,ax = plt.subplots()
    for i,(p,label,d,c) in enumerate(pressures):
        N_vals,stress_by_N = data_means_by_pop_size(pop_size,p,T_D_vals)
        plt.plot(N_vals,stress_by_N/d,'.',label=label,color=c,zorder=50-i*5)
    ax.plot(N_vals,np.zeros(len(N_vals)),'--k',lw=0.5)
    ax.set_xlabel('N')
    ax.set_ylabel('Mean pressure')
    if legend: plt.legend(loc='best')
    if ylims is not None: ax.set_ylim(ylims)
    if fname is not None: plt.savefig(fname)
    return ax    

def plot_areas_by_pop_size(pop_size,a_means,T_D_vals,color,ax=None,fname=None):
    if ax is None: fig,ax = plt.subplots()
    N_vals,area_by_N = data_means_by_pop_size(pop_size,a_means,T_D_vals)
    ax.plot(N_vals,area_by_N,'.',color=color)
    ax.set_ylabel('Mean area')
    if fname is not None: plt.savefig(fname)
    return ax


T_D_vals = (40.,30.,20.,15.,10.)
repeats = 10
folder = 'stress_measure_compare2/'

pop_size,n_dist,a_mean,p_v,p_r,p_m,p_f = import_all_data(folder,T_D_vals,repeats)
#plot_mean_stresses_by_pop_size(pop_size,((p_v,'virial',1.,palette[0]),),T_D_vals,fname='virial_pressure.pdf')
plot_mean_stresses_by_pop_size(pop_size,((p_f,'(1) full',1.,palette[1]),(p_m,'(2) magnitude',1.,palette[2]),(p_r,'(3) repulsive',1.,palette[4])),T_D_vals,fname='compare_pressures.pdf',ylims=(-500,2000),legend=True)
plot_mean_stresses_by_pop_size(pop_size,((p_v,'(6) virial',1.,palette[0]),(p_f,'(1) full (/6)',6.,palette[1])),T_D_vals,fname='compare_pressures_virial.pdf',ylims=(-80,300),legend=True)

#ax = plot_mean_stresses_by_pop_size(pop_size,((p_v,'virial',1.,palette[0]),),T_D_vals,fname='virial_pressure.pdf')
#ax1=ax.twinx()
#plot_areas_by_pop_size(pop_size,a_mean,T_D_vals,palette[-1],ax1)

