import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sort_data as sd
import itertools

sns.set_style('white')
TIMESTEP = 1.0
PKEYS = ['threshold','death_rate','domain_multiplier']
PLABELS = [r'$E_0$',r'$\gamma_D$',r'$m$']

PALETTE = sns.color_palette('colorblind')

def plot(raw_data,data_type,pkeys,filter_pkeys=None,filter_pvalue=None,plabels=None,ax=None,ylabel=True,xlabel=True,step=None,show_error=False):
    ptuples,data,std = sd.get_averaged_data(raw_data,data_type,pkeys,filter_pkeys=filter_pkeys,filter_pvalue=filter_pvalue)
    time = np.arange(0,len(data[0]),TIMESTEP)[::step]
    if ax is None:
        fig,ax = plt.subplots()
    for param,dat,error,color in zip(ptuples,data,std,itertools.cycle(PALETTE)):
        dat,error = dat[::step],error[::step]
        if plabels is None:
            ax.plot(time,dat,color=color)
            
        else:
            ax.plot(time,dat,label=r''.join([r'%s$ = %.1f$, '%(plabel,pvalue) for plabel,pvalue in zip(plabels,param)
                                    if plabel is not None]),color=color)
            ax.legend(loc='lower right')
        if show_error:
            ax.fill_between(time,dat-error/2.,dat+error/2.,alpha=0.2,color=color)
            
    if xlabel:
        ax.set_xlabel('Time (h)')    
    if ylabel:
        if isinstance(ylabel,str):
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(data_type)
    plt.show()

def plot_equilibrium_vs_threshold_over_msquared(data,data_type,pkeys,params,start,ax=None,show_error=False,color=None):
    d,m = params
    ptuples,eqm_data,std = sd.get_equilibrium_data(data,data_type,pkeys)
    thresholds = [threshold for (threshold,) in ptuples]
    if ax is None:
        fig,ax = plt.subplots()
    if not show_error: ax.plot(thresholds,eqm_data,'o',color=color,label=r'$m=%.1f$'%m)
    else:
        ax.errorbar(thresholds,eqm_data,std/2,fmt='o',color=color,label=r'$m=%.1f$'%m)

def plot_equilibriums(raw_data,data_type,start,show_error=False,ylabel=None):
    ptuples = sd.get_parameter_tuples(raw_data,PKEYS)
    death_rates = sorted(set(np.array(ptuples).T[1]))
    m_vals = sorted(set(np.array(ptuples).T[2]))
    fig,axs = plt.subplots(1,len(set(death_rates)))
    for i,(ax,death_rate) in enumerate(zip(axs,death_rates)):
        ax.set_title(r'$\gamma_D=%.2f$'%death_rate)
        for m,color in zip(m_vals,itertools.cycle(PALETTE)):
            plot_equilibrium_vs_threshold_over_msquared(sd.filter_data(raw_data,['death_rate','domain_multiplier'],(death_rate,m)),data_type,
                                            ['threshold'],(death_rate,m),start,ax=ax,show_error=show_error,color=color)
        ax.set_xlabel('Energy threshold')
        ax.set_ylabel(ylabel)
        ax.legend(loc='best')
    return axs


def plot_mean_cycle_length(raw_data,start,show_error=False):
    ptuples = sd.get_parameter_tuples(raw_data,PKEYS)
    death_rates = sorted(set(np.array(ptuples).T[1]))
    m_vals = sorted(set(np.array(ptuples).T[2]))
    fig,axs = plt.subplots(1,len(set(death_rates)))
    for i,(ax,death_rate) in enumerate(zip(axs,death_rates)):
        ax.set_title(r'$\gamma_D=%.2f$'%death_rate)
        for m,color in zip(m_vals,itertools.cycle(PALETTE)):
            filtered_data = sd.filter_data(raw_data,['death_rate','domain_multiplier'],(death_rate,m))
            ptuples,means,std = sd.get_age_distributions(filtered_data,'division_history',['threshold'],start=200)
            thresholds = [threshold for (threshold,) in ptuples]
            if not show_error: ax.plot(thresholds,means,'o',color=color,label=r'$m=%.1f$'%m)
            else:
                ax.errorbar(thresholds,means,std/2,fmt='o',color=color,label=r'$m=%.1f$'%m)
        ax.set_xlabel('Division threshold')
        ax.set_ylabel('Length cell cycle (hours)')
        ax.legend(loc='best')
    return axs
    
    
if __name__ == '__main__':
    raw_data = sd.load_data('CIP_data_fixed_cc_rates')
    # plot(raw_data,'pop_size',PKEYS,filter_pkeys=('death_rate','domain_multiplier'),filter_pvalue=(0.01,1.3),show_error=True,step=10,plabels=PLABELS)
    axs = plot_equilibriums(raw_data,'pop_size',200,show_error=False,ylabel='pop_size')
    # axs = plot_mean_cycle_length(sd.filter_data(raw_data,'death_rate',[0.01,0.04]),200,show_error=False)