import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sort_data as sd
import itertools

sns.set_style('white')
TIMESTEP = 1.0
PLABELS = {'threshold':r'$E_0$','death_rate':r'$\lambda$','domain_multiplier':r'$m$'}
PALETTE = sns.color_palette('colorblind')

def generate_labels(label,pkeys=None,pvalue=None,plabels=PLABELS):
    if label is None: return None
    elif isinstance(label,str): return label
    else:
        if isinstance(pkeys,str):
            if plabels is None:
                label = pkeys+r'$ = %.3f$'%pvalue
            else: 
                label = plabels[pkeys]+r'$ = %.3f$'%pvalue
        else: 
            if plabels is None:
                label = r''.join([key+r'$ = %.3f; $'%value for key,value in zip(pkeys,pvalue)])
            else: 
                label = r''.join([plabels[key]+r'$ = %.3f; $'%value for key,value in zip(pkeys,pvalue)])
    return label
    
def set_label(line,label,pkeys=None,pvalue=None,plabels=PLABELS):
    label = generate_labels(label,pkeys,pvalue,plabels)
    line.set_label(label)

def set_title(ax,title,pkeys=None,pvalue=None,plabels=PLABELS):
    label = generate_labels(title,pkeys,pvalue,plabels)
    ax.set_title(label)    

def plot(raw_data,data_type,pkeys,filter_pkeys=None,filter_pvalue=None,label=True,title=True,ax=None,ylabel=True,xlabel=True,step=None,show_error=False,plabels=PLABELS):
    ptuples,data,std = sd.get_averaged_data(raw_data,data_type,pkeys,filter_pkeys=filter_pkeys,filter_pvalue=filter_pvalue)
    time = np.arange(0,len(data[0]),TIMESTEP)[::step]
    if ax is None:
        fig,ax = plt.subplots()
    for pvalue,dat,error,color in zip(ptuples,data,std,itertools.cycle(PALETTE)):
        dat,error = dat[::step],error[::step]
        line, = ax.plot(time,dat,color=color)
        set_label(line,label,pkeys,pvalue,plabels)
        if show_error:
            ax.fill_between(time,dat-error/2.,dat+error/2.,alpha=0.2,color=color)
    if label is not None: 
        ax.legend(loc='lower right')          
    if xlabel:
        ax.set_xlabel('Time (h)')    
    if ylabel:
        if isinstance(ylabel,str):
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(data_type)
    set_title(ax,title,filter_pkeys,filter_pvalue,plabels=plabels)       
    plt.show()

def plot_equilibrium_vs_param(raw_data,data_type,pkey,start,filter_pkeys=None,filter_pvalue=None,ax=None,
                                show_error=False,color=None,label=None,logx=False,logy=False):
    data = sd.filter_data(raw_data,filter_pkeys,filter_pvalue)
    params,eqm_data,std = sd.get_equilibrium_data(data,data_type,pkey)
    if eqm_data.ndim==2:
        eqm_data,std = eqm_data[0],std[0]
    if logx: params = np.log(np.array(params))
    if logy: eqm_data,std = np.log(eqm_data),np.log(std)
    if ax is None:
        fig,ax = plt.subplots()
    if not show_error: line, = ax.plot(params,eqm_data,'o',color=color)
    else:
        line = ax.errorbar(params,eqm_data,std/2,fmt='o',color=color)[0]
    set_label(line,label,pkeys=filter_pkeys,pvalue=filter_pvalue)
    
def plot_equilibriums_varying_3_params(raw_data,data_type,pkeys,start,show_error=False,ylabel=None,logx=False,logy=False):
    ptuples = sd.get_parameter_tuples(raw_data,pkeys)
    death_rates = sorted(set(np.array(ptuples).T[1]))
    m_vals = sorted(set(np.array(ptuples).T[2]))
    fig,axs = plt.subplots(1,len(set(death_rates)))
    for i,(ax,death_rate) in enumerate(zip(axs,death_rates)):
        ax.set_title(r'$\gamma_D=%.2f$'%death_rate)
        for m,color in zip(m_vals,itertools.cycle(PALETTE)):
            plot_equilibrium_vs_param(raw_data,data_type,'threshold',start,['death_rate','domain_multiplier'],
                                    (death_rate,m),ax=ax,show_error=show_error,color=color,label=r'$m=%.1f$'%m,logx=logx,logy=logy)
        ax.set_xlabel('Energy threshold')
        ax.set_ylabel(ylabel)
        ax.legend(loc='best')
    return axs

def plot_equilibriums_same_axis(raw_data,data_type,start,show_error=False,ylabel=None,logx=False,logy=False):
    death_rates = sd.get_parameter_tuples(raw_data,'death_rate')
    fig,ax = plt.subplots()
    for death_rate,color in zip(death_rates,itertools.cycle(PALETTE)):
        plot_equilibrium_vs_param(raw_data,data_type,'threshold',start,'death_rate',death_rate,ax=ax,
                show_error=show_error,color=color,label=True,logx=logx,logy=logy)
        plt.legend(loc='best')
        ax.set_xlabel('Energy threshold')
        ax.set_ylabel(ylabel)
        
def plot_mean_cycle_length(raw_data,start,pkeys,show_error=False):
    ptuples = sd.get_parameter_tuples(raw_data,pkeys)
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
    pkeys = ['threshold','death_rate','domain_multiplier']
    raw_data = sd.load_data('CIP_data_fixed_cc_rates')
    # plot(raw_data,'pop_size','threshold',filter_pkeys=('death_rate','domain_multiplier'),filter_pvalue=(0.01,1.3),show_error=True,step=10,label=True)
    axs = plot_equilibriums_varying_3_params(raw_data,'pop_size',pkeys,200,show_error=False,ylabel='pop_size')
    # axs = plot_mean_cycle_length(sd.filter_data(raw_data,'death_rate',[0.01,0.04]),pkeys,200,show_error=False)