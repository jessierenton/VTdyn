import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import json
from scipy import stats
import os,sys
from multiprocessing import Pool,cpu_count
from functools import partial
from itertools import chain

PALETTE = sns.color_palette('colorblind')
T_TOL = 1e-4
TEXTWIDTH = 5.

def readfromfile(filename):
    with open(filename) as jsonfile:
        data = json.load(jsonfile)
    nn_data_keys = ["nn","nextnn"]
    exclude = nn_data_keys +['mean_separation','mean_distance']
    df_fates = get_fates_dataframe({key:value for key,value in data["cell_histories"].items() 
                                if key not in exclude})
    nn_data = {key:value for key,value in data["cell_histories"].items() 
                    if key in nn_data_keys}
    try:
        return df_fates,nn_data,data["parameters"]
    except KeyError:
        return df_fates,nn_data,None
               
    
def get_fates_dataframe(fates_data):
    position = fates_data.pop("position")
    fates_data["xcoord"] = [coords[0] for coords in position]
    fates_data["ycoord"] = [coords[1] for coords in position]
    return pd.DataFrame(fates_data)

def timeslice(df,startime,stoptime):
    """return dataframe with entries within the timeinterval"""
    if startime is None:
        startime = 0
    if stoptime is None: 
        stoptime = np.inf
    return df[((df.time>=startime) & (df.time<=stoptime))]

def imbalance(t_f,t,divided):
    """imbalance function returns +1 if cell has divided by time t, -1 if cell has died,
        0 if no fate decision.
    t_f = time of fate decision, divided = fate decision (True/False)"""
    if t<t_f:
        return 0
    else:
        if divided:
            return 1
        else:
            return -1

def imbalance_timesequence(t_f,timesteps,divided):
    """returns imbalance function for a given sequence of times"""
    np.array([imbalance(t_f,t,divided) for t in timesteps])

def net_imbalance_nn(t_f,t,nn_t_f,nn_divided):
    """net imbalance of the nearest neighbours after cell has made a fate decision until time t_max for given timestep.
    t_f = time of fate decision, nn_t_f = time of nearest neighbour fate decisions (list),
    nn_divided = fate decisions for nearest neighbours (bool list)"""
    return np.sum([imbalance(t_f1,t+t_f,divided) 
        for t_f1,divided in zip(nn_t_f,nn_divided)],axis=0)  
    
def background_imbalance(t1,t2,t_f_list,divided_list):
    """calculate the background imbalance of divisions and deaths in time period [t1,t2].
        t_f_list = list of fate decision times for cells in the tissue at time t1,
        divided_list = list of fate decisions for cells in the tissue at t1"""
    N = len(t_f_list)
    return 1/N*sum(imbalance(t_f,t2,divided) 
                    for t_f,divided in zip(t_f_list,divided_list))

def cells_in_tissue(df,t):
    #add T_TOL to exclude new divided cells
    return df[(((df.time-df.age+T_TOL)<t)&(t<=df.time))]

def get_background_imbalance(df,neighbours,t1,t2):
    """find background imbalance from dataframe"""
    if np.isclose(t1,t2): 
        return 0
    df_cells_t1 = cells_in_tissue(df,t1)
    return background_imbalance(t1,t2,list(df_cells_t1.time),list(df_cells_t1.divided))

def get_background_imbalance_over_time(df,neighbours,startime,stoptime,timesteps,fate):
    df_snapshot = timeslice(df,startime=startime,stoptime=stoptime-timesteps[-1])
    index = df_snapshot[df_snapshot.divided==fate_is_division(fate)].index
    t_f_vals = list(df_snapshot.loc[index].time)
    return [[get_background_imbalance(df,neighbours,t_f,t+t_f) for t in timesteps] for i,t_f in zip(index,t_f_vals)]
    
def get_net_imbalance_individual_contribution(df,neighbours,i,t,background_corrected=True):
    """return contribution from cell i to detrended net imbalance for time interval t"""
    t_f = df.loc[i].time
    nn_data = df.loc[df.cell_ids.isin(neighbours[i]),['time','divided']]
    nn_t_f = list(nn_data.time)
    nn_divided = list(nn_data.divided)
    nn_imbalance = net_imbalance_nn(t_f,t,nn_t_f,nn_divided)
    if background_corrected:
        nn_imbalance -= len(nn_t_f)*get_background_imbalance(df,neighbours,t_f,t+t_f)
    return nn_imbalance

def get_net_imbalance_mean_sem(df,neighbours,index,t,background_corrected=True):
    all_contributions = [get_net_imbalance_individual_contribution(df,neighbours,i,t,background_corrected) for i in index]
    return np.mean(all_contributions),stats.sem(all_contributions)

def fate_is_division(fate):
    if fate == "division":
        return True
    elif fate == "death":
        return False

def net_imbalance(df,neighbours,startime,stoptime,timesteps,fate,background_corrected=True):
    df_snapshot = timeslice(df,startime=startime,stoptime=stoptime-timesteps[-1])
    index = df_snapshot[df_snapshot.divided==fate_is_division(fate)].index
    return np.array([get_net_imbalance_mean_sem(df,neighbours,index,t,background_corrected) for t in timesteps])

def get_fate_balance_stats_from_file(filename,fate,timesteps,startime=None,stoptime=None,background_corrected=True):
    df,nn_data,parameters = readfromfile(filename)
    neighbours = nn_data["nn"]
    data = net_imbalance(df,neighbours,startime,stoptime,timesteps,fate,background_corrected)
    print(f"complete {filename} {fate}")
    return data
    
def get_fate_balance_stats_from_files(filenames,fate,timesteps,startime=None,stoptime=None,background_corrected=True):
    fate_balance_stats = [get_fate_balance_stats_from_file(f,fate,timesteps,startime,stoptime,background_corrected) for f in filenames]
    return fate_balance_stats
    
def read_params_from_filename(filename):
    """return [d-to-b-ratio, alpha, run#]"""
    return [float(filename[2:5]),float(filename[7:11]),int(filename[12:15])]
    
def get_fate_balance_stats_from_CIPfiles(readir,timesteps,startime=None,stoptime=None,background_corrected=True,savename=None):
    filenames = [filename for filename in os.listdir(readir) 
                    if filename[-5:]=='.json']
    parameters = [read_params_from_filename(filename) for filename in filenames]
    filenames = [readir+filename for filename in filenames]
    df = [{'db':params[0],'alpha':params[1],'run':params[2],'time':time,'mean':mean,'sem':sem,'fate':'division'} 
            for filename,params in zip(filenames,parameters) 
                for time,(mean,sem) in zip(timesteps,get_fate_balance_stats_from_file(filename,'division',timesteps,startime,stoptime,background_corrected))]
    df += [{'db':params[0],'alpha':params[1],'run':params[2],'time':time,'mean':mean,'sem':sem,'fate':'death'} 
            for filename,params in zip(filenames,parameters) 
                for time,(mean,sem) in zip(timesteps,get_fate_balance_stats_from_file(filename,'death',timesteps,startime,stoptime,background_corrected))]
    df = pd.DataFrame(df)
    if savename is not None:
        df.to_csv(savename)
    return df
    
def get_fate_balance_stats_from_CIPfiles_parallel(readir,timesteps,startime=None,stoptime=None,background_corrected=True,savename=None):
    filenames = [filename for filename in os.listdir(readir) 
                    if filename[-5:]=='.json']
    parameters = [read_params_from_filename(filename) for filename in filenames]
    filenames = [readir+filename for filename in filenames]
    args1 = ((filename,'division',timesteps,startime,stoptime,background_corrected) for filename in filenames)
    args2 = ((filename,'death',timesteps,startime,stoptime,background_corrected) for filename in filenames)
    pool = Pool(cpu_count()-1,maxtasksperchild=1000)
    data = pool.starmap(get_fate_balance_stats_from_file,chain(args1,args2))
    fates = ['division']*len(filenames)+['death']*len(filenames)
    df = [{'db':params[0],'alpha':params[1],'run':params[2],'time':time,'mean':mean,'sem':sem,'fate':fate} 
            for params,timeseries_data,fate in zip(parameters*2,data,fates) for time,(mean,sem) in zip(timesteps,timeseries_data)]
    df = pd.DataFrame(df)
    if savename is not None:
        df.to_csv(savename)
    return df

def plot_multi_fate_balance_stats(fate_balance_stats_div,fate_balance_stats_dead,labels,timesteps,palette=PALETTE,error=False):
    for fb_death_stats,fb_div_stats,label,color in zip(fate_balance_stats_div,fate_balance_stats_dead,labels,palette):
        if error:
            yerr_div,yerr_death = fb_div_stats[:,1]/2,fb_death_stats[:,1]/2
        else:
            yerr_div,yerr_death = None,None
        plt.errorbar(timesteps,fb_div_stats[:,0],yerr=yerr_div,
            color=color,marker='o',ls='',label=label)
        plt.errorbar(timesteps,fb_death_stats[:,0],yerr=yerr_death,
            color=color,marker='d',ls='')
        plt.legend()

def set_time_units(df,time_units):
    if time_units == 'hours':
        return df
    if time_units == "days":
        df['time'] = df['time']/24 
        return df

def plot_fate_balance_df(df=None,filename=None,error=False,run=None,time_units='days',savename=None,palette=PALETTE):
    sns.set_style("white")
    sns.set_palette(palette)
    if filename is not None:
        df = pd.read_csv(filename,index_col=0)
    df = set_time_units(df,time_units)
    if run is not None:
        df = df[df.run==run]
    if error:
        x_ci="sd"
    else:
        x_ci=None
    g = sns.FacetGrid(df,row='fate',col='alpha',hue='db',row_order=['death','division'],height=TEXTWIDTH/3,aspect=1,hue_kws={'marker':['d','v','o','^','P','s']}) 
    g.map(sns.regplot,'time','mean',x_estimator=np.mean,x_ci=x_ci,fit_reg=False,scatter_kws={'s':6})
    g.set_xlabels('Days from event')
    g.axes[0][0].set_ylabel('Net imbalance of\n nearest neighbours\n around dead cell',labelpad=6.0)
    g.axes[1][0].set_ylabel('Net imbalance of\n nearest neighbours\n around divided cell',labelpad=6.0)
    g.add_legend(title=r'$\lambda /\gamma$')
    g.set_titles(template=r'$\alpha = {col_name}$')
    g.set(xticks=[0,2,4,6])
    plt.subplots_adjust(left=0.17,bottom=0.16,right=0.87,top=0.90,wspace=0.17,hspace=0.31)
    if savename is not None:
        g.savefig(savename,dpi=450)
    return g   

def plot_fate_balance_df_compare_runs(df=None,filename=None,error=False,fate="death"):
    if filename is not None:
        df = pd.read_csv(filename,index_col=0)
    df = df[df.fate==fate]
    df['sem/2']=df['sem']/2
    g=sns.FacetGrid(df,row='alpha',hue='run',col='db')
    g.map(plt.errorbar,'time','mean','sem/2')

if __name__ == "__main__":        
    readir = 'CIP_fate_statistics3/'
    timeintervals = 24*np.arange(0.,7.5,0.5)
    # df = get_fate_balance_stats_from_CIPfiles_parallel(readir,timeintervals,100,6000,True,'CIP_fate_balance_background_corrected3')
    # g = plot_fate_balance_df(filename='fate_balance_data/CIP_fate_balance_not_background_corrected3',savename='CIP_imbalance_nbc.pdf')
    # plot_fate_balance_df_compare_runs(df=None,filename='CIP_fate_balance_background_corrected_new',error=False,fate="death")
    readir = 'dcdb_fate_statistics/'
    filenames = [readir+f for f in os.listdir(readir) if f[-4:]=='json']
    dfs_bc = get_fate_balance_stats_from_files(filenames,'death',timeintervals,startime=100,stoptime=6000,background_corrected=True)
    dfs_bc = get_fate_balance_stats_from_files(filenames,'division',timeintervals,startime=100,stoptime=6000,background_corrected=True)
    dfs_nbc = get_fate_balance_stats_from_files(filenames,'death',timeintervals,startime=100,stoptime=6000,background_corrected=False)
    dfs_nbc = get_fate_balance_stats_from_files(filenames,'division',timeintervals,startime=100,stoptime=6000,background_corrected=False)