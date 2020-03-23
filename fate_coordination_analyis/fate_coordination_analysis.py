import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import json
from scipy import stats

sns.set_style("white")
PALETTE = sns.color_palette()

def readfromfile(filename):
    with open(filename) as jsonfile:
        data = json.load(jsonfile)
    nn_data_keys = ["nn","nextnn"]
    df_fates = get_fates_dataframe({key:value for key,value in data["cell_histories"].items() 
                                if key not in nn_data_keys})
    nn_data = {key:value for key,value in data["cell_histories"].items() 
                    if key in nn_data_keys}
    return df_fates,nn_data,data["parameters"]

def get_fates_dataframe(fates_data):
    position = fates_data.pop("position")
    fates_data["xcoord"] = [coords[0] for coords in position]
    fates_data["ycoord"] = [coords[1] for coords in position]
    return pd.DataFrame(fates_data)
#
# def get_nn_dataframe(nn_data):
#     max_nn_length = max(len(nn) for nn in nn_data["nn"])
#     max_next_nn_length = max(len(nn) for nn in nn_data["nextnn"])
#     index_tuples = [("neighbours",i) for i in range(max_nn_length)]
#     index_tuples += [("next_neighbours",i) for i in range(max_next_nn_length)]
#     index = pd.MultiIndex.from_tuples(index_tuples)
#     data = [np.hstack((np.pad(neighbours,(0,max_nn_length-len(neighbours))),np.pad(next_neighbours,(0,max_next_nn_length-len(next_neighbours)))))
#                 for neighbours,next_neighbours in zip(nn_data["nn"],nn_data["nextnn"])]
#     return pd.DataFrame(data,columns=index)
     

def timeslice(df,startime,stoptime):
    """return dataframe with entries within the timeinterval"""
    if startime is None:
        startime = 0
    if stoptime is None: 
        stoptime = np.inf
    return df[((df.time>=startime) & (df.time<=stoptime))]

def random_coords():
    """return pair of random floats within the domain"""
    return np.array((rand.rand()*width-width/2.,rand.rand()*height-height/2.))
    
def check_in_box(coords,boxcoords,boxwidth):
    """check if the coords are inside the box"""
    cases = np.array(((0,0),(width,0),(0,height),(width,height)))
    for case in cases:
        if np.all(boxcoords+case<coords) and np.all(boxcoords+case+np.repeat(boxwidth,2)>coords):
            return True
    return False

def check_cases(allcoords,boxcoords,boxwidth,cases):
    """helper function to check list of coords are in box accounting for periodic boundary conditions"""
    casesbool = np.array([np.all(np.array((np.all(boxcoords+case<allcoords,axis=1),np.all(boxcoords+case+np.repeat(boxwidth,2)>allcoords,axis=1))),axis=0) 
            for case in cases])
    return np.any(casesbool,axis=0)

def check_in_box_all(df,boxcoords,boxwidth):
    """check list of coords are in box accounting for periodic boundary conditions. returns boolean array"""
    allcoords = np.array((df["xcoord"],df["ycoord"])).T
    cases = np.array(((0,0),(width,0),(0,height),(width,height)))
    return check_cases(allcoords,boxcoords,boxwidth,cases)

def fate_events_in_box(df,boxcoords,boxwidth,startime=None,stoptime=None):
    """returns (divisions,deaths) that have occured inside box between timestart and timestop"""
    if startime is not None or stoptime is not None:
        df = timeslice(df,startime,stoptime)
    fates = df["divided"][check_in_box_all(df,boxcoords,boxwidth)]
    return sum(fates),len(fates)-sum(fates)

def net_growth_in_box(df,boxcoords,boxwidth,startime=None,stoptime=None):
    divisions,deaths = fate_events_in_box(df,boxcoords,boxwidth,startime,stoptime)
    return divisions - deaths

def plot_box(ax,boxcoords,boxwidth):
    """add box onto plot accounting for periodic bcs"""
    box = patches.Rectangle(boxcoords,boxwidth,boxwidth,linewidth=1,edgecolor="grey",facecolor="none")
    ax.add_patch(box)
    boxcoords2 = boxcoords+np.repeat(boxwidth,2)
    if boxcoords2[0]>width/2:
        box = patches.Rectangle(boxcoords-np.array((width,0)),boxwidth,boxwidth,linewidth=1,edgecolor="grey",facecolor="none")
        ax.add_patch(box)
    if boxcoords2[1]>height/2:
        box = patches.Rectangle(boxcoords-np.array((0,height)),boxwidth,boxwidth,linewidth=1,edgecolor="grey",facecolor="none")
        ax.add_patch(box)
    if np.all(boxcoords2>domain_size/2):
        box = patches.Rectangle(boxcoords-domain_size,boxwidth,boxwidth,linewidth=1,edgecolor="grey",facecolor="none")
        ax.add_patch(box)

def plot_fate_events(df,width,height,boxcoords=None,boxwidth=None,startime=None,stoptime=None):
    """plot fate events"""
    if startime is not None or stoptime is not None:
        df = timeslice(df,startime,stoptime)
    g = sns.lmplot(x="xcoord",y="ycoord",data=df,hue="divided",fit_reg=False,markers=".")
    if boxwidth is not None:
        if boxcoords is None:
            boxcoords = random_coords()
        ax = g.axes[0][0]
        plot_box(ax,boxcoords,boxwidth)
    plt.axis('scaled')
    ax.set_xlim(-width/2,width/2)
    ax.set_ylim(-height/2,height/2)
    return g    

def plot_fate_events(filename,boxcoords=None,w=None,startime=None,stoptime=None):
    df_fates,nn_data,parameters = readfromfile(filename)
    width = parameters["width"]
    height = np.sqrt(3)/2*width
    domain_size = np.array((width,height))
    if w is not None: 
        cell_seperation = width/10.
        boxwidth = w*cell_seperation
    g = plot_fate_events(df,width,height,boxcoords=boxcoords,boxwidth=boxwidth,startime=startime,stoptime=stoptime)
    return g

def neighbour_behaviours(df,neighbours,hours_from_event,cell_id=None,idx=None):
    """for a given cell_id or dataframe index return the number of neighbours when
        that cell divides/dies, the number of neighbours which divide (in the 
        timeframe) and number of neighbours which 
        died (in the time frame)"""
    if idx is None:
        idx = df.index[df.cell_ids[cell_id]]
    event_time = df.time[idx]
    number_neighbours = len(neighbours[idx])
    df_neighbours = df[df["cell_ids"].isin(neighbours[idx])]
    df_neighbours = timeslice(df_neighbours,event_time,event_time+hours_from_event)
    total_events,divided = len(df_neighbours),sum(df_neighbours.divided)
    return number_neighbours,divided,total_events-divided

def neighbour_fate_balance(df,neighbours,hours_from_event,cell_id=None,idx=None):
    """for a given cell_id or dataframe index and time from event return the 
    difference in number of divided neighbours vs dead neighbours in that time"""
    number_neighbours,divided,dead = neighbour_behaviours(df,neighbours,hours_from_event,cell_id=cell_id,idx=idx)
    return divided - dead

def neighbour_fate_balance_mean_sem(df,neighbours,hours_from_event,cell_ids=None,indices=None):
    """from the population data in df and neighbours calculate the fate balance (difference)
    between neighbour divisions and deaths for the given cell_ids or df indices, in
    the time interval given. return the mean and sem"""
    if cell_ids is not None:
        data = [neighbour_fate_balance(df,neighbours,hours_from_event,cell_id=cell_id) 
            for cell_id in cell_ids]
    elif indices is not None:
        data = [neighbour_fate_balance(df,neighbours,hours_from_event,idx=idx) 
            for idx in indices] 
    return np.mean(data),stats.sem(data)
    
def neighbour_fate_balance_stats(df,neighbours,divided,timeintervals,startime=None):
    """return the population fate balance statistics (mean and sem) for given timeintervals
    for divided cells (divided=True) or dead cells (divided=False)"""
    df = timeslice(df,startime,df.time.max()-timeintervals[-1])
    index = df[df.divided==divided].index
    return np.array([neighbour_fate_balance_mean_sem(df,neighbours,hours_from_event,indices=index)
        for hours_from_event in timeintervals]), len(index)

def get_neighbour_fate_balance_stats_from_file(filename,divided,timeintervals,startime=None):
    df,nn_data,parameters = readfromfile(filename)
    neighbours = nn_data["nn"]
    return neighbour_fate_balance_stats(df,neighbours,divided,timeintervals,startime)
    
def get_fate_balance_stats_from_files(filenames,divided,timeintervals,startime=None):
    data = [get_neighbour_fate_balance_stats_from_file(f,divided,timeintervals,startime) for f in filenames]
    fate_balance_stats = [d[0] for d in data]   
    number_data = [d[1] for d in data]
    return fate_balance_stats,number_data

def plot_multi_fate_balance_stats(fate_balance_stats_div,fate_balance_stats_dead,labels,timeintervals,palette=PALETTE,error=True):
    for fb_death_stats,fb_div_stats,label,color in zip(fate_balance_stats_div,fate_balance_stats_dead,labels,palette):
        if error:
            yerr_div,yerr_death = fb_div_stats[:,1]/2,fb_death_stats[:,1]/2
        else:
            yerr_div,yerr_death = None,None
        plt.errorbar(timeintervals,fb_div_stats[:,0],yerr=yerr_div,
            color=color,marker='o',ls='',label=label)
        plt.errorbar(timeintervals,fb_death_stats[:,0],yerr=yerr_death,
            color=color,marker='d',ls='')
        plt.legend()
        

if __name__ == "__main__":    
    rand=np.random.RandomState()
    filenames = [f"fate_test_{runtype}_000.json" for runtype in ("db","dc","cip")]

    timeintervals = 24*np.arange(0.,5.5,0.5)
    fb_deaths,num_deaths = get_fate_balance_stats_from_files(filenames,False,timeintervals)
    fb_divs,num_divs = get_fate_balance_stats_from_files(filenames,True,timeintervals)