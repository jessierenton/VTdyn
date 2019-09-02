import json
import numpy as np
import os

def memoize(func):
    cache = dict()

    def memoized_func(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return memoized_func

@memoize
def load_data(indir):
    return [json.load(open(indir+'/'+f)) for f in os.listdir(indir) if f[-4:]=='json']
    
def filter_data(raw_data,pkeys,pvalue):
    if isinstance(pkeys,str):
        try: 
            return [d for d in raw_data if d['parameters'][pkeys] in pvalue]
        except TypeError:
            return [d for d in raw_data if d['parameters'][pkeys]==pvalue]
        
    else:
        return [d for d in raw_data if filter_check_multiple(d['parameters'],pkeys,pvalue)]

def filter_check_multiple(param_dict,pkeys,pvalue):
    for key,value in zip(pkeys,pvalue):
        try:
            if param_dict[key] not in value:
                return False
        except TypeError:
            if param_dict[key] != value:
                return False
    return True

def average_equilibrium(raw_data,data_type,pkeys,pvalue,start=0):
    averaged = average_data(raw_data,data_type,pkeys,pvalue)[0][start:]
    return np.mean(averaged,axis=0),np.std(averaged,axis=0)

def get_equilibrium_data(raw_data,data_type,pkeys,filter_pkeys=None,filter_pvalue=None,start=0): 
    if filter_pkeys is not None:
        raw_data = filter_data(raw_data,filter_pkeys,filter_pvalue)
    ptuples = get_parameter_tuples(raw_data,pkeys)
    pdata   = np.array([average_equilibrium(raw_data,data_type,pkeys,pvalue,start=start) for pvalue in ptuples]).T
    return ptuples,pdata[0],pdata[1]

def age_distribution(raw_data,data_type,pkeys,pvalue,start=0):
    """takes data type extrusion_history or division_history"""
    filtered = filter_data(raw_data,pkeys,pvalue)
    cycle_histories = [np.array(data[data_type]) for data in filtered]
    cycle_lengths = [cycle_history[:,1][cycle_history[:,2]>start] for cycle_history in cycle_histories]
    return np.hstack(cycle_lengths)

def get_age_distributions(raw_data,data_type,pkeys,filter_pkeys=None,filter_pvalue=None,start=0,return_full=False):
    """takes data type extrusion_history or division_history"""
    if filter_pkeys is not None:
        raw_data = filter_data(raw_data,filter_pkeys,filter_pvalue)
    ptuples = get_parameter_tuples(raw_data,pkeys)
    pdata = [age_distribution(raw_data,data_type,pkeys,pvalue,start) for pvalue in ptuples]
    if return_full: 
        return ptuples,pdata
    else: 
        pdata = np.array([(np.mean(pd),np.std(pd)) for pd in pdata]).T
        return ptuples,pdata[0],pdata[1]

def average_data(raw_data,data_type,pkeys,pvalue):
    filtered = filter_data(raw_data,pkeys,pvalue)
    filtered_data_type = np.array([data[data_type] for data in filtered])
    return np.mean(filtered_data_type,axis=0),np.std(filtered_data_type,axis=0)
    
def get_parameter_tuples(raw_data,pkeys):
    ptuples = [tuple([data['parameters'][pk] for pk in pkeys]) for data in raw_data]
    ptuples = list(set(ptuples))
    ptuples.sort(key=lambda t: [t[i] for i in range(len(pkeys))])
    return ptuples
        
def get_averaged_data(raw_data,data_type,pkeys,filter_pkeys=None,filter_pvalue=None):
    if filter_pkeys is not None:
        raw_data = filter_data(raw_data,filter_pkeys,filter_pvalue)
    ptuples = get_parameter_tuples(raw_data,pkeys)
    pdata = np.array([np.array(average_data(raw_data,data_type,pkeys,pvalue)).T for pvalue in ptuples]).T
    return ptuples,pdata[0].T,pdata[1].T # returns parameters,means,std