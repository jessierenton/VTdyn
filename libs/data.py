import os
import numpy as np
import functools
from itertools import product

#library of functions for saving data from a history object (list of tissues)

def memoize(func):
    cache =  dict()

    def memoized_func(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result

    return memoized_func

def counter(func):
    def wrapper(*args,**kwargs):
        wrapper.count += 1
        result = func(*args,**kwargs)
        print wrapper.count
        return result
    wrapper.count = 0
    return wrapper

def save_mean_area(history,outdir,index=0):
    """saves mean area of cells in each tissue"""
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    filename = '%s/area_mean_%d'%(outdir,index)
    np.savetxt(filename,[np.mean(tissue.mesh.areas) for tissue in history])

def save_areas(history,outdir,index=0):
    """saves all areas of cells in each tissue"""
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    filename = '%s/areas_%d'%(outdir,index)
    wfile = open(filename,'w')
    for tissue in history:
        for area in tissue.mesh.areas:        
            wfile.write('%.3e    '%area)
        wfile.write('\n')
    
def save_force(history,outdir,index=0):
    """saves mean magnitude of force on cells in each tissue"""
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    wfile = open('%s/%s_%d'%(outdir,'force',index),'w')
    for tissue in history:        
        wfile.write('%.3e \n'%np.mean(np.sqrt((tissue.Force(tissue)**2).sum(axis=1))))
    wfile.close() 

def save_neighbour_distr(history,outdir,index=0):
    """save neighbour distributions in each tissue"""
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    wfilename = '%s/%s_%d'%(outdir,'neigh_distr',index) 
    np.savetxt(wfilename,[np.bincount([len(tissue.mesh.neighbours[i]) for i in range(len(tissue))],minlength=18) for tissue in history],fmt=(['%d']*18))


        
def save_N_cell(history,outdir,index=0):
    """save number of cells in each tissue"""
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    wfilename = '%s/%s_%d'%(outdir,'N_cell',index)  
    np.savetxt(wfilename,[len(tissue) for tissue in history],fmt=('%d'))

def save_N_mutant(history,outdir,index=0):
    """saves number of mutants in each tissue given by 'mutant' property"""
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    wfilename = '%s/%s_%d'%(outdir,'N_mutant',index)  
    np.savetxt(wfilename,[sum(tissue.properties['mutant']) for tissue in history],fmt=('%d'))

def save_N_mutant_type(history,outdir,index=0):
    """saves number of mutants in each tissue given by 'type' property"""
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    wfilename = '%s/%s_%d'%(outdir,'N_mutant',index)  
    np.savetxt(wfilename,[sum(tissue.properties['type']) for tissue in history],fmt=('%d'))

@memoize
def get_local_density(mesh):
    return mesh.local_density()

def get_cell_history(history,cell_id,area=False,density=False):
    """generate a history for a given cell id with area at each timestep, age at each timestep and 
    fate (reproduction=1 or death=0)"""
    cell_history = {'age':[],'fate':None,'mother':None}
    if area: cell_history['area']=[]
    if density: cell_history['density']=[]
    for tissue in history:
        if cell_id not in tissue.cell_ids:
            if cell_id in tissue.mother: 
                cell_history['fate'] = 1
                break
            elif len(cell_history['age']) > 0: 
                cell_history['fate'] = 0
                break
        elif cell_id in tissue.cell_ids:
            if cell_history['mother'] is None:
                cell_history['mother'] = tissue.mother[np.where(tissue.cell_ids==cell_id)[0][0]]
            mesh_id = np.where(tissue.cell_ids == cell_id)[0][0]
            if area: cell_history['area'].append(tissue.mesh.areas[mesh_id])
            cell_history['age'].append(tissue.age[mesh_id])
            if density: 
                cell_history['density'].append(get_local_density(tissue.mesh)[mesh_id]) 
    return cell_history

def get_cell_histories(history,start=0,area=False,density=False):
    """generate history for all cells (see above get_cell_history)"""
    return [ch for ch in (get_cell_history(history[start:],i,area,density) for i in range(max(history[-1].cell_ids)+1))]

def save_age_of_death(history,outdir,index=0):
    """save cell lifetimes for each cell in history"""
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    cell_histories = np.array(get_cell_histories(history))
    has_fate = np.array([h['fate'] is not None for h in cell_histories])
    cell_histories = cell_histories[has_fate]
    fates = np.array([h['fate'] for h in cell_histories],dtype=bool)
    final_age_d = [cell['age'][-1] for cell in cell_histories[fates]]
    final_age_a = [cell['age'][-1] for cell in cell_histories[~fates]]
    np.savetxt('%s/division_age_%d'%(outdir,index),final_age_d)
    np.savetxt('%s/apoptosis_age_%d'%(outdir,index),final_age_a)
    
def save_ages(history,outdir,index=0):
    """saves all cell ages for each tissue in history"""
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    filename = '%s/ages_%d'%(outdir,index)
    wfile = open(filename,'w')
    for tissue in history:
        for age in tissue.age:        
            wfile.write('%.3e    '%age)
        wfile.write('\n')    
    
def save_mean_age(history,outdir,index=0):
    """save mean age of cells for each tissue in history"""
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    filename = '%s/age_mean_%d'%(outdir,index)
    np.savetxt(filename,[np.mean(tissue.age) for tissue in history])

def save_var_to_mean_ratio_all(history,outdir,s,index=0):
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    initial_types = np.unique()
    if type is None: filename = '%s/var_to_mean_ratio_%d'%(outdir,index)
    else: filename = '%s/var_to_mean_ratio_type_%d_%d'%(outdir,type_,index) 
    np.savetxt(filename,[_calc_var_to_mean(tissue,s,type_) for tissue in history],fmt='%.4f')

    
def _calc_var_to_mean(width,height,centres,s):
    # width,height,centres = tissue.mesh.geometry.width,tissue.mesh.geometry.height,tissue.mesh.centres
    range_=((-width/2.,width/2),(-height/2,height/2))
    n_array=np.histogram2d(centres[:,0],centres[:,1],s,range_)[0]
    return np.var(n_array)/np.mean(n_array)

def calc_density(width,height,centres,s):
    range_=((-width/2.,width/2),(-height/2,height/2))
    return np.histogram2d(centres[:,0],centres[:,1],s,range_)[0]
        
def save_all(history,outdir,index=0):
    save_N_cell(history,outdir,index)
    save_N_mutant(history,outdir,index)
    save_ages(history,outdir,index)
    save_neighbour_distr(history,outdir,index)
    save_force(history,outdir,index)
    try: save_areas(history,outdir,index)
    except: pass
    save_age_of_death(history,outdir,index)
    
    
    
    