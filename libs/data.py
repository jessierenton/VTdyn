import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def save_mean_area(history,outdir,index=0):
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    filename = '%s/area_mean_%d'%(outdir,index)
    np.savetxt(filename,[np.mean(tissue.mesh.areas) for tissue in history])

def save_area_snapshot(history,outdir,index=0,snap=-1):
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    if snap == -1: filename = '%s/area_snapshot_%s_%d'%(outdir,'f',index)
    else: filename = '%s/area_snapshot_%d_%d'%(outdir,snap,index)
    np.savetxt(filename,history[snap].mesh.areas)
    
def save_force(history,outdir,index=0):
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    wfile = open('%s/%s_%d'%(outdir,'force',index),'w')
    for tissue in history:        
        wfile.write('%.6e \n'%np.linalg.norm(tissue.Force.magnitude(tissue)))
    wfile.close() 

def save_neighbour_distr(history,outdir,index=0,snap=-1):
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    wfilename = '%s/%s_%d'%(outdir,'neigh_distr',index) 
    np.savetxt(wfilename,[np.bincount([len(tissue.mesh.neighbours[i]) for i in range(len(tissue))],minlength=18) for tissue in history],fmt=(['%d']*18))
        
def save_N_cell(history,outdir,index=0):
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    wfilename = '%s/%s_%d'%(outdir,'N_cell',index)  
    np.savetxt(wfilename,[len(tissue) for tissue in history],fmt=('%d'))

def save_N_mutant(history,outdir,index=0):
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    wfilename = '%s/%s_%d'%(outdir,'N_mutant',index)  
    np.savetxt(wfilename,[sum(tissue.properties['mutant']) for tissue in history],fmt=('%d'))

def save_division_times(history,outdir,index=0):
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    divfile = '%s/%s_%d'%(outdir,'division_age',index)
    deathfile = '%s/%s_%d'%(outdir,'death_age',index)
    death_age = np.array([])
    division_age = np.array([])
    for i in range(len(history)-1):
        dead = np.setdiff1d(history[i].cell_ids,history[i+1].cell_ids)
        for id in dead:
            idx = np.where(history[i].cell_ids == id)[0]
            age = history[i].age[idx]
            if id in history[i+1].mother: division_age = np.append(division_age,age)
            else: death_age = np.append(death_age,age)
            
    np.savetxt(deathfile,death_age,fmt=('%d'))
    np.savetxt(divfile,division_age,fmt=('%d'))

def get_cell_history(history,cell_id):
    cell_history = {'area':[],'age':[],'fate':0}
    for tissue in history:
        if cell_id not in tissue.cell_ids and len(cell_history['area']) > 0: 
            if cell_id in tissue.mother: cell_history['fate'] = 1
            break
        elif cell_id in tissue.cell_ids:
            mesh_id = np.where(tissue.cell_ids == cell_id)[0]
            cell_history['area'].append(tissue.mesh.areas[mesh_id])
            cell_history['age'].append(tissue.age[mesh_id])
    return cell_history
    
def save_age_snapshot(history,outdir,index=0,snap=-1):
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    if snap == -1: filename = '%s/age_snapshot_%s_%d'%(outdir,'f',index)
    else: filename = '%s/age_snapshot_%d_%d'%(outdir,snap,index)
    np.savetxt(filename,history[snap].age)
    
def save_mean_age(history,outdir,index=0):
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    filename = '%s/age_mean_%d'%(outdir,index)
    np.savetxt(filename,[np.mean(tissue.age) for tissue in history])
    
def plot_age_dist(readfile):
    ages = np.loadtxt(readfile)
    min,max = np.min(ages),np.max(ages)
    bins=np.bincount(ages)[min:]
    plt.bar(range(min,max),bins,0.4)
        
    
def save_all(history,outdir,index=0):
    save_N_cell(history,outdir,index)
    # save_N_mutant(history,outdir,index)
    save_mean_age(history,outdir,index)
    save_age_snapshot(history,outdir,index)
    save_neighbour_distr(history,outdir,index)
    save_force(history,outdir,index)
    save_area_snapshot(history,outdir,index)
    save_mean_area(history,outdir,index)
    
    
    
    