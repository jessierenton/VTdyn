import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def save_mean(history,outdir,key,index=0):
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    wfile = open('%s/%s_mean_%d'%(outdir,key,index),'w')
    for tissue in history:
        wfile.write('%.6f \n'%(np.mean(tissue.by_mesh(key))))
    wfile.close() 

def save_snapshots(history,outdir,key,timestep=1,snapshots='all',index=0):
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    wfile = open('%s/%s_snapshot_%d'%(outdir,key,index),'w')
    if snapshots == 'all': times = xrange(len(history))
    else: 
        N = len(history)
        step = N/snapshots
        times = xrange(0,N,step)
    for i in times:
        wfile.write('%.2f'    %(i*timestep))
        for attr in history[i].by_mesh(key):
            wfile.write('%.6f    '%attr)
        wfile.write('\n')
    wfile.close()


def save_total(history,outdir,key,index=0):
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    wfile = open('%s/%s_mean_%d'%(outdir,key,index),'w')
    for tissue in history:
        wfile.write('%.6f \n'%(np.sum(tissue.by_mesh(key))))
    wfile.close()   

def save_force(history,outdir,index=0):
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    wfile = open('%s/%s_%d'%(outdir,'force',index),'w')
    for tissue in history:        
        wfile.write('%.6e \n'%np.linalg.norm(tissue.force_total()))
    wfile.close() 

def save_neighbour_distr(history,outdir,index=0):
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    wfilename = '%s/%s_%d'%(outdir,'neigh_distr',index)
    np.savetxt(wfilename,[np.bincount([len(tissue.mesh.neighbours(i)) for i in range(len(tissue))],minlength=18) for tissue in history],fmt=(['%d']*18))
        
def save_N_cell(history,outdir,index=0):
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    wfilename = '%s/%s_%d'%(outdir,'N_cell',index)  
    np.savetxt(wfilename,[len(tissue) for tissue in history],fmt=('%d'))
    
def save_all(history,outdir,index=0):
    save_N_cell(history,outdir,index)
    save_neighbour_distr(history,outdir,index)
    
    
    
    
    