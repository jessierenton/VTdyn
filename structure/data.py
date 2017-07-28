import os
import numpy as np

def save_mean(history,outdir,key,index=0):
    if not os.path.exists(outdir): # if the folder doesn't exist create it
         os.makedirs(outdir)
    wfile = open('%s/%s_mean_%d'%(outdir,key,index),'w')
    for tissue in history:
        wfile.write('%.6f \n'%(np.mean(tissue.by_mesh(key))))
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
    np.savetxt(wfilename,[np.bincount([len(tissue.mesh.neighbours(i)) for i in range(tissue.N_real)],minlength=18) for tissue in history],fmt=(['%d']*18))
        
    

def plot_averages(history,key,timestep):
    None