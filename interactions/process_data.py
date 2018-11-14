import numpy as np
import os

datdir = 'raw_data2'

def load_all_data(datdir):
    data_files = [datdir+'/'+f for f in os.listdir(datdir) if not f.startswith('.')]
    return np.vstack([np.loadtxt(file,dtype=float) for file in data_files]).T

def make_dirs(dirnames):
    for dirname in dirnames:
        if not os.path.exists(dirname): # if the outdir doesn't exist create it
             os.makedirs(dirname)
    return dirnames

def sort_all_data(data):    
    data = data[:,np.argsort(data[0])]
    data_by_n = []
    for n in range(100):
        n_locs = np.where(data[0]==n)[0]
        if len(n_locs)!=0:
            split_index = np.max(n_locs)+1
            data_n,data = np.split(data,(split_index,),axis=1)
            data_by_n.append(data_n)
    return data_by_n

def save_sorted(outdirs,data_by_n):        
    for data in data_by_n:
        n = data[0][0]
        data = data[1:]
        for i,outdir in enumerate(outdirs):
            if i < 2:
                np.savetxt(outdir+'/n_%d'%n,data[i],fmt='%3d')
            else: 
                np.savetxt(outdir+'/n_%d'%n,data[i],fmt='%.6f')
        

def main():    
    data_by_n = sort_all_data(load_all_data(datdir))
    outdirs = make_dirs(('ints_CC','ints_CD','wints_CC','wints_CD'))
    save_sorted(outdirs,data_by_n)

main()