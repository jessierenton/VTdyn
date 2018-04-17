import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os



def get_data(foldername,prefix):
    l = len(prefix)
    return {int(f[l+1:]):np.loadtxt(foldername+'/'+f) for f in os.listdir(foldername) if f[:l] == prefix }

def formatting(xlabel,ylabel):
    plt.xlabel(xlabel,size=16)
    plt.ylabel(ylabel,size=16)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.tight_layout()

def line_plot(foldername,prefix,indices,timestep,ylab,interval=1,savename=None,legend=True):
    data = get_data(foldername,prefix)
    time = np.arange(len(data[0]))*timestep
    plt.figure()
    for i in indices:
        plt.plot(time[::interval],data[i][::interval],label=str(i))
    if legend: plt.legend(loc='best')
    formatting('Time (hours)',ylab)
    if savename is None: savename = prefix+'_plot'
    plt.savefig('%s/%s.pdf'%(foldername,savename))

def single_hist_plot(foldername,filename,xlab,savename,bins=10,legend=True):
    data = np.loadtxt(foldername+'/'+filename)
    plt.figure()
    plt.hist(data,bins)
    formatting(xlab, '# cells')
    plt.savefig('%s/%s.pdf'%(foldername,savename))

if __name__ == '__main__':
    folder = 'T_d_data/T_d10'
    line_plot(folder,'force',range(8),0.1,'Mean force',interval=1)
    # single_hist_plot(folder,'division_age_2','Time (hours)', 'divage2',20)