import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style('white')
    
def load_data(folder,dtype=float):    
    return [np.loadtxt(folder+'/n_%d'%n,dtype=dtype) for n in range(1,100)]

def heat_data(data):
    max_val = int(max(max(data_n) for data_n in data))+1
    heat_data = np.array([np.bincount(data_n,minlength=max_val) for data_n in data],dtype=float)
    return (heat_data/np.repeat(np.sum(heat_data,axis=1),heat_data.shape[1]).reshape(heat_data.shape)).T

def plot_heat_map(folder):
    data = load_data('ints_CC',int)
    fig,ax = plt.subplots()
    heat_data = heat_data(data)
    ax = sns.heatmap(heat_data,xticklabels=20,yticklabels=100,robust=True)
    plt.savefig('heat_map.pdf')

def plot_mean_std(data,ax,label=None,mult=1.,show_error=True):
    x = np.arange(1,100)
    mean = np.array([np.mean(data_n)*mult for data_n in data])
    if show_error: std = np.array([np.std(data_n)*mult for data_n in data])
    ax.plot(x,mean,label=label)
    if show_error: ax.fill_between(x,mean-std,mean+std,alpha=0.3) 
    
def plot_Lambda_CC(data,ax,label=None,mult=1.,show_error=True,fit=False,discrete=False):
    x=np.arange(0.01,1.,0.01)
    Lambda_CC = np.array([np.mean(data_n)/(i+1.) for i,data_n in enumerate(data)])
    if discrete and not show_error: ax.plot(x,Lambda_CC,label=label,ls='',marker='.') 
    if not discrete: ax.plot(x,Lambda_CC,label=label)
    if show_error:
        std = np.array([np.std(data_n)/(i+1.) for i,data_n in enumerate(data)])
        if discrete:
            ax.errorbar(x,Lambda_CC,yerr=std/2,label=label,ls='',marker='.',elinewidth=0.7)
        else: 
            ax.fill_between(x,Lambda_CC-std/2,Lambda_CC+std/2,alpha=0.3) 
    if fit and normalised:
        from scipy.optimize import curve_fit
        f = lambda x,a,b,c: a/x**2+b/x+(1-c)+c*x
        popt,pcov=curve_fit(f, x, Lambda_CC)
        ax.plot(x,f(x,*popt))
        return popt
        
        
def interactions_plot():
    data = load_data('ints_CC'),load_data('ints_CD')
    fig,ax = plt.subplots()
    plot_mean_std(data_CC,ax,'C-C interactions')
    plot_mean_std(data_CD,ax,'C-D interactions')
    plt.xlabel('cluster size, n')
    plt.legend(loc='best')
    plt.savefig('interactions.pdf')

def weighted_interactions_plot():
    data_CC,data_CD = load_data('wints_CC'),load_data('wints_CD')
    fig,ax = plt.subplots()
    plot_mean_std(data_CC,ax,'C-C interactions (degree-weighted)')
    plot_mean_std(data_CD,ax,'C-D interactions (degree-weighted)')
    plt.xlabel('cluster size, n')
    plt.legend(loc='best')
    plt.savefig('interactions_weighted.pdf')
    
def weighted_interactions_compare(show_error=False):
    data_VT,data_hex = load_data('wints_CC'),load_data('./hex/wints_CC')
    fig,ax = plt.subplots()
    plot_mean_std(data_VT,ax,'VT model',show_error=show_error)
    plot_mean_std(data_hex,ax,'EGT',show_error=show_error)
    plt.xlabel('cluster size, n')
    plt.legend(loc='best')
    plt.savefig('interactions_weighted.pdf')
    
def Lambda_CC_plot(show_error=True,fit=False,well_mixed=False,discrete=False):
    sns.set_context('paper')
    data_CC = load_data('wints_CC')
    fig,ax = plt.subplots()
    popt = plot_Lambda_CC(data_CC,ax,show_error=show_error,fit=fit,discrete=discrete)
    if well_mixed:
        n=np.arange(1,100)
        ax.plot(n/N,(n-1)/(N-1))
    formatting(r'Fraction of cooperators, $n/N$',r'$\Lambda^{CC}_n$',large=False)
    # plt.show()
    plt.savefig('Lambda_CC.pdf')
    if fit: return popt
    
def superimposed_weighted_unweighted_interactions():
    data_CC,data_CD = load_data('wints_CC'),load_data('wints_CD')
    fig,ax = plt.subplots()
    plot_mean_std(data_CC,ax,'C-C interactions (degree-weighted)',show_error=False)
    plot_mean_std(data_CD,ax,'C-D interactions (degree-weighted)',show_error=False)
    data_CC,data_CD = load_data('ints_CC'),load_data('ints_CD')  
    plot_mean_std(data_CC,ax,'C-C interactions',mult=1./6,show_error=False)
    plot_mean_std(data_CD,ax,'C-D interactions',mult=1./6,show_error=False)
    plt.xlabel('cluster size, n')
    plt.legend(loc='best')
    plt.savefig('superimposed.pdf')
 

def formatting(xlabel,ylabel,large=False):
    if large: labelsize,ticksize = 26,18
    else: labelsize,ticksize = 16,12   
    plt.xlabel(xlabel,size=labelsize,labelpad=10)
    plt.ylabel(ylabel,size=labelsize,labelpad=10)
    plt.xticks(size=ticksize)
    plt.yticks(size=ticksize)
    plt.tight_layout() 
    
if __name__ == '__main__':
    N=100
    # interactions_plot()
    # weighted_interactions_compare()
    # superimposed_weighted_unweighted_interactions()
    # popt = Lambda_CC_plot(fit=False,well_mixed=False,normalised=True)
    Lambda_CC_plot(show_error=True,discrete=True)
    
