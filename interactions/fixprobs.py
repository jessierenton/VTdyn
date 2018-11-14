import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns
import numpy as np
import os

sns.set_style('white')

def save_averaged_coop_interaction(data_CC):
    coop_int = np.array([np.mean(dat_n)/(i+1.) for i,dat_n in enumerate(data_CC)])
    np.savetxt('Lambda_cc',coop_int)
    
def fix_prob_1(b,c,Lambda_cc,DELTA):
    f = lambda j: -c/N +b/N*(Lambda_cc[j-1]*N-j)/(N-j)
    return 1./N +DELTA/N*sum(sum(f(j) for j in range(1,k+1)) for k in range(1,N))
    
def fix_prob(i,b,c,Lambda_cc,DELTA):
    if i==0: return 0
    elif i==N: return 1
    g = lambda j: -c/N +b/N*(Lambda_cc[j-1]*N-j)/(N-j)
    return float(i)/N*(1.+DELTA*(sum(sum(g(j) for j in range(1,k+1)) for k in range(1,N))
            -float(N)/i*sum(sum(g(j) for j in range(1,k+1)) for k in range(1,i))))

def critical_ratio(Lambda_cc,DELTA):
    return N/2*(N-1)/sum(sum((Lambda_cc[j-1]*N-j)/(N-j) for j in range(1,k+1)) for k in range(1,N))

def read_data(filename):
    dat = np.loadtxt(filename,dtype=float).T
    fix = dat[0].sum()
    lost = dat[1].sum()
    return fix,lost    
    
def beneficial_ST(Lambda_cc,DELTA):
    cr = critical_ratio(Lambda_cc,DELTA)
    sigma = (cr+1)/(cr-1)
    print sigma
    
    fig,main_ax = plt.subplots()
    main_ax.set_frame_on(False)
    cp = sns.color_palette('deep6')
    formatting('T','S')
    plt.xticks((0,1,2))
    plt.yticks((-1,0,1))
    plt.plot(np.ones(2),np.linspace(-1,1,2),ls='--',color='black',linewidth=1)
    plt.plot(np.linspace(0,2,2),np.zeros(2),ls='--',color='black',linewidth=1)
    # plt.plot(np.linspace(0,2,2),1-(np.linspace(0,2,2)),ls=':',color=cp[2],lw=1)
    pd = plt.text(1.9, -0.9, 'Prisoner\'s dilemma',size=14,ha='right',va='bottom')
    sh = plt.text(0.1, -0.9, 'Stag-hunt',size=14,ha='left',va='bottom')
    hg = plt.text(0.1, 0.9, 'Harmony',size=14,ha='left',va='top')
    sd = plt.text(1.9, 0.9, 'Snowdrift',size=14,ha='right',va='top')
    
    T = np.linspace(0,2,100)
    plt.plot(T,T-1,label='WM')
    
    sigma_hl = (6.67+1)/(6.67-1)
    T = np.linspace(sigma_hl-1,2,100)
    plt.plot(T,T-sigma_hl,label='HL')
    
    T = np.linspace(sigma-1,2,100)
    plt.plot(T,T-sigma,label='VT')
    
    plt.legend(bbox_to_anchor=(0.9,0.9),loc='upper right',frameon=True,bbox_transform=main_ax.transData)
    rect = patches.Rectangle((0,-1),2,2,linewidth=1,edgecolor='black',facecolor='none')
    main_ax.add_patch(rect)
    plt.show()

    
def plot_vt_type(folder,ax,color,marker='o',label=None,error=False,bestfit=False):
    filenames = [filename for filename in os.listdir(folder)if filename[:3]=='fix']
    results = np.array([read_data('%s/%s'%(folder,filename)) for filename in filenames])
    b_vals = np.array([filename[3:] for filename in filenames],dtype=float)
    fixprobs = results[:,0]/results[:,1]
    ax.plot(b_vals,fixprobs*100,marker=marker,ls='',linewidth=1.5,label=label,color=color)

def formatting(xlabel,ylabel,large=False):
    if large: labelsize,ticksize = 26,18
    else: labelsize,ticksize = 16,12   
    plt.xlabel(xlabel,size=labelsize,labelpad=10)
    plt.ylabel(ylabel,size=labelsize,labelpad=10)
    plt.xticks(size=ticksize)
    plt.yticks(size=ticksize)
    plt.tight_layout()

def plot_fixprob_vs_b(DELTA_vals,b_vals,fname):
    sns.set_context('paper')
    sns.set_style('white')
    Lambda_cc = np.loadtxt('Lambda_cc')
    x = np.arange(0.01,1.0,0.01)
    fig,ax=plt.subplots()
    for col,DELTA in zip(sns.color_palette(),DELTA_vals):
        fix_probs = [fix_prob_1(b,c,Lambda_cc,DELTA)*100 for b in b_vals]
        ax.plot(b_vals,fix_probs,color=col)
        plot_vt_type('sim_data/delta%.3f'%DELTA,ax,col)
    ax.plot(b_vals,[1.0]*len(b_vals),linestyle='--',color='grey')
    
    formatting(r'Benefit-to-cost ratio, $b/c$',r'Fixation probability, $\rho_C$ (%)',large=False)
    # plt.legend()
    plt.savefig(fname+'.pdf')
    plt.show()
    
def plot_fixprob_vs_clustersize(b_vals,DELTA,fname=None):
    if len(b_vals) > 6: cp = sns.color_palette('hls',len(b_vals))
    else: cp = sns.color_palette()
    sns.set_context('paper')
    sns.set_style('white')
    Lambda_cc = np.loadtxt('Lambda_cc')
    cluster_prob = [[fix_prob(i,b,c,Lambda_cc,DELTA)*100 for i in range(0,101)] for b in b_vals]
    fig,ax=plt.subplots()
    for b,fix_probs,color in zip(b_vals,cluster_prob,cp): ax.plot(np.arange(0,101),fix_probs,label=r'$b=%.1f$'%b,color = color)
    ax.plot(range(0,101),[100.0]*101,linestyle='--',color='grey')
    # ax.plot(np.arange(0,101),np.arange(0,101),linestyle='--',color='grey')
    formatting(r'Initial cluster size, $i$',r'Fixation probability, $\rho_{i}$ (%)',large=False)
    plt.legend()
    if fname is not None: plt.savefig(fname)
    plt.show()

def plot_cluster_data():
    b=3. 
    sns.set_context('paper')
    sns.set_style('white')
    Lambda_cc = np.loadtxt('Lambda_cc')
    fix_probs = [fix_prob(i,b,c,Lambda_cc,DELTA)*100 for i in range(0,51)]
    fig,ax=plt.subplots()
    data = np.loadtxt('./sim_data/clusters/fix3',dtype=float)
    data_single = np.append(1,np.sum(np.loadtxt('./sim_data/clusters/fix3_single',dtype=float),axis=0))
    data = np.vstack((data_single,data)).T
    ax.plot(range(0,51),fix_probs)
    ax.plot(data[0],data[1]*100/(data[1]+data[2]),marker='o',ls='')
    ax.plot()
    formatting(r'Initial cluster size, $i$',r'Fixation probability, $\rho_{i}$ (%)',large=False)
    plt.savefig('cluster_data.pdf')
    plt.show()

def save_fix_probs(DELTA_vals,b_vals,fname):
    Lambda_cc = np.loadtxt('Lambda_cc')
    with open(fname,'w') as f:
        for b in b_vals:
            f.write('%.1f     %.3f\n'%(b,fix_prob_1(b,c,Lambda_cc,DELTA)*100))
    

N = 100
DELTA = 0.025
c = 1.
if __name__ == '__main__':    
    b_vals = np.arange(2.,8.5,0.5)
    # plot_fixprob_vs_clustersize(b_vals,0.025,'fixprobs_vs_cluster.pdf')
    # b_vals = (4.,5.,6.,8.,10.)
#     plot_fixprob_vs_clustersize(b_vals,0.025,'fixprobs_vs_cluster_invalid_range.pdf')
    # b_vals = np.arange(1.5,4.1,0.1)
    # plot_fixprob_vs_b((0.025,),b_vals,'fixprobs_vt_theory')
    # plot_cluster_data()
    # Lambda_cc = np.loadtxt('Lambda_cc')
    # print critical_ratio(Lambda_cc,DELTA)
    # beneficial_ST(Lambda_cc,DELTA)
    save_fix_probs(DELTA,b_vals,'fix_probs.txt')
