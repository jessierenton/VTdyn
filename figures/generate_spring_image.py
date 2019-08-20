import numpy as np
import libs.pd_lib as lib #library for simulation routines
import libs.data as data
import libs.plot as vplt #plotting library
from structure.global_constants import *
import structure.initialisation as init
from structure.cell import Tissue, BasicSpringForceNoGrowth
import matplotlib.pyplot as plt

pink="#EE5474"
light_pink="#FBDCE3" #FDEDF1
dark_purple ="#4C384B"
purple = "#6F6879"
blue = "#8AA1A0"
light_blue = "#E7ECEC"
light_purple ="#F5F5F8"
beige = "#F0E7C7"
green = "#DDE5B0"
light_green = "#F8F9EF"

BACKGROUND_COLOUR=light_green
FIGSIZE = (8,8)
FONT ="Trebuchet MS"

def plot_springs_for_cell(ax,mesh,cell,do_not_plot=[],colour=None):
    r1 = mesh.centres[cell]
    for n,r2 in zip(mesh.neighbours[cell],mesh.centres[mesh.neighbours[cell]]):
        if n not in do_not_plot:
            vplt.plot_spring(r1,r2,ax,colour,lw=1.2)


def plot_springs_recursively(ax,mesh,focal_cell,level,do_not_plot,colour):
    plot_springs_for_cell(ax,mesh,focal_cell,do_not_plot,colour)
    if level !=0:
        do_not_plot = [focal_cell]
        for cell in mesh.neighbours[focal_cell]:
            plot_springs_recursively(ax,mesh,cell,level-1,do_not_plot,colour)
            do_not_plot.append(cell)

def plot_springs(tissue,focal_cell,levels_to_plot,spring_colour,cell_colour,edge_colour,figsize=FIGSIZE):
    mesh = tissue.mesh
    if cell_colour is not None: cell_colour = (cell_colour,)
    ax = vplt.torus_plot(tissue,figsize=figsize,lw=5.,palette=cell_colour,edgecolor=edge_colour)
    ax.set_xlim((-2.6,1.6))
    ax.set_ylim(-1.4,1.8)
    ax.set_facecolor(BACKGROUND_COLOUR)
    plot_springs_recursively(ax,mesh,focal_cell,levels_to_plot,[],spring_colour)
    ax.scatter(mesh.centres[:,0],mesh.centres[:,1],c=edge_colour,s=200,zorder=100)

def plot_death_birth(tissue,stage,dead,mother,line_colour,mother_colour,type_colour,type_colour_light,figsize=FIGSIZE,nodesize=3000,lw=5):
    mesh=tissue.mesh
    types = np.zeros(100,dtype=int)
    type_1 = np.array((12,13,16,42,68,3,67))
    types[type_1]=1
    plt.ion()
    fig,ax = vplt.create_axes(tissue,figsize)
    ax = vplt.plot_tri_torus(tissue,fig,ax,line_colour=line_colour,node_colour=line_colour,lw=3)
    ax.set_xlim((-2.2,1.2))
    ax.set_ylim(-1.5,1.5)
    ax.set_facecolor(BACKGROUND_COLOUR)
    if stage == 0: 
        ax.scatter(mesh.centres[:,0],mesh.centres[:,1],c=type_colour[types],s=nodesize,zorder=100,edgecolor=line_colour,lw=lw)
    elif stage==1:
        ax.scatter(mesh.centres[:,0],mesh.centres[:,1],c=type_colour_light[types],s=nodesize,zorder=100,edgecolor=line_colour,lw=lw)
        ax.scatter(mesh.centres[mesh.neighbours[dead]][:,0],mesh.centres[mesh.neighbours[dead]][:,1],c=type_colour[types][mesh.neighbours[dead]],edgecolor=line_colour,s=nodesize,zorder=105,lw=lw)
        ax.scatter(mesh.centres[dead][0],mesh.centres[dead][1],c=line_colour,s=nodesize,zorder=107,lw=lw)
        ax.scatter(mesh.centres[mother][0],mesh.centres[mother][1],c=type_colour[types[mother]],edgecolor=mother_colour,s=nodesize+40,zorder=110,lw=lw)
        ax.text(mesh.centres[mother][0]-.085,mesh.centres[mother][1]-0.075,'R',color=mother_colour,zorder=120,fontsize=32,fontname=FONT)
        ax.text(mesh.centres[dead][0]-.065,mesh.centres[dead][1]-0.09,'D',color=type_colour_light[types[dead]],zorder=120,fontsize=32,fontname=FONT)
    elif stage==2:
        types[dead]=types[mother]
        ax.scatter(mesh.centres[:,0],mesh.centres[:,1],c=type_colour[types],s=nodesize,zorder=100,edgecolor=line_colour,lw=lw) 

def plot_birth_death(tissue,stage,mother,dead,line_colour,mother_colour,type_colour,type_colour_light,figsize=FIGSIZE,nodesize=3000,lw=5):
    mesh=tissue.mesh
    types = np.zeros(100,dtype=int)
    type_1 = np.array((12,13,16,42,68,3,67))
    types[type_1]=1
    plt.ion()
    fig,ax = vplt.create_axes(tissue,figsize)
    ax = vplt.plot_tri_torus(tissue,fig,ax,line_colour=line_colour,node_colour=line_colour,lw=3)
    ax.set_xlim((-2.2,1.2))
    ax.set_ylim(-1.5,1.5)
    ax.set_facecolor(BACKGROUND_COLOUR)
    if stage == 0: ax.scatter(mesh.centres[:,0],mesh.centres[:,1],c=type_colour[types],s=nodesize,zorder=100,edgecolor=line_colour,lw=lw)
    elif stage==1:
        ax.scatter(mesh.centres[:,0],mesh.centres[:,1],c=type_colour_light[types],s=nodesize,zorder=100,edgecolor=line_colour,lw=lw)
        ax.scatter(mesh.centres[mesh.neighbours[mother]][:,0],mesh.centres[mesh.neighbours[mother]][:,1],color=type_colour[types][mesh.neighbours[mother]],edgecolor=line_colour,s=nodesize,zorder=105,lw=lw)
        ax.scatter(mesh.centres[mother][0],mesh.centres[mother][1],c=type_colour[types[mother]],edgecolor=mother_colour,s=nodesize+40,zorder=110,lw=lw)
        ax.scatter(mesh.centres[dead][0],mesh.centres[dead][1],c=line_colour,s=nodesize,zorder=107,lw=lw)
        ax.text(mesh.centres[mother][0]-.085,mesh.centres[mother][1]-0.075,'R',color=mother_colour,zorder=120,fontsize=32,fontname="Trebuchet MS")
        ax.text(mesh.centres[dead][0]-.065,mesh.centres[dead][1]-0.09,'D',color=type_colour_light[types[dead]],zorder=120,fontsize=32,fontname="Trebuchet MS")
    elif stage==2:
        types[focal_cell]=types[mother]
        ax.scatter(mesh.centres[:,0],mesh.centres[:,1],c=type_colour[types],s=nodesize,zorder=100,edgecolor=line_colour,lw=lw) 

def plot_types(tissue,ancestor_index,line_colour,type_colour,figsize=FIGSIZE,nodesize=700,lw=3):
    mesh=tissue.mesh
    types = np.zeros(100,dtype=int)
    for a in ancestor_index:
        types[tissue.properties['ancestors']==a]=1
    fig,ax = vplt.create_axes(tissue,figsize)
    ax.set_xlim((-4.7,4.4))
    ax.set_ylim(-3.9,3.6)
    ax.set_facecolor(BACKGROUND_COLOUR)
    ax = vplt.plot_tri_torus(tissue,fig,ax,line_colour=line_colour,node_colour=line_colour,lw=2)    
    ax.scatter(mesh.centres[:,0],mesh.centres[:,1],c=type_colour[types],s=nodesize,zorder=100,edgecolor=line_colour,lw=lw)
    return fig,ax

def plot_cells(tissue,cell_colour,edge_colour,figsize=FIGSIZE):
    fig,ax = vplt.create_axes(tissue,figsize)
    ax.set_xlim((-3.7,3.))
    ax.set_ylim(-2.9,0.7)
    vplt.torus_plot(tissue,fig=fig,ax=ax,palette=(cell_colour,),figsize=figsize,edgecolor=edge_colour)
        
if __name__ == "__main__":
    l = 10 # population size N=l*l
    timend = 20. # simulation time (hours)
    timestep = 1. # time intervals to save simulation history

    rand = np.random.RandomState(101)
    b,c,DELTA = 3.,1.0,0.025 #prisoner's dilemma game parameters


    simulation = lib.simulation_decoupled_update  #simulation routine imported from lib
    game = lib.prisoners_dilemma_averaged #game imported from lib
    game_parameters = (b,c)
    
    spring_colour,cell_colour,edge_colour = pink,BACKGROUND_COLOUR,dark_purple
    line_colour=dark_purple
    mother_colour=green
    type_colour=np.array((blue,pink))
    type_colour_light=np.array((light_blue,light_pink))
    
    history1 = lib.run_simulation(simulation,l,timestep,timend,rand,DELTA,game,game_parameters,til_fix=False,init_time=None,save_areas=True)
    tissue = history1[-2]
    focal_cell = 78
    plot_springs(tissue,focal_cell,3,spring_colour=spring_colour,cell_colour=cell_colour,edge_colour=edge_colour,figsize=(10,10))
    plt.savefig('voronoi_springs3.pdf',dpi=300,pad_inches=0.,bbox_inches='tight')
    # for s in (0,1,2):
#         plot_death_birth(tissue,s,78,13,line_colour,mother_colour,type_colour,type_colour_light)
#         plt.savefig('db_%d.pdf'%s,dpi=300,pad_inches=0.,bbox_inches='tight',facecolor=BACKGROUND_COLOUR)
#     plot_birth_death(tissue,1,13,78,line_colour,mother_colour,type_colour,type_colour_light)
#     plt.savefig('bd_1.pdf',dpi=300,pad_inches=0.,bbox_inches='tight',facecolor=BACKGROUND_COLOUR)
#
#     timend = 60
#     history2 = lib.run_simulation(simulation,l,timestep,timend,rand,DELTA,game,game_parameters,til_fix=False,init_time=None,save_areas=True)
#     tissue = history2[-1]
#     plot_types(tissue,(55,46,4863,73,63,48),line_colour,type_colour)
#     plt.savefig('types_graph.pdf',dpi=300,pad_inches=0.,bbox_inches='tight')
#
#     plot_cells(history2[-10],BACKGROUND_COLOUR,green)
#     plt.savefig('cells.pdf',dpi=300,pad_inches=0.,bbox_inches='tight')