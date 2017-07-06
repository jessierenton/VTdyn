import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.patches as patches
import seaborn as sns
from shapely.ops import polygonize
from shapely.geometry import LineString, MultiPolygon, MultiPoint, Point

current_palette = sns.color_palette()


def plot_tri(cells,ax=None,time = None,label=False,palette=current_palette):
    fig = plt.figure()
    fig.set_ylim(-1.5,1.5)
    fig.set_xlim(-1.0,2.5)
    plot = []    
    real_centres = cells.mesh.centres[cells.mesh.ghost_mask]
    ghost_centres = cells.mesh.centres[~cells.mesh.ghost_mask]        
    plot += plt.triplot(cells.mesh.centres[:,0], cells.mesh.centres[:,1], cells.mesh.tri.copy(),color=palette[3])
    plot += plt.plot(real_centres[:,0], real_centres[:,1], 'o',color = palette[1])
    plot += plt.plot(ghost_centres[:,0],ghost_centres[:,1], 'o')
    if label:
        for i, coords in enumerate(cells.mesh.centres):
            plot.append(plt.text(coords[0],coords[1],str(i)))
    if time is not None:
        lims = plt.axis()
        plot.append(plt.text(lims[0]+0.1,lims[3]+0.1,'t = %.2f hr'%time))
    plt.show()
    return plot

def plot_cells(cells,current_palette=current_palette,key=None,ax=None,label=False,time = False,colors=None,centres=True):
    fig = plt.Figure()
    if ax is None:        
        ax = plt.axes()
        plt.axis('scaled')
        xmin,xmax = min(cells.mesh.centres[:,0]), max(cells.mesh.centres[:,0])
        ymin,ymax = min(cells.mesh.centres[:,1]), max(cells.mesh.centres[:,1])      
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
    ax.cla()
    vor = cells.mesh.voronoi()
    cells_by_vertex = np.array(vor.regions)[np.array(vor.point_region)]
    verts = [vor.vertices[cv] for cv in cells_by_vertex[cells.mesh.ghost_mask]]
    if colors is not None: 
        coll = PolyCollection(verts,linewidths=[2.],facecolors=colors)
    elif key is not None:
        colors = np.array(current_palette)[cells.by_meshidx(key,False)]
        coll = PolyCollection(verts,linewidths=[2.],facecolors=colors)
    else: coll = PolyCollection(verts,linewidths=[2.])
    ax.add_collection(coll)
    if label:
        for i, coords in enumerate(cells.mesh.centres):
            if cells.mesh.ghost_mask[i]: plt.text(coords[0],coords[1],str(cells.mesh.cell_ids[i]))
    if time:
        lims = plt.axis()
        plt.text(lims[0]+0.1,lims[3]+0.1,'t = %.2f hr'%time)
    if centres: 
        real_centres = cells.mesh.centres[cells.mesh.ghost_mask]
        plt.plot(real_centres[:,0], real_centres[:,1], 'o',color='black')        
    plt.show()

def plot_no_ghost(cells,current_palette=sns.color_palette(),key=None,ax=None,label=False,time = False,centres=True):
    fig = plt.Figure()
    if ax is None:        
        ax = plt.axes()
        plt.axis('scaled')
        xmin,xmax = min(cells.mesh.centres[:,0]), max(cells.mesh.centres[:,0])
        ymin,ymax = min(cells.mesh.centres[:,1]), max(cells.mesh.centres[:,1])      
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        # ax.xaxis.set_major_locator(plt.NullLocator())
        # ax.yaxis.set_major_locator(plt.NullLocator())
    plot = []
    vor = cells.mesh.voronoi()
    cells_by_vertex = np.array(vor.regions)[np.array(vor.point_region)]
    verts = np.array([vor.vertices[cv] for cv in cells_by_vertex[cells.mesh.ghost_mask]])
    bf = lambda vs: np.any(np.sqrt(vs[:,0]**2+vs[:,1]**2)>mmax)
    mmax = cells.mesh.extreme_point() +0.5
    flag_border = np.array([-1 in region for region in vor.regions])
    flag_border = flag_border[np.array(vor.point_region)]
    flag_border[np.where([bf(vs) for vs in verts])] = True
    if key is None: coll = PolyCollection(verts[~flag_border],linewidths=[2.])
    else:
        colors = np.array(current_palette)[cells.by_meshidx(key,False)]
        coll = PolyCollection(verts[~flag_border],linewidths=[2.],facecolors=colors)
    ax.add_collection(coll)
    if label:
        for i, coords in enumerate(cells.mesh.centres):
            if cells.mesh.ghost_mask[i]: plot.append(plt.text(coords[0],coords[1],str(cells.mesh.cell_ids[i])))
    if time:
        lims = plt.axis()
        plot.append(plt.text(lims[0]+0.1,lims[3]+0.1,'t = %.2f hr'%time))
    if centres: plot+=plt.plot(cells.mesh.centres[:,0], cells.mesh.centres[:,1], 'o',color='black')         
    plt.show()    
    return plot


def plot_type(cells,label=False):
    centres = cells.mesh.centres
    A_type = np.where(cells.by_meshidx('type')==0)[0]
    A_type = A_type[cells.mesh.ghost_mask[A_type]]
    B_type = np.where(cells.by_meshidx('type')==1)[0]
    B_type = B_type[cells.mesh.ghost_mask[B_type]]
    ghost_centres = cells.mesh.centres[~cells.mesh.ghost_mask]        
    plt.triplot(cells.mesh.centres[:,0], cells.mesh.centres[:,1], cells.mesh.tri.copy())
    plt.plot(ghost_centres[:,0],ghost_centres[:,1], 'o')
    plt.plot(centres[A_type][:,0], centres[A_type][:,1], 'o')
    plt.plot(centres[B_type][:,0], centres[B_type][:,1], 'o')
    if label:
        for i, coords in enumerate(cells.mesh.centres):
            plt.text(coords[0],coords[1],str(i))
    plt.show()

def animate(cells_array, key = None, timestep=None):
    plt.ion()
    v_max = np.max((np.max(cells_array[0].mesh.centres), np.max(cells_array[-1].mesh.centres)))
    if key: key_max = np.max(cells_array[0].properties[key])
    size = 2.0*v_max
    fig = plt.figure()
    ax = plt.axes()
    plt.axis('scaled')
    lim = [-0.55*size, 0.55*size]    
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    if key is not None:
        palette = sns.color_palette("husl", key_max+1)
        np.random.shuffle(palette)
        for n, cells in enumerate(cells_array):
            if timestep is not None: plot_cells(cells,palette,key,ax,time=n*timestep)
            else: plot_cells(cells,palette,key,ax)
            plt.pause(0.001)
    else:
        for cells in cells_array:
            plot_cells(cells,key,ax)
            plt.pause(0.001)
            
def animate_no_ghost(cells_array, key = None, timestep=None):
    plt.ion()
    v_max = np.max((np.max(cells_array[0].mesh.centres), np.max(cells_array[-1].mesh.centres)))
    if key: key_max = np.max(cells_array[0].properties[key])
    size = 2.0*v_max
    fig = plt.figure()
    ax = plt.axes()
    plt.axis('scaled')
    lim = [-0.55*size, 0.55*size]    
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    fig.set_size_inches(6, 6)
    ax.set_autoscale_on(False)
    plot = []
    if key is not None:
        palette = sns.color_palette("husl", key_max+1)
        np.random.shuffle(palette)
        for n, cells in enumerate(cells_array):
            if len(plot)>0: 
                for p in plot: p.remove()
            for coll in (ax.collections): ax.collections.remove(coll)
            if timestep is not None: plot = plot_no_ghost(cells,palette,key,ax,time=n*timestep)
            else: plot = plot_no_ghost(cells,palette,key,ax)
            plt.pause(0.001)
    else:
        for cells in cells_array:
            plot_no_ghost(cells,key,ax)
            plt.pause(0.001)
 
def animate_mesh(cells_array,timestep):
    plt.ion()
    v_max = np.max((np.max(cells_array[0].mesh.centres), np.max(cells_array[-1].mesh.centres)))
    size = 2.0*v_max
    fig = plt.figure()
    ax = plt.axes()
    plt.axis('scaled')
    lim = [-0.55*size, 0.55*size]    
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    fig.set_size_inches(6, 6)
    ax.set_autoscale_on(False)
    plot = []
    for n,cells in enumerate(cells_array):
        if len(plot)>0: 
            for p in plot: p.remove()
        plot = plot_tri(cells,ax,n*timestep)
        plt.pause(0.001)
    
def animate_video_mpg(cells_array,name,index,facecolours='Default'):
    v_max = np.max((np.max(cells_array[0].mesh.centres), np.max(cells_array[-1].mesh.centres)))
    if key: key_max = np.max(cells_array[0].properties[key])
    size = 2.0*v_max
    outputdir="images"
    if not os.path.exists(outputdir): # if the folder doesn't exist create it
        os.makedirs(outputdir)
    fig = plt.figure()
    ax = plt.axes()
    plt.axis('scaled')
    lim = [-0.55*size, 0.55*size]
    ax.set_xlim(lim)
    ax.set_ylim(lim)    
    frames=[]
    i = 0
    for cells in cells_array:
        plot_cells(cells,key,ax)
        i=i+1
        frame="images/image%04i.png" % i
        fig.savefig(frame,dpi=500)
        frames.append(frame)
    os.system("mencoder 'mf://images/image*.png' -mf type=png:fps=20 -ovc lavc -lavcopts vcodec=wmv2 -oac copy  -o " + "%s%0.3f.mpg" %(name,index))               