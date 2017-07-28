import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.patches as patches
import seaborn as sns
from shapely.ops import polygonize
from shapely.geometry import LineString, MultiPolygon, MultiPoint, Point

current_palette = sns.color_palette()


def plot_tri(tissue,ax=None,time = None,label=False,palette=current_palette):
    ghosts = tissue.by_mesh('ghost')
    fig = plt.figure()
    plot = []    
    real_centres = tissue.mesh.centres[~ghosts]
    ghost_centres = tissue.mesh.centres[ghosts]        
    plot += plt.triplot(tissue.mesh.centres[:,0], tissue.mesh.centres[:,1], tissue.mesh.tri.copy(),color=palette[3])
    plot += plt.plot(real_centres[:,0], real_centres[:,1], 'o',color = palette[1])
    plot += plt.plot(ghost_centres[:,0],ghost_centres[:,1], 'o')
    if label:
        for i, coords in enumerate(tissue.mesh.centres):
            plot.append(plt.text(coords[0],coords[1],str(i)))
    if time is not None:
        lims = plt.axis()
        plot.append(plt.text(lims[0]+0.1,lims[3]+0.1,'t = %.2f hr'%time))
    plt.show()
    return plot

def plot_cells(tissue,current_palette=current_palette,key=None,ax=None,label=False,time = False,colors=None,centres=True):
    ghosts = tissue.by_mesh('ghost')
    fig = plt.Figure()
    if ax is None:        
        ax = plt.axes()
        plt.axis('scaled')
        xmin,xmax = min(tissue.mesh.centres[:,0]), max(tissue.mesh.centres[:,0])
        ymin,ymax = min(tissue.mesh.centres[:,1]), max(tissue.mesh.centres[:,1])      
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
    ax.cla()
    vor = tissue.mesh.voronoi()
    cells_by_vertex = np.array(vor.regions)[np.array(vor.point_region)]
    verts = [vor.vertices[cv] for cv in cells_by_vertex[~ghosts]]
    if colors is not None: 
        coll = PolyCollection(verts,linewidths=[2.],facecolors=colors)
    elif key is not None:
        colors = np.array(current_palette)[tissue.by_mesh(key)]
        coll = PolyCollection(verts,linewidths=[2.],facecolors=colors)
    else: coll = PolyCollection(verts,linewidths=[2.])
    ax.add_collection(coll)
    if label:
        ids = tissue.by_mesh('id')
        for i, coords in enumerate(tissue.mesh.centres):
            if ~ghosts[i]: plt.text(coords[0],coords[1],str(ids[i]))
    if time:
        lims = plt.axis()
        plt.text(lims[0]+0.1,lims[3]+0.1,'t = %.2f hr'%time)
    if centres: 
        real_centres = tissue.mesh.centres[~ghosts]
        plt.plot(real_centres[:,0], real_centres[:,1], 'o',color='black')        
    plt.show()

# def plot_no_ghost(cells,current_palette=sns.color_palette(),key=None,ax=None,label=False,time = False,centres=True):
#     fig = plt.Figure()
#     if ax is None:
#         ax = plt.axes()
#         plt.axis('scaled')
#         xmin,xmax = min(cells.mesh.centres[:,0]), max(cells.mesh.centres[:,0])
#         ymin,ymax = min(cells.mesh.centres[:,1]), max(cells.mesh.centres[:,1])
#         ax.set_xlim(xmin,xmax)
#         ax.set_ylim(ymin,ymax)
#         # ax.xaxis.set_major_locator(plt.NullLocator())
#         # ax.yaxis.set_major_locator(plt.NullLocator())
#     plot = []
#     vor = cells.mesh.voronoi()
#     cells_by_vertex = np.array(vor.regions)[np.array(vor.point_region)]
#     verts = np.array([vor.vertices[cv] for cv in cells_by_vertex[cells.mesh.ghost_mask]])
#     bf = lambda vs: np.any(np.sqrt(vs[:,0]**2+vs[:,1]**2)>mmax)
#     mmax = cells.mesh.extreme_point() +0.5
#     flag_border = np.array([-1 in region for region in vor.regions])
#     flag_border = flag_border[np.array(vor.point_region)]
#     flag_border[np.where([bf(vs) for vs in verts])] = True
#     if key is None: coll = PolyCollection(verts[~flag_border],linewidths=[2.])
#     else:
#         colors = np.array(current_palette)[cells.by_meshidx(key,False)]
#         coll = PolyCollection(verts[~flag_border],linewidths=[2.],facecolors=colors)
#     ax.add_collection(coll)
#     if label:
#         for i, coords in enumerate(cells.mesh.centres):
#             if cells.mesh.ghost_mask[i]: plot.append(plt.text(coords[0],coords[1],str(cells.mesh.cell_ids[i])))
#     if time:
#         lims = plt.axis()
#         plot.append(plt.text(lims[0]+0.1,lims[3]+0.1,'t = %.2f hr'%time))
#     if centres: plot+=plt.plot(cells.mesh.centres[:,0], cells.mesh.centres[:,1], 'o',color='black')
#     plt.show()
#     return plot


def animate(history, key=None, timestep=None):
    plt.ion()
    v_max = np.max((np.max(history[0].mesh.centres), np.max(history[-1].mesh.centres)))
    if key: key_max = np.max(history[0].properties[key])
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
        for n, tissue in enumerate(history):
            if timestep is not None: plot_cells(tissue,palette,key,ax,time=n*timestep)
            else: plot_cells(tissue,palette,key,ax)
            plt.pause(0.001)
    else:
        for n, tissue in enumerate(history):
            if timestep is not None: plot_cells(tissue,key=None,ax=ax,time=n*timestep)
            else: plot_cells(tissue,ax=None)
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