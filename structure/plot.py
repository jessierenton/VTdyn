import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, PatchCollection
import matplotlib.patches as patches
import seaborn as sns
from shapely.ops import polygonize
from shapely.geometry import LineString, MultiPolygon, Polygon, MultiPoint, Point
from scipy.spatial import Voronoi
from descartes.patch import PolygonPatch

current_palette = sns.color_palette()


def plot_tri(tissue,ax=None,time = None,label=False,palette=current_palette):
    fig = plt.figure() 
    centres = tissue.mesh.centres     
    plt.triplot(tissue.mesh.centres[:,0], tissue.mesh.centres[:,1], tissue.mesh.tri.copy(),color=palette[3])
    plt.plot(centres[:,0], centres[:,1], 'o',color = palette[1])
    if label:
        for i, coords in enumerate(tissue.mesh.centres):
            plt.text(coords[0],coords[1],str(i))
    if time is not None:
        lims = plt.axis()
        plt.text(lims[0]+0.1,lims[3]+0.1,'t = %.2f hr'%time)
    plt.show()

def get_vertices_for_inf_line(vor,i,pointidx,center,ptp_bound):
    t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
    t /= np.linalg.norm(t)
    n = np.array([-t[1], t[0]])  # normal

    midpoint = vor.points[pointidx].mean(axis=0)
    direction = np.sign(np.dot(midpoint - center, n)) * n
    far_point = vor.vertices[i] + direction * ptp_bound.max()

    return far_point

def get_region_for_infinite(region,vor,center,ptp_bound):    
    infidx= region.index(-1)
    finite_end1 = region[infidx-1]
    finite_end2 = region[(infidx+1)%len(region)]
    
    try: pointidx1 = vor.ridge_points[vor.ridge_vertices.index([-1,finite_end1])]
    except ValueError: pointidx1 = vor.ridge_points[vor.ridge_vertices.index([finite_end1,-1])]
    far_point1 = get_vertices_for_inf_line(vor,finite_end1,pointidx1,center,ptp_bound)
        
    try: pointidx2 = vor.ridge_points[vor.ridge_vertices.index([-1,finite_end2])]
    except ValueError: pointidx2 = vor.ridge_points[vor.ridge_vertices.index([finite_end2,-1])]
    far_point2 = get_vertices_for_inf_line(vor,finite_end2,pointidx2,center,ptp_bound)
    region_vertices = []
    for pt in region:
        if pt != -1: region_vertices.append(vor.vertices[pt])
        else: region_vertices.append(far_point1); region_vertices.append(far_point2)
    return np.array(region_vertices)

def torus_plot(tissue,palette=current_palette,key=None,key_label=None,ax=None,show_centres=False,cell_ids=False,mesh_ids=False):
    width, height = tissue.mesh.width, tissue.mesh.height 
    centres = tissue.mesh.centres 
    centres_3x3 = np.vstack([centres+[dx, dy] for dx in [-width, 0, width] for dy in [-height, 0, height]])
    N = tissue.mesh.N_mesh
    mask = np.full(9*N,False,dtype=bool)
    mask[4*N:5*N]=True
    vor = Voronoi(centres_3x3)
    
    mp = MultiPolygon([Polygon(vor.vertices[region])
                for region in (np.array(vor.regions)[np.array(vor.point_region)])[mask]])
    
    if ax is None: 
        fig = plt.figure()
        ax = fig.add_subplot(111)
        minx, miny, maxx, maxy = 0,0,width,height
        w, h = maxx - minx, maxy - miny
        ax.set_xlim(minx - 0.5 * w, maxx + 0.5 * w)
        ax.set_ylim(miny - 0.5 * h, maxy + 0.5 * h)
        ax.set_aspect(1)
    
    if key is None: ax.add_collection(PatchCollection([PolygonPatch(p) for p in mp]))
    else:
        colours = palette[tissue.by_mesh(key)]
        coll = PatchCollection([PolygonPatch(p,facecolor = c) for p,c in zip(mp,colours)],match_original=True)
        ax.add_collection(coll)
    
    if show_centres: 
        plt.plot(centres[:,0], centres[:,1], 'o',color='black')
    if cell_ids:
        ids = tissue.by_mesh('id')
        for i, coords in enumerate(tissue.mesh.centres):
            plt.text(coords[0],coords[1],str(ids[i]))
    if mesh_ids:
        for i, coords in enumerate(tissue.mesh.centres):
            plt.text(coords[0],coords[1],str(i))
    if key_label is not None:
        ids = tissue.by_mesh(key_label)
        for i, coords in enumerate(tissue.mesh.centres):
            plt.text(coords[0],coords[1],str(ids[i]))
    
    plt.show()
    
    
def finite_plot(tissue,palette=current_palette,key=None,key_label=None,ax=None,show_centres=False,cell_ids=False,mesh_ids=False):
    centres = tissue.mesh.centres 
    vor = tissue.mesh.voronoi()
    center = vor.points.mean(axis=0)
    ptp_bound = vor.points.ptp(axis=0)
    
    regions = [Polygon(vor.vertices[region]) if -1 not in region else
                Polygon(get_region_for_infinite(region,vor,center,ptp_bound))
                for region in np.array(vor.regions)[np.array(vor.point_region)] if len(region)>=2
                ]
    convex_hull = MultiPoint([Point(i) for i in centres]).convex_hull
    mp = MultiPolygon(
        [poly.intersection(convex_hull) for poly in regions])
   
    if ax is None: 
        fig = plt.figure()
        ax = fig.add_subplot(111)
        minx, miny, maxx, maxy = mp.bounds
        w, h = maxx - minx, maxy - miny
        ax.set_xlim(minx - 0.2 * w, maxx + 0.2 * w)
        ax.set_ylim(miny - 0.2 * h, maxy + 0.2 * h)
        ax.set_aspect(1)
    
    if key is None: ax.add_collection(PatchCollection([PolygonPatch(p) for p in mp]))
    else:
        colours = palette[tissue.by_mesh(key)]
        coll = PatchCollection([PolygonPatch(p,facecolor = c) for p,c in zip(mp,colours)],match_original=True)
        # coll.set_facecolors(palette[tissue.by_mesh(key)])
        ax.add_collection(coll)
    
    if show_centres: 
        plt.plot(centres[:,0], centres[:,1], 'o',color='black')
    if cell_ids:
        ids = tissue.by_mesh('id')
        for i, coords in enumerate(tissue.mesh.centres):
            plt.text(coords[0],coords[1],str(ids[i]))
    if mesh_ids:
        for i, coords in enumerate(tissue.mesh.centres):
            plt.text(coords[0],coords[1],str(i))
    if key_label is not None:
        ids = tissue.by_mesh(key_label)
        for i, coords in enumerate(tissue.mesh.centres):
            plt.text(coords[0],coords[1],str(ids[i]))
    
    plt.show()


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


def animate_finite(history, key = None, timestep=None):
    xmin,ymin = np.amin([np.amin(tissue.mesh.centres,axis=0) for tissue in history],axis=0)*1.5
    xmax,ymax = np.amax([np.amax(tissue.mesh.centres,axis=0) for tissue in history],axis=0)*1.5
      
    plt.ion()
    fig = plt.figure()
    ax = plt.axes()
    plt.axis('scaled')  
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    fig.set_size_inches(6, 6)
    ax.set_autoscale_on(False)
    plot = []
    if key is not None:
        key_max = max((max(tissue.by_mesh(key)) for tissue in history))
        palette = np.array(sns.color_palette("husl", key_max+1))
        np.random.shuffle(palette)
        for tissue in history:
            ax.cla()
            finite_plot(tissue,palette,key,ax=ax)
            plt.pause(0.001)
    else:
        for tissue in history:
            ax.cla()
            finite_plot(tissue,ax=ax)
            plt.pause(0.001)

def animate_torus(history, key = None, timestep=None):
    width,height = history[0].mesh.width,history[0].mesh.height
    xmin, ymin, xmax, ymax = -0.5*width,-0.5*width ,width*1.5,height*1.5
    plt.ion()
    fig = plt.figure()
    ax = plt.axes()
    plt.axis('scaled')  
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    fig.set_size_inches(6, 6)
    ax.set_autoscale_on(False)
    plot = []
    if key is not None:
        key_max = max((max(tissue.by_mesh(key)) for tissue in history))
        palette = np.array(sns.color_palette("husl", key_max+1))
        np.random.shuffle(palette)
        for tissue in history:
            ax.cla()
            torus_plot(tissue,palette,key,ax=ax)
            plt.pause(0.001)
    else:
        for tissue in history:
            ax.cla()
            torus_plot(tissue,ax=ax)
            plt.pause(0.01)
            
            
def animate_video_mpg(history,name,index,facecolours='Default'):
    v_max = np.max((np.max(history[0].mesh.centres), np.max(history[-1].mesh.centres)))
    if key: key_max = np.max(history[0].properties[key])
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
    for cells in history:
        plot_cells(cells,key,ax)
        i=i+1
        frame="images/image%04i.png" % i
        fig.savefig(frame,dpi=500)
        frames.append(frame)
    os.system("mencoder 'mf://images/image*.png' -mf type=png:fps=20 -ovc lavc -lavcopts vcodec=wmv2 -oac copy  -o " + "%s%0.3f.mpg" %(name,index))               