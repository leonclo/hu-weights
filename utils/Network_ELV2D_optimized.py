import sys
import os
import numpy as np
import scipy as sp
from numba import njit
import matplotlib.pyplot as plt
import networkx as nx
import time
import psutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add parent directory to path for database imports
from db_utils import NetworkResultsDB
@njit(fastmath=True)
def URL(a, Z2): #a is the deformation parameter; Z2 is a flattened Z2 lattice
    for i in range(Z2.shape[0]):
        shift_x = (np.random.random() - 0.5) * a
        shift_y = (np.random.random() - 0.5) * a
        Z2[i, 0] += shift_x
        Z2[i, 1] += shift_y
# Apply Uniformly Random Local (URL) disorder to lattice points by adding random shifts
def tile(flatlat,N):
    tiled = np.empty([(flatlat.shape[0])*9,2])
    im = 0
    for i in [0,-1,1]:
        for j in [0,-1,1]:
            tiled[im * flatlat.shape[0]:(im * flatlat.shape[0])+flatlat.shape[0]] = flatlat + np.array([i*N, j*N])
            im += 1
    return(tiled)
# Create a 3x3 tiled array of the lattice with periodic boundary conditions
@njit(fastmath=True)
def wrap(N,flatlat):
    for i in range(flatlat.shape[0]):
        if flatlat[i,0] < 0: flatlat[i,0] += N
        elif flatlat[i,0] > N: flatlat[i,0] -= N
        if flatlat[i,1] < 0: flatlat[i,1] += N
        elif flatlat[i,1] > N: flatlat[i,1] -= N
# Wrap points back into the simulation box using periodic boundary conditions
@njit(fastmath=True)
def mindist(ep1, ep2, wind):
    diff = ep2 - ep1
    normsq = np.sum(diff**2)
    if normsq == 0:
        return np.sqrt(np.sum((wind - ep1)**2))
    t = max(0.0, min(1.0, np.dot(wind - ep1, diff)/normsq))
    proj = ep1 + t * diff
    return np.sqrt(np.sum((wind-proj)**2))
# Calculate minimum distance from a point to a line segment
def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2, full_line=False, tangent_tol=1e-9):
    """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

    :param circle_center: The (x, y) location of the circle center
    :param circle_radius: The radius of the circle
    :param pt1: The (x, y) location of the first point of the segment
    :param pt2: The (x, y) location of the second point of the segment
    :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
    :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.

    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
    """
    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center 
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy) 
    dx, dy = (x2 - x1), (y2 - y1) 
    dr = (dx ** 2 + dy ** 2)**.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections
# Find intersection points between a circle and line segment
@njit(fastmath=True)
def eclen(cent, rad, v1, v1len, v2, v2len, edgelen):
    if v1len < rad and v2len < rad:
        return edgelen
    
    # Simplified geometric approach for circle-line intersection length
    diff = v2 - v1
    to_cent = cent - v1
    
    proj_length = np.dot(to_cent, diff) / np.dot(diff, diff)
    proj_length = max(0.0, min(1.0, proj_length))
    
    closest_point = v1 + proj_length * diff
    dist_to_center = np.sqrt(np.sum((closest_point - cent)**2))
    
    if dist_to_center >= rad:
        return 0.0
    
    # Calculate chord length using geometry
    chord_half = np.sqrt(rad**2 - dist_to_center**2)
    line_dir = diff / np.sqrt(np.sum(diff**2))
    
    intersect1 = closest_point - chord_half * line_dir
    intersect2 = closest_point + chord_half * line_dir
    
    # Project intersections back onto line segment
    t1 = np.dot(intersect1 - v1, diff) / np.dot(diff, diff)
    t2 = np.dot(intersect2 - v1, diff) / np.dot(diff, diff)
    
    t1 = max(0.0, min(1.0, t1))
    t2 = max(0.0, min(1.0, t2))
    
    p1 = v1 + t1 * diff
    p2 = v1 + t2 * diff
    
    return np.sqrt(np.sum((p2 - p1)**2))
# Calculate weighted edge length inside a circular window as percentage of total edge 
def gabriel_graph_2d(points):
    """
    Compute the Gabriel graph of a set of points in 2D or 3D space.

    Parameters:
    - points: (N, D) numpy array where N is the number of points and D is the dimension (2 or 3).

    Returns:
    - edge_array: (M, 3) numpy array where each row represents an edge in the format [node1, node2, weight].
    """

    # Build a KD-tree for efficient neighbor searches
    tree = sp.spatial.cKDTree(points)

    # Compute the Delaunay triangulation
    tri = sp.spatial.Delaunay(points)
    simplices = tri.simplices  # Indices of points forming the simplices


    # Generate all possible edges from the simplices
    if simplices.shape[1] == 3:  # 2D case
        edges = np.vstack([simplices[:, [0, 1]],
                           simplices[:, [1, 2]],
                           simplices[:, [2, 0]]])

    elif simplices.shape[1] == 4:  # 3D case
        edges = np.vstack([simplices[:, [0, 1]],
                           simplices[:, [0, 2]],
                           simplices[:, [0, 3]],
                           simplices[:, [1, 2]],
                           simplices[:, [1, 3]],
                           simplices[:, [2, 3]]])
    else:
        raise ValueError('Input points must be 2D or 3D.')


    # Sort and remove duplicate edges
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)


    # Compute midpoints and radii for the Gabriel condition
    i = edges[:, 0]
    j = edges[:, 1]
    points_i = points[i]
    points_j = points[j]
    midpoints = (points_i + points_j) / 2
    radii = 0.5 * np.linalg.norm(points_i - points_j, axis=1)
    weights = 2 * radii  # Edge weights are the Euclidean distances

    # Query KD-tree to find neighboring points within the radius
    idx_list = tree.query_ball_point(midpoints, radii)

    # Build the edge list for the Gabriel graph
    edge_list = []
    for k in range(len(edges)):
        idx = set(idx_list[k]) - {i[k], j[k]}
        if not idx:
            edge_list.append([i[k], j[k], weights[k]])

    edge_array = np.array(edge_list)
    return edge_array
# Generate Gabriel graph edges from 2D point set
def remove_duplicate_rows(arr):
    dtype = [('min', arr.dtype), ('max', arr.dtype)]
    structured = np.empty(arr.shape[0], dtype=dtype)
    structured['min'] = np.minimum(arr[:, 0], arr[:, 1])
    structured['max'] = np.maximum(arr[:, 0], arr[:, 1])
    _, unique_indices = np.unique(structured, return_index=True)
    unique_indices.sort()
    return arr[unique_indices]
# Remove duplicate edges from edge array (handles both [i,j] and [j,i])
def restructure_array_numpy(triangles):
    reshaped = triangles
    output = np.empty((len(reshaped) * 3, 2), dtype=triangles.dtype)
    output[::3, 0] = reshaped[:, 0]
    output[::3, 1] = reshaped[:, 1]
    output[1::3, 0] = reshaped[:, 1]
    output[1::3, 1] = reshaped[:, 2]
    output[2::3, 0] = reshaped[:, 2]
    output[2::3, 1] = reshaped[:, 0]
    return output
# Convert triangle array to edge array (3 edges per triangle)
def delaunay_graph_2d(points):
    # Perform Delaunay triangulation
    delaunay = sp.spatial.Delaunay(points)
    triangles = np.array(delaunay.simplices)

    # Convert triangles to edges
    edges = restructure_array_numpy(triangles)

    # Remove duplicate edges
    unique_edges = remove_duplicate_rows(edges)

    return unique_edges
# Generate Delaunay triangulation edges from 2D point set
@njit(fastmath=True)
def Z2_gen(N): #N is the number of sites along one lattice vector
    hold = np.zeros((N,N,2))
    for i in range(N):
        for j in range(N):
            hold[i,j,0] = i
            hold[i,j,1] = j
    return hold
# Generate square lattice (Z2) with N sites along each direction
def calculate_edge_weights(tURL, edges, weight_type='uniform', **kwargs):
    n_of_edges = len(edges)
    weights = np.ones((n_of_edges,3))

    if weight_type == 'uniform':
        return weights
    
    # Vectorized edge length calculation
    edge_vectors = tURL[edges[:,1]] - tURL[edges[:,0]]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    
    if weight_type == 'length_power_law':
        alpha = kwargs.get('alpha',2.0)
        weights[:,0] = edge_lengths ** alpha
        weights[:,1] = edge_lengths
    elif weight_type == 'length_binned':
        # 4-bin categorization based on edge length percentiles - vectorized
        percentiles = np.percentile(edge_lengths, [25, 50, 75])
        bin_weights = kwargs.get('bin_weights', [0.2, 0.7, 1.3, 2.5])
        
        # Vectorized binning
        bin_indices = np.digitize(edge_lengths, percentiles)
        weight_multipliers = np.array(bin_weights)[bin_indices]
        weights[:,0] = edge_lengths * weight_multipliers
        weights[:,1] = edge_lengths
    elif weight_type == 'length_threshold':
        # Binary short/long classification - vectorized
        threshold = kwargs.get('threshold', np.median(edge_lengths))
        short_weight = kwargs.get('short_weight', 0.3)
        long_weight = kwargs.get('long_weight', 2.5)
        
        weight_multipliers = np.where(edge_lengths <= threshold, short_weight, long_weight)
        weights[:,0] = edge_lengths * weight_multipliers
        weights[:,1] = edge_lengths
    elif weight_type == 'length_inverse':
        # 1/L^beta weighting (emphasizes short edges) - vectorized
        beta = kwargs.get('beta', 1.0)
        weights[:,0] = edge_lengths ** (-beta)
        weights[:,1] = edge_lengths
    elif weight_type == 'betweenness_centrality':
        # Use NetworkX to compute edge betweenness centrality
        G = nx.Graph()
        G.add_edges_from(edges)
        edge_centrality = nx.edge_betweenness_centrality(G)
        for i in range(n_of_edges):
            edge_length = np.linalg.norm(tURL[edges[i,0]] - tURL[edges[i,1]])
            edge_tuple = tuple(sorted([edges[i,0], edges[i,1]]))
            centrality = edge_centrality.get(edge_tuple, 0.5)
            weights[i,0] = edge_length * (1 + centrality)
            weights[i,1] = edge_length
    elif weight_type == 'closeness_centrality':
        # Weight edges based on node closeness centrality
        G = nx.Graph()
        G.add_edges_from(edges)
        node_centrality = nx.closeness_centrality(G)
        for i in range(n_of_edges):
            edge_length = np.linalg.norm(tURL[edges[i,0]] - tURL[edges[i,1]])
            avg_centrality = (node_centrality[edges[i,0]] + node_centrality[edges[i,1]]) / 2
            weights[i,0] = edge_length * (1 + avg_centrality)
            weights[i,1] = edge_length
    elif weight_type == 'degree_centrality':
        # Weight edges based on node degree
        G = nx.Graph()
        G.add_edges_from(edges)
        degrees = dict(G.degree())
        max_degree = max(degrees.values())
        for i in range(n_of_edges):
            edge_length = np.linalg.norm(tURL[edges[i,0]] - tURL[edges[i,1]])
            avg_degree = (degrees[edges[i,0]] + degrees[edges[i,1]]) / 2
            normalized_degree = avg_degree / max_degree
            weights[i,0] = edge_length * (1 + normalized_degree)
            weights[i,1] = edge_length
    elif weight_type == 'sinusoidal_curvature':
        amplitude = kwargs.get('curvature_amplitude', 0.1)
        frequency = kwargs.get('frequency', 1.0)
        weights[:,0] = edge_lengths * (1 + amplitude * np.sin(frequency * edge_lengths * np.pi))
        weights[:,1] = edge_lengths
    elif weight_type == 'parabolic_curvature':
        strength = kwargs.get('curvature_strength', 0.05)
        weights[:,0] = edge_lengths * (1 + strength * edge_lengths**2)
        weights[:,1] = edge_lengths
    ''' or bin? # LEON HERE - CALCULATE THE WEIGHT PERCENTAGE TO MULTIPLY BY
    # in case of scaling that isn't directly based on size, we calculate the size difference in edges here after weighting'''
    weights[:, 2] = weights[:, 0] / weights[:, 1]
    # weights = np.clip(weights, 0.0000001, 100.0)
    return weights
# Calculate edge weights based on various schemes (uniform, length-based, centrality-based, etc.) 
def edge_density(N, tURL, using, uselen):
    """Compute total path length in fundamental cell divided by area"""
    runningtotal = 0
    L = N
    squarepoints = np.array([[0,0],[L,0],[L,L],[0,L]])

    for ed, edge in enumerate(using):
        p1 = tURL[edge[0]]
        p2 = tURL[edge[1]]

        # Check if points are inside fundamental cell [0,L] x [0,L]
        p1_inside = (0 <= p1[0] <= L) and (0 <= p1[1] <= L)
        p2_inside = (0 <= p2[0] <= L) and (0 <= p2[1] <= L)

        if p1_inside and p2_inside:
            # Both inside - count full edge length
            runningtotal += uselen[ed]
        elif p1_inside and not p2_inside:
            # Only p1 inside - count from p1 to boundary intersection
            intersection_points = square_line_intersection(squarepoints, [p1, p2])
            if intersection_points:
                # Take closest intersection point
                distances = [np.linalg.norm(p1 - np.array(pt)) for pt in intersection_points]
                min_dist = min(distances)
                runningtotal += min_dist
        elif not p1_inside and p2_inside:
            # Only p2 inside - count from p2 to boundary intersection
            intersection_points = square_line_intersection(squarepoints, [p1, p2])
            if intersection_points:
                distances = [np.linalg.norm(p2 - np.array(pt)) for pt in intersection_points]
                min_dist = min(distances)
                runningtotal += min_dist
        # If neither inside, contribute 0 (implicit)

    return runningtotal / (L**2)

# def edge_density(N, tURL, using, uselen):
#     ###Just going to compute the total path length in the fundamental cell and divide by the volume
#     # would need some different scaling for edge density for weights? how about sinusoidal?
#     runningtotal = 0
#     L = N
#     squarepoints = np.array([[0,0],[L,0],[L,L],[0,L]])
#     ed = 0
#     for edge in using: #BOUNDARY CONDITIONS?
#         if tURL[edge[0]][0] < L and tURL[edge[0]][0] > 0 and tURL[edge[0]][1] < L and tURL[edge[0]][1] > 0:
#             #So we know point 1 is in there
#             if tURL[edge[1]][0] < L and tURL[edge[1]][0] > 0 and tURL[edge[1]][1] < L and tURL[edge[1]][1] > 0:
#                 runningtotal += uselen[ed]
#             else:
#                 runningtotal += np.linalg.norm(tURL[edge[0]]-square_line_intersection(squarepoints, tURL[edge]))
#         elif tURL[edge[1]][0] < L and tURL[edge[1]][0] > 0 and tURL[edge[1]][1] < L and tURL[edge[1]][1] > 0:
#             runningtotal += np.linalg.norm(tURL[edge[1]]-square_line_intersection(squarepoints, tURL[edge]))

#         ed += 1
#     return(runningtotal/L**2)
# Calculate edge density (total edge length per unit area) within fundamental cell

def square_line_intersection(square_bounds, line_seg):
      """Find intersection of line segment with square boundary"""
      p1, p2 = line_seg
      xmin, ymin = 0, 0
      xmax, ymax = square_bounds[2]  # [L, L]

      intersections = []

      # Check intersection with each boundary
      # Left boundary (x = 0)
      if p1[0] != p2[0]:
          t = -p1[0] / (p2[0] - p1[0])
          if 0 <= t <= 1:
              y = p1[1] + t * (p2[1] - p1[1])
              if ymin <= y <= ymax:
                  intersections.append([0, y])

      # Right boundary (x = L)
      if p1[0] != p2[0]:
          t = (xmax - p1[0]) / (p2[0] - p1[0])
          if 0 <= t <= 1:
              y = p1[1] + t * (p2[1] - p1[1])
              if ymin <= y <= ymax:
                  intersections.append([xmax, y])

      # Bottom boundary (y = 0)
      if p1[1] != p2[1]:
          t = -p1[1] / (p2[1] - p1[1])
          if 0 <= t <= 1:
              x = p1[0] + t * (p2[0] - p1[0])
              if xmin <= x <= xmax:
                  intersections.append([x, 0])

      # Top boundary (y = L)
      if p1[1] != p2[1]:
          t = (ymax - p1[1]) / (p2[1] - p1[1])
          if 0 <= t <= 1:
              x = p1[0] + t * (p2[0] - p1[0])
              if xmin <= x <= xmax:
                  intersections.append([x, ymax])

      return intersections
# def square_line_intersection(square_points, line_points):
#     """Finds the intersection points between a square and a line segment"""
#     intersection_points = []

#     for i in range(4):
#         p1 = square_points[i]
#         p2 = square_points[(i + 1) % 4]  # Wrap around to the first point
#         intersection = intersect(p1, p2, line_points[0], line_points[1])
#         if intersection:
#             intersection_points.append(intersection)

#     return intersection_points
# Find intersection points between square boundary and line segment
def intersect(p1, p2, p3, p4):
    """Checks if line segment p1-p2 intersects with line segment p3-p4"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:  # parallel lines
        return None

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    if ua < 0 or ua > 1:  # intersection point not on line segment p1-p2
        return None

    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    if ub < 0 or ub > 1:  # intersection point not on line segment p3-p4
        return None

    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    return (x, y)
# Check if two line segments intersect and return intersection point
# Moved timing and processing to main computation section
N = int(sys.argv[1]) #Square root of the number of particles in the system
a = float(sys.argv[2]) #Parameter describing the degree of local translational disorder in the URL point patterns
simtag = str(sys.argv[3]) #Specific identifying name for particular simulation
ctype = str(sys.argv[4]) #Configuration type
ntype = str(sys.argv[5]) #Tessellation type (C,V,G,D)
weight_type = str(sys.argv[6]) if len(sys.argv) > 6 else 'uniform' #beam characteristics for function calls
params_str = str(sys.argv[7]) if len(sys.argv) > 7 else None
'''If loading a configuration, the path to the file is loaded here
Will read in a .txt file whose lines are the global coordinates of each point in the configuration '''



fn = 0
if ctype == "load":
    fn = str(sys.argv[8])
if ctype == 'poi': flatlat = np.random.rand(N**2,2)*N #Generating/loading point pattern
elif ctype == 'load':  
    flatlat = np.zeros([N**2,2])
    with open(fn, 'r') as f:
        for i in range(N**2):
            flatlat[i] = f.readline().split(' ')
else: 
    ctype = 'URL'
    lat = Z2_gen(N)
    flatlat = np.reshape(lat, [N**2,2])
    URL(a, flatlat)         
if ctype == 'poi': np.savetxt("../ELV_poi_N"+str(N)+"_"+ str(ntype) +"_set"+str(simtag)+"_config.txt", flatlat)
elif ctype == 'URL': np.savetxt("../ELV_URL_a"+str(a)+"_N"+str(N)+"_"+ str(ntype) +"_set"+str(simtag)+"_config.txt", flatlat)
wrap(N, flatlat) #Making sure every point is inside the simulation box and making the nearest neighbor images
tURL = tile(flatlat,N)
if ntype == 'D': #Generating list of edges from the choice of spatial tessellation
    using = np.array(delaunay_graph_2d(tURL)[:,:2], dtype=int)
elif ntype == 'G':
    using = np.array(gabriel_graph_2d(tURL)[:,:2], dtype=int)
elif ntype == 'V':
    vor = sp.spatial.Voronoi(tURL)
    using = vor.ridge_vertices
    using = remove_duplicate_rows(np.array(using))
    using = using[np.argwhere(np.sum(using>=0,axis=1)==2).flatten()]
    tURL = vor.vertices
elif ntype == 'C':
    tri = sp.spatial.Delaunay(tURL)
    newpoints = np.zeros([tri.simplices.shape[0],2])
    for i in range(newpoints.shape[0]):
        newpoints[i] = np.mean(tURL[tri.simplices[i]], axis = 0)
    tURL = newpoints
    using = []
    for i in range(tri.simplices.shape[0]):
        for j in range(3):
            if tri.neighbors[i][j] != -1: using.append([i,tri.neighbors[i][j]])
    using = remove_duplicate_rows(np.array(using))  
using = np.array(using)
'''
EDGE WEIGHT CALCULATION 
'''
edge_weights = None
if weight_type != 'uniform':
    # Prepare parameters for different length-based schemes
    length_params = {
        'correlation_length': N/10,
        'disorder_strength': 0.3,
        'curvature_amplitude': 0.1,
        # Power law parameters
        'alpha': 1.5,
        'beta': 1.0,
        'gamma': 0.5,
        'delta': 2.0,
        # Binning parameters
        'n_bins': 4,
        'bin_weights': [0.2, 0.7, 1.3, 2.5],  # short to long edge weights
        # Threshold parameters
        'short_weight': 0.3,
        'long_weight': 2.5,
        # Normalization parameters
        'scale': 3.0,
        'offset': 0.1,
        # Gaussian kernel parameters
        'amplitude': 1.5,
        # Other parameters
        'epsilon': 1e-6
        }
    if params_str is not None:
        for param in params_str.split(','):
            key, value = param.split('=')
            length_params[key] = float(value)
    
    edge_weights = calculate_edge_weights(tURL, using, weight_type, **length_params)
else: pass
''' else:
     edge_weights[:,0] = np.ones(len(using))'''
Nwind = 250 ###Settings to change the number of windows used in the calculation or resolution in window radius
reso = 500
wind = np.random.rand(Nwind,2)*N
Rs = np.logspace(-2,np.log10(N/4),reso)
# Calculate edge lengths with weights
uselen = np.zeros(using.shape[0])
uselen_w = np.zeros(using.shape[0])
for i in range(using.shape[0]):
    base_length = np.linalg.norm(tURL[using[i,0]]-tURL[using[i,1]])
    uselen[i] = base_length
    if edge_weights is not None:
        uselen_w[i] = base_length * edge_weights[i,2]
    else:
        uselen_w[i] = base_length
    # if edge_weights is not None:
    #     uselen[i] = edge_weights[i,2]
    #     # assert base_length == edge_weights[i,1]
    #     # uselen_w[i] = base_length * edge_weights[i,0]
    #     # unweighted[i] = base_length
    # else:
    #     uselen[i] = base_length
''' option with single matrix
# uselen = np.zeros(using.shape[2]) # 0: unweighted 1: weighted 2: weight
# for i in range(using.shape[0]):
#     base_length = np.linalg.norm(tURL[using[i,0]]-tURL[using[i,1]])
#     if edge_weights is not None:
#         uselen[i,0] = base_length * edge_weights[i]
#         unweighted = base_length
#     else:
#         uselen[i] = base_length'''
# Main computation timing
start_time = time.time() 
process = psutil.Process()

# Main ELV computation with optimizations
tovar = np.zeros([3,Nwind, reso])
if edge_weights is not None:
    for i in range(Nwind):
        distto = np.zeros(using.shape[0])
        verttowind = np.zeros(tURL.shape[0])
        
        # Vectorized distance calculations where possible
        for j in range(using.shape[0]):
            distto[j] = mindist(tURL[using[j,0]], tURL[using[j,1]], wind[i])
        for j in range(tURL.shape[0]):
            diff = wind[i] - tURL[j]
            verttowind[j] = np.sqrt(np.sum(diff**2))
            
        for j in range(Rs.shape[0]):
            etc = np.where(distto < Rs[j])[0]
            for k in etc:
                tr = eclen(wind[i], Rs[j], tURL[using[k,0]], verttowind[using[k,0]], tURL[using[k,1]], verttowind[using[k,1]], uselen[k])
                percent_inside = tr / uselen[k] if uselen[k] > 0 else 0.0
                tovar[0,i,j] += tr
                tovar[1,i,j] += percent_inside
                tovar[2,i,j] += edge_weights[k,2]
else:
    for i in range(Nwind):
        distto = np.zeros(using.shape[0])
        verttowind = np.zeros(tURL.shape[0])
        
        for j in range(using.shape[0]):
            distto[j] = mindist(tURL[using[j,0]], tURL[using[j,1]], wind[i])
        for j in range(tURL.shape[0]):
            diff = wind[i] - tURL[j]
            verttowind[j] = np.sqrt(np.sum(diff**2))
            
        for j in range(Rs.shape[0]):
            etc = np.where(distto < Rs[j])[0]
            for k in etc:
                tr = eclen(wind[i], Rs[j], tURL[using[k,0]], verttowind[using[k,0]], tURL[using[k,1]], verttowind[using[k,1]], uselen[k])
                tovar[0,i,j] += tr
# Calculate edge densities after main computation
weighted_edge_density = edge_density(N, tURL, using, uselen_w) 
edge_density_n = edge_density(N, tURL, using, uselen)
def plot_elv_variance(Rs, tovar, edge_density, weighted_edge_density, N, ctype, ntype, simtag, weight_type):
    """
    Plot edge length variance vs proper dimensionless radius using edge density

    Parameters:
    - Rs: window radii array
    - tovar: variance matrix [Nwind, reso]
    - weighted_edge_density: output from edge_density() function
    - N, ctype, ntype, simtag, weight_type, curvature_type: configuration info
    """
    # Calculate variance across windows for each radius
    rho_l_w = weighted_edge_density
    rho_l_n = edge_density_n
    dimensionless_R_w = Rs * rho_l_w^(1/2)
    dimensionless_R = Rs * rho_l_n^(1/2)
    plt.figure(figsize=(8, 6))
    plt.grid(True, alpha=0.3)
    plt.xlabel(r'$R \rho_\ell^{1/2}$', fontsize=12)
    plt.ylabel('Edge Length Variance', fontsize=12)
    plt.title(f'ELV: {ctype.upper()} N={N} {ntype} wt={weight_type} rho_l={rho_l_w:.3f}', fontsize=10)
    if tovar.shape[0] >= 3:  # Has weight data
        unweighted_var = np.var(tovar[0], axis=0)
        weighted_var = np.var(tovar[0] * tovar[2], axis=0)
        # Plot both
        plt.loglog(dimensionless_R, unweighted_var, 'b-', label='Unweighted')
        plt.loglog(dimensionless_R_w, weighted_var, 'r-', label=f'Weighted ({weight_type})')
        plt.legend()
    else:
        variances = np.var(tovar[0], axis=0)
        plt.loglog(dimensionless_R, variances, 'o-', linewidth=2, markersize=4)
    filename = f"../ELV_{ctype}_N{N}_{ntype}_set{simtag}_wt{weight_type}_plot.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return filename
# Plot edge length variance vs dimensionless radius with optional weighted comparison
if ctype == 'load': 
    np.save("../ELV_ld_N"+str(N)+"_"+ str(ntype) +"_set"+str(simtag)+"_wt"+str(weight_type), tovar)
    np.save("../ELV_ld_N"+str(N)+"_"+ str(ntype) +"_set"+str(simtag)+"_Rs", Rs)
    np.save("../ELV_ld_N"+str(N)+"_"+ str(ntype) +"_set"+str(simtag)+"_density", weighted_edge_density)
elif ctype == 'poi': 
    np.save("../ELV_poi_N"+str(N)+"_"+ str(ntype) +"_set"+str(simtag)+"_wt"+str(weight_type), tovar)
    # np.save("../ELV_poi_N"+str(N)+"_"+ str(ntype) +"_set"+str(simtag), tovar)
    np.save("../ELV_poi_N"+str(N)+"_"+ str(ntype) +"_set"+str(simtag)+"_Rs", Rs)
    np.save("../ELV_poi_N"+str(N)+"_"+ str(ntype) +"_set"+str(simtag)+"_density", weighted_edge_density)
elif ctype == 'URL': 
    np.save("../ELV_URL_a"+str(a)+"_N"+str(N)+"_"+ str(ntype) +"_set"+str(simtag)+"_wt"+str(weight_type), tovar)
    np.save("../ELV_URL_a"+str(a)+"_N"+str(N)+"_"+ str(ntype) +"_set"+str(simtag)+"_Rs", Rs)
    np.save("../ELV_URL_a"+str(a)+"_N"+str(N)+"_"+ str(ntype) +"_set"+str(simtag)+"_density", weighted_edge_density)
plot_filename = plot_elv_variance(Rs, tovar, edge_density_n, weighted_edge_density, N, ctype, ntype, simtag, weight_type)
print(f"ELV plot saved to: {plot_filename}")
end_time = time.time()
computation_time = end_time - start_time
memory_usage = process.memory_info().rss / 1024 / 1024  # MB
print(f"Computation completed in {computation_time:.2f} seconds")
print(f"Peak memory usage: {memory_usage:.2f} MB")
print("Saving results to database...")
try:
    db = NetworkResultsDB("../network_results_weights.db")
    
    # Prepare weight parameters for database
    if edge_weights is not None:
        # Get the weight parameters used
        weight_params = {
            'alpha': length_params.get('alpha', 1.5),
            'beta': length_params.get('beta', 1.0),
            'bin_weights': length_params.get('bin_weights', [0.2, 0.7, 1.3, 2.5]),
            'curvature_amplitude': length_params.get('curvature_amplitude', 0.1),
            'curvature_strength': length_params.get('curvature_strength', 0.05)
        }
    else:
        weight_params = {}
    
    # Save configuration
    config_id = db.save_configuration(
        N=N, a=a, simname=simtag, config_type=ctype, tess_type=ntype, 
        dimension=2, weight_type=weight_type, weight_parameters=weight_params,
        point_pattern=flatlat
    )
    
    # Save ELV results  
    db.save_elv_results(
        config_id=config_id,
        window_radii=Rs,
        variance_matrix=tovar,
        edge_density=weighted_edge_density,
        num_windows=Nwind,
        resolution=reso
    )
    
    # Save computation metadata
    metadata = {
        'Nwind': Nwind,
        'reso': reso,
        'weight_scheme': weight_type,
        'edge_count': len(using),
        'plot_filename': plot_filename,
        'memory_usage_mb': memory_usage
    }
    
    db.save_computation_metadata(
        config_id=config_id,
        computation_type='ELV2D_weighted',
        computation_time=computation_time,
        parameters=metadata
    )
    
    print(f"Results saved to database with config_id: {config_id}")
except Exception as e:
    print(f"Database save failed: {e}")
    print("Results still saved to .npy files")