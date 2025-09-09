#!/usr/bin/env python3
"""
Edge Length Variance (ELV) Analysis for 2D Networks
Analyzes hyperuniformity in spatial networks using weighted/unweighted edge schemes
"""

import sys
import os
import time
import numpy as np
import scipy as sp
from numba import njit
import matplotlib.pyplot as plt
import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_utils import NetworkResultsDB
from backup_database import DatabaseBackupManager

# Core Geometric Functions

@njit(fastmath=True)
def URL(a, Z2):
    """Apply Uniformly Random Local disorder to lattice points"""
    for i in range(Z2.shape[0]):
        shift = (np.random.rand(2) - 0.5) * a
        Z2[i] += shift

def tile(flatlat, N):
    """Create 3x3 tiled array with periodic boundary conditions"""
    tiled = np.empty([(flatlat.shape[0])*9, 2])
    im = 0
    for i in [0, -1, 1]:
        for j in [0, -1, 1]:
            tiled[im * flatlat.shape[0]:(im * flatlat.shape[0])+flatlat.shape[0]] = flatlat + np.array([i*N, j*N])
            im += 1
    return tiled

@njit(fastmath=True)
def wrap(N, flatlat):
    """Wrap points into simulation box using periodic boundaries"""
    for i in range(flatlat.shape[0]):
        if flatlat[i,0] < 0: flatlat[i,0] += N
        if flatlat[i,0] > N: flatlat[i,0] -= N
        if flatlat[i,1] < 0: flatlat[i,1] += N
        if flatlat[i,1] > N: flatlat[i,1] -= N

@njit(fastmath=True)
def mindist(ep1, ep2, wind):
    """Calculate minimum distance from point to line segment"""
    normsq = np.sum((ep1-ep2)**2)
    t = max(0, min(1, np.dot(wind - ep1, ep2-ep1)/normsq))
    proj = ep1 + t * (ep2 - ep1)
    return np.sum((wind-proj)**2)**0.5

@njit(fastmath=True)
def Z2_gen(N):
    """Generate square lattice with N sites per direction"""
    hold = np.zeros((N, N, 2))
    for i in range(N):
        hold[i, :, 0] = np.linspace(0, N-1, N)
    for j in range(N):
        hold[:, j, 1] = np.linspace(0, N-1, N)
    return hold

# Circle-Line Intersection Functions

def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2, full_line=False, tangent_tol=1e-9):
    """Find intersection points between circle and line segment"""
    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center 
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy) 
    dx, dy = (x2 - x1), (y2 - y1) 
    dr = (dx ** 2 + dy ** 2)**.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

    if discriminant < 0:
        return []
    else:
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]
        if not full_line:
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:
            return [intersections[0]]
        else:
            return intersections

def eclen(cent, rad, v1, v1len, v2, v2len, edgelen):
    """Calculate edge length inside circular window"""
    if v1len < rad and v2len < rad: 
        return edgelen
    else:
        res = np.array(circle_line_segment_intersection(cent, rad, v1, v2))
        if len(res) == 2: 
            return np.linalg.norm(res[0]-res[1])
        elif v1len < rad: 
            return np.linalg.norm(res - v1)
        else: 
            return np.linalg.norm(res-v2)

# Graph Generation Functions

def gabriel_graph_2d(points):
    """Generate Gabriel graph edges from 2D point set"""
    tree = sp.spatial.cKDTree(points)
    tri = sp.spatial.Delaunay(points)
    simplices = tri.simplices

    if simplices.shape[1] == 3:
        edges = np.vstack([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]])
    elif simplices.shape[1] == 4:
        edges = np.vstack([simplices[:, [0, 1]], simplices[:, [0, 2]], simplices[:, [0, 3]], 
                          simplices[:, [1, 2]], simplices[:, [1, 3]], simplices[:, [2, 3]]])
    else:
        raise ValueError('Input points must be 2D or 3D.')

    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    i = edges[:, 0]
    j = edges[:, 1]
    points_i = points[i]
    points_j = points[j]
    midpoints = (points_i + points_j) / 2
    radii = 0.5 * np.linalg.norm(points_i - points_j, axis=1)
    weights = 2 * radii

    idx_list = tree.query_ball_point(midpoints, radii)
    edge_list = []
    for k in range(len(edges)):
        idx = set(idx_list[k]) - {i[k], j[k]}
        if not idx:
            edge_list.append([i[k], j[k], weights[k]])

    return np.array(edge_list)

def remove_duplicate_rows(arr):
    """Remove duplicate edges handling both [i,j] and [j,i]"""
    dtype = [('min', arr.dtype), ('max', arr.dtype)]
    structured = np.empty(arr.shape[0], dtype=dtype)
    structured['min'] = np.minimum(arr[:, 0], arr[:, 1])
    structured['max'] = np.maximum(arr[:, 0], arr[:, 1])
    _, unique_indices = np.unique(structured, return_index=True)
    unique_indices.sort()
    return arr[unique_indices]

def restructure_array_numpy(triangles):
    """Convert triangle array to edge array"""
    reshaped = triangles
    output = np.empty((len(reshaped) * 3, 2), dtype=triangles.dtype)
    output[::3, 0] = reshaped[:, 0]
    output[::3, 1] = reshaped[:, 1]
    output[1::3, 0] = reshaped[:, 1]
    output[1::3, 1] = reshaped[:, 2]
    output[2::3, 0] = reshaped[:, 2]
    output[2::3, 1] = reshaped[:, 0]
    return output

def delaunay_graph_2d(points):
    """Generate Delaunay triangulation edges"""
    delaunay = sp.spatial.Delaunay(points)
    triangles = np.array(delaunay.simplices)
    edges = restructure_array_numpy(triangles)
    return remove_duplicate_rows(edges)

# Edge Weighting Functions

def calculate_edge_weights(tURL, edges, weight_type='uniform', **kwargs):
    """Calculate edge weights based on various schemes"""
    n_of_edges = len(edges)
    weights = np.ones((n_of_edges, 3))

    if weight_type == 'uniform':
        return weights
    
    elif weight_type == 'length_power_law':
        alpha = kwargs.get('alpha', 2.0)
        for i in range(n_of_edges):
            edge_length = np.linalg.norm(tURL[edges[i,0]] - tURL[edges[i,1]])
            weights[i,0] = edge_length ** alpha
            weights[i,1] = edge_length
    
    elif weight_type == 'length_inverse':
        beta = kwargs.get('beta', 1.0)
        for i in range(n_of_edges):
            edge_length = np.linalg.norm(tURL[edges[i,0]] - tURL[edges[i,1]])
            weights[i,0] = edge_length ** (-beta)
            weights[i,1] = edge_length
    
    elif weight_type == 'length_binned':
        lengths = [np.linalg.norm(tURL[edges[i,0]] - tURL[edges[i,1]]) for i in range(n_of_edges)]
        percentiles = np.percentile(lengths, [25, 50, 75])
        bin_weights = kwargs.get('bin_weights', [0.2, 0.7, 1.3, 2.5])
        for i in range(n_of_edges):
            edge_length = lengths[i]
            if edge_length <= percentiles[0]:
                weights[i,0] = edge_length * bin_weights[0]
            elif edge_length <= percentiles[1]:
                weights[i,0] = edge_length * bin_weights[1]
            elif edge_length <= percentiles[2]:
                weights[i,0] = edge_length * bin_weights[2]
            else:
                weights[i,0] = edge_length * bin_weights[3]
            weights[i,1] = edge_length
    
    elif weight_type == 'length_threshold':
        lengths = [np.linalg.norm(tURL[edges[i,0]] - tURL[edges[i,1]]) for i in range(n_of_edges)]
        threshold = kwargs.get('threshold', np.median(lengths))
        short_weight = kwargs.get('short_weight', 0.3)
        long_weight = kwargs.get('long_weight', 2.5)
        for i in range(n_of_edges):
            edge_length = lengths[i]
            if edge_length <= threshold:
                weights[i,0] = edge_length * short_weight
            else:
                weights[i,0] = edge_length * long_weight
            weights[i,1] = edge_length
    
    elif weight_type == 'betweenness_centrality':
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
        G = nx.Graph()
        G.add_edges_from(edges)
        node_centrality = nx.closeness_centrality(G)
        for i in range(n_of_edges):
            edge_length = np.linalg.norm(tURL[edges[i,0]] - tURL[edges[i,1]])
            avg_centrality = (node_centrality[edges[i,0]] + node_centrality[edges[i,1]]) / 2
            weights[i,0] = edge_length * (1 + avg_centrality)
            weights[i,1] = edge_length
    
    elif weight_type == 'degree_centrality':
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
        for i in range(n_of_edges):
            edge_length = np.linalg.norm(tURL[edges[i,0]] - tURL[edges[i,1]])
            weights[i,0] = edge_length * (1 + amplitude * np.sin(frequency * edge_length * np.pi))
            weights[i,1] = edge_length
    
    elif weight_type == 'parabolic_curvature':
        strength = kwargs.get('curvature_strength', 0.05)
        for i in range(n_of_edges):
            edge_length = np.linalg.norm(tURL[edges[i,0]] - tURL[edges[i,1]])
            weights[i,0] = edge_length * (1 + strength * edge_length**2)
            weights[i,1] = edge_length

    weights[:, 2] = weights[:, 0] / weights[:, 1]
    return weights

# Edge Density Calculation

def edge_density(N, tURL, using, uselen):
    """Calculate edge density (total edge length per unit area) within fundamental cell"""
    runningtotal = 0
    L = N
    squarepoints = np.array([[0,0],[L,0],[L,L],[0,L]])
    ed = 0
    for edge in using:
        if tURL[edge[0]][0] < L and tURL[edge[0]][0] > 0 and tURL[edge[0]][1] < L and tURL[edge[0]][1] > 0:
            if tURL[edge[1]][0] < L and tURL[edge[1]][0] > 0 and tURL[edge[1]][1] < L and tURL[edge[1]][1] > 0:
                runningtotal += uselen[ed]
            else:
                runningtotal += np.linalg.norm(tURL[edge[0]]-square_line_intersection(squarepoints, tURL[edge]))
        elif tURL[edge[1]][0] < L and tURL[edge[1]][0] > 0 and tURL[edge[1]][1] < L and tURL[edge[1]][1] > 0:
            runningtotal += np.linalg.norm(tURL[edge[1]]-square_line_intersection(squarepoints, tURL[edge]))
        ed += 1
    return runningtotal/L**2

def square_line_intersection(square_points, line_points):
    """Find intersection points between square boundary and line segment"""
    intersection_points = []
    for i in range(4):
        p1 = square_points[i]
        p2 = square_points[(i + 1) % 4]
        intersection = intersect(p1, p2, line_points[0], line_points[1])
        if intersection:
            intersection_points.append(intersection)
    return intersection_points

def intersect(p1, p2, p3, p4):
    """Check if two line segments intersect and return intersection point"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return None

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    if ua < 0 or ua > 1:
        return None

    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    if ub < 0 or ub > 1:
        return None

    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    return (x, y)

# Plotting Functions

def plot_elv_variance(Rs, tovar, edge_density, weighted_edge_density, N, ctype, ntype, simtag, weight_type, weight_params=None):
    """Plot edge length variance vs dimensionless radius"""
    rho_l_w = weighted_edge_density
    rho_l_n = edge_density
    dimensionless_R_w = Rs * rho_l_w
    dimensionless_R = Rs * rho_l_n
    
    plt.figure(figsize=(8, 6))
    plt.grid(True, alpha=0.3)
    plt.xlabel(r'$R \rho_\ell$', fontsize=12)
    plt.ylabel('Edge Length Variance', fontsize=12)
    plt.title(f'ELV: {ctype.upper()} N={N} {ntype} wt={weight_type}\\nrho_w={rho_l_w:.3f}, rho_u={rho_l_n:.3f}', fontsize=10)
    
    if tovar.shape[0] >= 3:
        unweighted_var = np.var(tovar[0], axis=0)
        weighted_var = np.var(tovar[2], axis=0)
        
        weight_detail = ""
        if weight_params:
            if weight_type == 'length_power_law':
                alpha = weight_params.get('alpha', 1.5)
                weight_detail = f" L^{alpha}"
            elif weight_type == 'length_inverse':
                beta = weight_params.get('beta', 1.0)  
                weight_detail = f" L^-{beta}"
            elif weight_type == 'sinusoidal_curvature':
                weight_detail = " sin(L)"
            elif weight_type == 'parabolic_curvature':
                weight_detail = " L²"
            
        plt.loglog(dimensionless_R, unweighted_var, 'b-', label=f'Unweighted (ρ={rho_l_n:.3f})')
        plt.loglog(dimensionless_R_w, weighted_var, 'r-', label=f'Weighted{weight_detail} (ρ={rho_l_w:.3f})')
        plt.legend()
    else:
        variances = np.var(tovar[0], axis=0)
        plt.loglog(dimensionless_R, variances, 'o-', linewidth=2, markersize=4)
    
    filename = f"../outputs/plots/ELV_{ctype}_N{N}_{ntype}_set{simtag}_wt{weight_type}_plot.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return filename

# Main Computation

if __name__ == "__main__":
    start_time = time.time()
    
    # Parse command line arguments
    N = int(sys.argv[1])
    a = float(sys.argv[2])
    simtag = str(sys.argv[3])
    ctype = str(sys.argv[4])
    ntype = str(sys.argv[5])
    weight_type = str(sys.argv[6]) if len(sys.argv) > 6 else 'uniform'
    params_str = str(sys.argv[7]) if len(sys.argv) > 7 else None

    # Generate or load point pattern
    fn = 0
    if ctype == "load":
        fn = str(sys.argv[8])
        
    if ctype == 'poi': 
        flatlat = np.random.rand(N**2, 2) * N
    elif ctype == 'load':  
        flatlat = np.zeros([N**2, 2])
        with open(fn, 'r') as f:
            for i in range(N**2):
                flatlat[i] = f.readline().split(' ')
    else: 
        ctype = 'URL'
        lat = Z2_gen(N)
        flatlat = np.reshape(lat, [N**2, 2])
        URL(a, flatlat)
    
    # Ensure output directories exist
    os.makedirs("../outputs/data", exist_ok=True)
    os.makedirs("../outputs/plots", exist_ok=True)
    
    # Save configuration
    if ctype == 'poi': 
        np.savetxt(f"../outputs/data/ELV_poi_N{N}_{ntype}_set{simtag}_config.txt", flatlat)
    elif ctype == 'URL': 
        np.savetxt(f"../outputs/data/ELV_URL_a{a}_N{N}_{ntype}_set{simtag}_config.txt", flatlat)
    
    wrap(N, flatlat)
    tURL = tile(flatlat, N)
    
    # Generate tessellation
    if ntype == 'D':
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
        newpoints = np.zeros([tri.simplices.shape[0], 2])
        for i in range(newpoints.shape[0]):
            newpoints[i] = np.mean(tURL[tri.simplices[i]], axis = 0)
        tURL = newpoints
        using = []
        for i in range(tri.simplices.shape[0]):
            for j in range(3):
                if tri.neighbors[i][j] != -1: 
                    using.append([i, tri.neighbors[i][j]])
        using = remove_duplicate_rows(np.array(using))  
    using = np.array(using)

    # Calculate edge weights
    edge_weights = None
    if weight_type != 'uniform':
        length_params = {
            'correlation_length': N/10, 'disorder_strength': 0.3, 'curvature_amplitude': 0.1,
            'alpha': 1.5, 'beta': 1.0, 'gamma': 0.5, 'delta': 2.0,
            'n_bins': 4, 'bin_weights': [0.2, 0.7, 1.3, 2.5],
            'short_weight': 0.3, 'long_weight': 2.5,
            'scale': 3.0, 'offset': 0.1, 'amplitude': 1.5, 'epsilon': 1e-6
        }
        if params_str is not None:
            for param in params_str.split(','):
                key, value = param.split('=')
                length_params[key] = float(value)
        
        edge_weights = calculate_edge_weights(tURL, using, weight_type, **length_params)

    # ELV calculation setup
    Nwind = 250
    reso = 500
    wind = np.random.rand(Nwind, 2) * N
    Rs = np.logspace(-2, np.log10(N/4), reso)
    
    uselen = np.zeros(using.shape[0])
    uselen_w = np.zeros(using.shape[0])
    
    for i in range(using.shape[0]):
        base_length = np.linalg.norm(tURL[using[i,0]]-tURL[using[i,1]])
        if edge_weights is not None:
            uselen[i] = base_length
            uselen_w[i] = base_length * edge_weights[i,2]
        else:
            uselen[i] = base_length
            uselen_w[i] = base_length

    # ELV computation
    tovar = np.zeros([3, Nwind, reso])
    
    if edge_weights is not None:
        for i in range(Nwind): #update to keep taack of percent edge weight * percent in window per edge in window not per window
            distto = np.zeros(using.shape[0])
            verttowind = np.zeros(tURL.shape[0])
            for j in range(using.shape[0]):
                distto[j] = mindist(tURL[using[j,0]], tURL[using[j,1]], wind[i])
            for j in range(verttowind.shape[0]):
                verttowind[j] = np.linalg.norm(wind[i]-tURL[j])
            for j in range(Rs.shape[0]):
                etc = np.argwhere(distto < Rs[j]).flatten()
                for k in etc:
                    tr = eclen(wind[i], Rs[j], tURL[using[k,0]], verttowind[using[k,0]], tURL[using[k,1]], verttowind[using[k,1]], uselen[k])
                    percent_inside = tr / uselen[k]
                    weighted_contribution = percent_inside * edge_weights[k,2] * uselen[k]
                    tovar[0,i,j] += tr
                    tovar[1,i,j] += percent_inside
                    tovar[2,i,j] += weighted_contribution
    else:
        for i in range(Nwind):
            distto = np.zeros(using.shape[0])
            verttowind = np.zeros(tURL.shape[0])
            for j in range(using.shape[0]):
                distto[j] = mindist(tURL[using[j,0]], tURL[using[j,1]], wind[i])
            for j in range(verttowind.shape[0]):
                verttowind[j] = np.linalg.norm(wind[i]-tURL[j])
            for j in range(Rs.shape[0]):
                etc = np.argwhere(distto < Rs[j]).flatten()
                for k in etc:
                    tr = eclen(wind[i], Rs[j], tURL[using[k,0]], verttowind[using[k,0]], tURL[using[k,1]], verttowind[using[k,1]], uselen[k])
                    tovar[0,i,j] += tr

    # Calculate edge densities
    weighted_edge_density = edge_density(N, tURL, using, uselen_w)
    edge_density_n = edge_density(N, tURL, using, uselen)

    # Save data files
    if ctype == 'load': 
        np.save(f"../outputs/data/ELV_ld_N{N}_{ntype}_set{simtag}_wt{weight_type}", tovar)
        np.save(f"../outputs/data/ELV_ld_N{N}_{ntype}_set{simtag}_Rs", Rs)
        np.save(f"../outputs/data/ELV_ld_N{N}_{ntype}_set{simtag}_density", weighted_edge_density)
    elif ctype == 'poi': 
        np.save(f"../outputs/data/ELV_poi_N{N}_{ntype}_set{simtag}_wt{weight_type}", tovar)
        np.save(f"../outputs/data/ELV_poi_N{N}_{ntype}_set{simtag}_Rs", Rs)
        np.save(f"../outputs/data/ELV_poi_N{N}_{ntype}_set{simtag}_density", weighted_edge_density)
    elif ctype == 'URL': 
        np.save(f"../outputs/data/ELV_URL_a{a}_N{N}_{ntype}_set{simtag}_wt{weight_type}", tovar)
        np.save(f"../outputs/data/ELV_URL_a{a}_N{N}_{ntype}_set{simtag}_Rs", Rs)
        np.save(f"../outputs/data/ELV_URL_a{a}_N{N}_{ntype}_set{simtag}_density", weighted_edge_density)

    # Generate plot
    plot_filename = plot_elv_variance(Rs, tovar, edge_density_n, weighted_edge_density, N, ctype, ntype, simtag, weight_type, 
                                     length_params if edge_weights is not None else None)
    print(f"ELV plot saved to: {plot_filename}")

    end_time = time.time()
    computation_time = end_time - start_time

    # Save to database
    print("Saving results to database...")
    try:
        db_path = "../network_results_weights.db"
        db = NetworkResultsDB(db_path)
        
        if edge_weights is not None:
            weight_params = {
                'alpha': length_params.get('alpha', 1.5),
                'beta': length_params.get('beta', 1.0),
                'bin_weights': length_params.get('bin_weights', [0.2, 0.7, 1.3, 2.5]),
                'curvature_amplitude': length_params.get('curvature_amplitude', 0.1),
                'curvature_strength': length_params.get('curvature_strength', 0.05)
            }
        else:
            weight_params = {}
        
        config_id = db.save_configuration(
            N=N, a=a, simname=simtag, config_type=ctype, tess_type=ntype, 
            dimension=2, weight_type=weight_type, weight_parameters=weight_params,
            point_pattern=flatlat
        )
        #variance_matrix should store the per edge data for percent inside * percent weighted
        db.save_elv_results(
            config_id=config_id,
            window_radii=Rs,
            variance_matrix=tovar,
            weighted_edge_density=weighted_edge_density,
            unweighted_edge_density=edge_density_n,
            num_windows=Nwind,
            resolution=reso
        )
        
        metadata = {
            'Nwind': Nwind,
            'reso': reso,
            'weight_scheme': weight_type,
            'edge_count': len(using),
            'plot_filename': plot_filename
        }
        
        db.save_computation_metadata(
            config_id=config_id,
            computation_type='ELV2D_weighted',
            computation_time=computation_time,
            parameters=metadata
        )
        
        print(f"Results saved to database with config_id: {config_id}")
        print(f"Computation time: {computation_time:.2f} seconds")
        
        # Automatic database backup (every few runs)
        try:
            backup_manager = DatabaseBackupManager(db_path)
            backup_path = backup_manager.create_backup(force=False, compress=True)
            if backup_path:
                print(f"Database backup created: {backup_path.name}")
        except Exception as backup_error:
            print(f"Backup warning: {backup_error}")
            
    except Exception as e:
        print(f"Database save failed: {e}")
        print("Results still saved to .npy files")