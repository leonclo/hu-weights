#!/usr/bin/env python3
"""
Group comparison plots from ELV analysis database
Generates hyperuniformity comparison visualizations across weight schemes and configurations
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import sqlite3
import pandas as pd
from db_utils import NetworkResultsDB
from backup_database import DatabaseBackupManager

def _timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

def create_weight_comparison_plots(db_path="../network_results_weights.db"):
    """Generate comparison plots across different weight schemes"""
    
    # Ensure output directories exist
    os.makedirs("../outputs/plots", exist_ok=True)
    
    db = NetworkResultsDB(db_path)
    configs = db.get_configurations(dimension=2)
    
    if configs.empty:
        print("No configurations found in database")
        return
    
    print(f"Found {len(configs)} configurations in database")
    
    # Group by tessellation and configuration type for comparison
    for tess_type in configs['tess_type'].unique():
        for config_type in configs['config_type'].unique():
            
            subset = configs[
                (configs['tess_type'] == tess_type) & 
                (configs['config_type'] == config_type)
            ]
            
            if len(subset) < 2:
                continue
                
            weight_groups = subset.groupby('weight_type')
            
            if len(weight_groups) < 2:
                continue
                
            # Create comparison plot
            plt.figure(figsize=(12, 8))
            
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
            color_idx = 0
            
            for weight_type, group in weight_groups:
                
                config_ids = group['id'].tolist()
                
                # Average across realizations
                all_unweighted_variances = []
                all_weighted_variances = []
                Rs = None
                weighted_edge_densities = []
                unweighted_edge_densities = []
                
                for config_id in config_ids:
                    window_radii, variance_matrix, num_windows, resolution = db.get_elv_results(config_id)
                    
                    if window_radii is not None and variance_matrix is not None:
                        if Rs is None:
                            Rs = window_radii
                        
                        if len(variance_matrix.shape) == 3:
                            unweighted_var = np.var(variance_matrix[0], axis=0)
                            weighted_var = np.var(variance_matrix[2], axis=0)
                            all_weighted_variances.append(weighted_var)
                        else:
                            unweighted_var = np.var(variance_matrix, axis=0)
                        
                        all_unweighted_variances.append(unweighted_var)
                        
                        # Get edge densities from database
                        try:
                            with sqlite3.connect(db.db_path) as conn:
                                cursor = conn.cursor()
                                cursor.execute("""
                                    SELECT weighted_edge_density, unweighted_edge_density 
                                    FROM edge_length_variance_results
                                    WHERE config_id = ?
                                """, (config_id,))
                                result = cursor.fetchone()
                                
                                if result and result[0] is not None and result[1] is not None:
                                    weighted_edge_densities.append(result[0])
                                    unweighted_edge_densities.append(result[1])
                                else:
                                    # Fallback to old edge_density column
                                    cursor.execute("""
                                        SELECT edge_density FROM edge_length_variance_results
                                        WHERE config_id = ?
                                    """, (config_id,))
                                    fallback = cursor.fetchone()
                                    if fallback:
                                        weighted_edge_densities.append(fallback[0])
                                        unweighted_edge_densities.append(fallback[0])
                        except Exception as e:
                            print(f"Database query error for config_id {config_id}: {e}")
                            continue
                
                if all_unweighted_variances:
                    # Calculate dimensionless radius using average densities
                    avg_weighted_density = np.mean(weighted_edge_densities) if weighted_edge_densities else 1.0
                    avg_unweighted_density = np.mean(unweighted_edge_densities) if unweighted_edge_densities else 1.0
                    
                    dimensionless_R_weighted = Rs * avg_weighted_density
                    dimensionless_R_unweighted = Rs * avg_unweighted_density
                    
                    # Average variances across realizations
                    mean_unweighted_variance = np.mean(all_unweighted_variances, axis=0)
                    
                    # Plot unweighted variance
                    plt.loglog(dimensionless_R_unweighted, mean_unweighted_variance, 
                             'o-', color=colors[color_idx % len(colors)], alpha=0.7,
                             label=f'{weight_type} (unweighted, ρ={avg_unweighted_density:.3f})', 
                             linewidth=2, markersize=4)
                    
                    # Plot weighted variance if available
                    if all_weighted_variances:
                        mean_weighted_variance = np.mean(all_weighted_variances, axis=0)
                        plt.loglog(dimensionless_R_weighted, mean_weighted_variance, 
                                 's--', color=colors[color_idx % len(colors)], alpha=0.9,
                                 label=f'{weight_type} (weighted, ρ={avg_weighted_density:.3f})', 
                                 linewidth=2, markersize=4)
                
                color_idx += 1
            
            plt.xlabel(r'$R \rho_\ell$', fontsize=14)
            plt.ylabel('Edge Length Variance', fontsize=14)
            plt.title(f'ELV Comparison: {config_type.upper()}-{tess_type}', fontsize=16)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            timestamp = _timestamp()
            filename = f"../outputs/plots/hyperuniformity_comparison_{config_type}_{tess_type}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Comparison plot saved: {filename}")

def create_summary_plot(db_path="../network_results_weights.db"):
    """Create summary plot showing all configurations"""
    
    # Ensure output directories exist
    os.makedirs("../outputs/plots", exist_ok=True)
    
    db = NetworkResultsDB(db_path)
    configs = db.get_configurations(dimension=2)
    
    if configs.empty:
        print("No configurations found")
        return
    
    plt.figure(figsize=(16, 12))
    
    # Create subplot grid: tessellations x config types
    tess_types = sorted(configs['tess_type'].unique())
    config_types = sorted(configs['config_type'].unique())
    
    n_tess = len(tess_types)
    n_config = len(config_types)
    
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    for i, tess_type in enumerate(tess_types):
        for j, config_type in enumerate(config_types):
            
            subplot_idx = i * n_config + j + 1
            plt.subplot(n_tess, n_config, subplot_idx)
            
            subset = configs[
                (configs['tess_type'] == tess_type) & 
                (configs['config_type'] == config_type)
            ]
            
            if subset.empty:
                plt.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=plt.gca().transAxes)
                plt.title(f'{config_type.upper()}-{tess_type}')
                continue
            
            color_idx = 0
            for weight_type in subset['weight_type'].unique():
                
                weight_subset = subset[subset['weight_type'] == weight_type]
                
                for _, config in weight_subset.iterrows():
                    config_id = config['id']
                    
                    window_radii, variance_matrix, num_windows, resolution = db.get_elv_results(config_id)
                    
                    if window_radii is not None and variance_matrix is not None:
                        
                        # Get edge density
                        try:
                            with sqlite3.connect(db.db_path) as conn:
                                cursor = conn.cursor()
                                cursor.execute("""
                                    SELECT weighted_edge_density, unweighted_edge_density 
                                    FROM edge_length_variance_results
                                    WHERE config_id = ?
                                """, (config_id,))
                                result = cursor.fetchone()
                                
                                if result and result[1] is not None:
                                    edge_density = result[1]  # Use unweighted density
                                else:
                                    cursor.execute("""
                                        SELECT edge_density FROM edge_length_variance_results
                                        WHERE config_id = ?
                                    """, (config_id,))
                                    fallback = cursor.fetchone()
                                    edge_density = fallback[0] if fallback else 1.0
                        except:
                            edge_density = 1.0
                        
                        dimensionless_R = window_radii * edge_density
                        
                        if len(variance_matrix.shape) == 3:
                            variance = np.var(variance_matrix[0], axis=0)
                        else:
                            variance = np.var(variance_matrix, axis=0)
                        
                        plt.loglog(dimensionless_R, variance, 
                                 'o-', color=colors[color_idx % len(colors)], 
                                 alpha=0.7, linewidth=1, markersize=2,
                                 label=weight_type if config_id == weight_subset.iloc[0]['id'] else "")
                
                color_idx += 1
            
            plt.title(f'{config_type.upper()}-{tess_type}')
            plt.grid(True, alpha=0.3)
            
            if subplot_idx > n_tess * n_config - n_config:
                plt.xlabel(r'$R \rho_\ell$')
            if subplot_idx % n_config == 1:
                plt.ylabel('ELV')
    
    plt.tight_layout()
    
    timestamp = _timestamp()
    filename = f"../outputs/plots/hyperuniformity_summary_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plot saved: {filename}")

def create_weight_effect_analysis(db_path="../network_results_weights.db"):
    """Analyze weight effects on hyperuniformity scaling"""
    
    # Ensure output directories exist
    os.makedirs("../outputs/plots", exist_ok=True)
    
    db = NetworkResultsDB(db_path)
    configs = db.get_configurations(dimension=2)
    
    if configs.empty:
        print("No configurations found")
        return
    
    for tess_type in configs['tess_type'].unique():
        for config_type in configs['config_type'].unique():
            
            subset = configs[
                (configs['tess_type'] == tess_type) & 
                (configs['config_type'] == config_type)
            ]
            
            if len(subset) < 2:
                continue
            
            weight_types = subset['weight_type'].unique()
            if 'uniform' not in weight_types:
                continue
            
            plt.figure(figsize=(10, 8))
            
            uniform_data = None
            colors = ['blue', 'red', 'green', 'purple', 'orange']
            color_idx = 0
            
            for weight_type in weight_types:
                
                weight_subset = subset[subset['weight_type'] == weight_type]
                
                for _, config in weight_subset.iterrows():
                    config_id = config['id']
                    
                    window_radii, variance_matrix, num_windows, resolution = db.get_elv_results(config_id)
                    
                    if window_radii is not None and variance_matrix is not None:
                        
                        # Get edge densities
                        try:
                            with sqlite3.connect(db.db_path) as conn:
                                cursor = conn.cursor()
                                cursor.execute("""
                                    SELECT weighted_edge_density, unweighted_edge_density 
                                    FROM edge_length_variance_results
                                    WHERE config_id = ?
                                """, (config_id,))
                                result = cursor.fetchone()
                                
                                if result and result[0] is not None and result[1] is not None:
                                    weighted_density = result[0]
                                    unweighted_density = result[1]
                                else:
                                    cursor.execute("""
                                        SELECT edge_density FROM edge_length_variance_results
                                        WHERE config_id = ?
                                    """, (config_id,))
                                    fallback = cursor.fetchone()
                                    weighted_density = unweighted_density = fallback[0] if fallback else 1.0
                        except:
                            weighted_density = unweighted_density = 1.0
                        
                        dimensionless_R = window_radii * unweighted_density
                        
                        if len(variance_matrix.shape) == 3:
                            unweighted_variance = np.var(variance_matrix[0], axis=0)
                            weighted_variance = np.var(variance_matrix[2], axis=0)
                        else:
                            unweighted_variance = np.var(variance_matrix, axis=0)
                            weighted_variance = unweighted_variance
                        
                        if weight_type == 'uniform':
                            uniform_data = (dimensionless_R, unweighted_variance)
                            plt.loglog(dimensionless_R, unweighted_variance, 
                                     'k-', linewidth=3, label='Uniform (baseline)', alpha=0.8)
                        else:
                            # Show both weighted and unweighted for comparison
                            plt.loglog(dimensionless_R, unweighted_variance, 
                                     '--', color=colors[color_idx % len(colors)], 
                                     label=f'{weight_type} (unweighted)', alpha=0.6)
                            
                            if not np.array_equal(weighted_variance, unweighted_variance):
                                dimensionless_R_w = window_radii * weighted_density
                                plt.loglog(dimensionless_R_w, weighted_variance, 
                                         '-', color=colors[color_idx % len(colors)], 
                                         linewidth=2, label=f'{weight_type} (weighted)')
                        
                        color_idx += 1
            
            plt.xlabel(r'$R \rho_\ell$', fontsize=14)
            plt.ylabel('Edge Length Variance', fontsize=14)
            plt.title(f'Weight Effect Analysis: {config_type.upper()}-{tess_type}', fontsize=16)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            timestamp = _timestamp()
            filename = f"../outputs/plots/weight_effect_analysis_{config_type}_{tess_type}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Weight effect analysis saved: {filename}")

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "../network_results_weights.db"
    
    # Create backup before major plotting operation
    try:
        backup_manager = DatabaseBackupManager(db_path)
        backup_path = backup_manager.create_backup(force=False, compress=True)
        if backup_path:
            print(f"Pre-processing backup created: {backup_path.name}")
    except Exception as e:
        print(f"Backup warning: {e}")
    
    print("Generating group comparison plots...")
    create_weight_comparison_plots(db_path)
    
    print("Generating summary plot...")
    create_summary_plot(db_path)
    
    print("Generating weight effect analysis...")
    create_weight_effect_analysis(db_path)
    
    print("All plots generated successfully!")