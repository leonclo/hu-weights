#!/usr/bin/env python3
"""
Post-processing script to generate group comparison plots from database
Run after cluster job completes to create hyperuniformity analysis visualizations
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from db_utils import NetworkResultsDB
import pandas as pd
from datetime import datetime
import sqlite3


def _timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

def create_weight_comparison_plots(db_path="network_results_weights.db"):
    """Generate comparison plots across different weight schemes"""
    
    db = NetworkResultsDB(db_path)
    configs = db.get_configurations(dimension=2)
    
    if configs.empty:
        print("No configurations found in database")
        return
    
    print(f"Found {len(configs)} configurations in database")
    
    # Group by tessellation and configuration type for comparison
    for tess_type in configs['tess_type'].unique():
        for config_type in configs['config_type'].unique():
            
            # Filter configs for this tessellation/config combination
            subset = configs[
                (configs['tess_type'] == tess_type) & 
                (configs['config_type'] == config_type)
            ]
            
            if len(subset) < 2:  # Need at least 2 for comparison
                continue
                
            # Group by weight type to get multiple realizations
            weight_groups = subset.groupby('weight_type')
            
            if len(weight_groups) < 2:  # Need at least 2 weight types
                continue
                
            # Create comparison plot
            plt.figure(figsize=(12, 8))
            
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
            color_idx = 0
            
            for weight_type, group in weight_groups:
                
                # Get all realizations for this weight type
                config_ids = group['id'].tolist()
                
                # Average across realizations
                all_variances = []
                Rs = None
                edge_densities = []
                
                for config_id in config_ids:
                    window_radii, variance_matrix, num_windows, resolution = db.get_elv_results(config_id)
                    
                    if window_radii is not None and variance_matrix is not None:
                        if Rs is None:
                            Rs = window_radii
                        
                        # Use unweighted variance for comparison (variance_matrix[0])
                        if len(variance_matrix.shape) >= 3:
                            unweighted_var = np.var(variance_matrix[0], axis=0)
                        else:
                            unweighted_var = np.var(variance_matrix, axis=0)
                        
                        all_variances.append(unweighted_var)
                        
                        # Get edge density from database
                        with sqlite3.connect(db.db_path) as conn:
                            cursor = conn.cursor()
                            cursor.execute("""
                                SELECT edge_density FROM edge_length_variance_results
                                WHERE config_id = ?
                            """, (config_id,))
                            result = cursor.fetchone()
                        
                        if result:
                            edge_densities.append(result[0])
                
                if len(all_variances) > 0 and Rs is not None:
                    # Average across realizations
                    mean_variance = np.mean(all_variances, axis=0)
                    std_variance = np.std(all_variances, axis=0)
                    mean_edge_density = np.mean(edge_densities) if edge_densities else 1.0
                    
                    # Plot with proper dimensionless scaling
                    dimensionless_R = Rs * (mean_edge_density ** 0.5)
                    
                    # Main curve
                    plt.loglog(dimensionless_R, mean_variance, 
                              color=colors[color_idx % len(colors)], 
                              label=f'{weight_type} (ρₗ={mean_edge_density:.3f})',
                              linewidth=2, marker='o', markersize=3)
                    
                    # Error bars (if multiple realizations)
                    if len(all_variances) > 1:
                        plt.fill_between(dimensionless_R, 
                                       mean_variance - std_variance,
                                       mean_variance + std_variance,
                                       alpha=0.2, color=colors[color_idx % len(colors)])
                    
                    color_idx += 1
            
            # Formatting
            plt.xlabel(r'$R \rho_\ell^{1/2}$', fontsize=14)
            plt.ylabel('Edge Length Variance', fontsize=14)
            plt.title(f'Hyperuniformity Analysis: {config_type.upper()} {tess_type} Tessellation', fontsize=16)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Add hyperuniformity reference lines
            x_range = plt.xlim()
            x_ref = np.linspace(x_range[0], x_range[1], 100)
            
            # Hyperuniform reference (flat)
            plt.axhline(y=mean_variance[-1], color='black', linestyle='--', alpha=0.5, label='Hyperuniform (R⁰)')
            
            # Non-hyperuniform reference (slope=2)
            y_ref = (x_ref/x_ref[0])**2 * mean_variance[0]
            plt.loglog(x_ref, y_ref, 'k:', alpha=0.5, label='Non-hyperuniform (R²)')
            
            plt.legend()
            
            # Save plot
            filename = f"hyperuniformity_comparison_{config_type}_{tess_type}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Generated comparison plot: {filename}")

def create_summary_plot(db_path="network_results_weights.db"):
    """Create overall summary showing hyperuniformity breaking/creation effects"""
    
    db = NetworkResultsDB(db_path)
    configs = db.get_configurations(dimension=2)
    
    if configs.empty:
        return
    
    # Create figure with subplots for each config type
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    config_types = ['URL', 'poi', 'URL']  # URL twice for a=0.5 and a=0.0
    titles = ['URL', 
              'Poisson', 
              'A=0 URL']
    
    for idx, (ax, config_type, title) in enumerate(zip(axes, config_types, titles)):
        
        # Filter configs
        if idx == 0:  # URL a=0.5
            subset = configs[(configs['config_type'] == 'URL') & (configs['a'] == 0.5)]
        elif idx == 1:  # Poisson
            subset = configs[configs['config_type'] == 'poi']
        else:  # Square lattice a=0.0
            subset = configs[(configs['config_type'] == 'URL') & (configs['a'] == 0.0)]
        
        if len(subset) == 0:
            continue
            
        # Plot different weight types
        for weight_type in subset['weight_type'].unique():
            weight_configs = subset[subset['weight_type'] == weight_type]
            
            all_variances = []
            Rs = None
            
            for _, config in weight_configs.iterrows():
                window_radii, variance_matrix, _, _ = db.get_elv_results(config['id'])
                
                if window_radii is not None:
                    if Rs is None:
                        Rs = window_radii
                    
                    if len(variance_matrix.shape) >= 3:
                        unweighted_var = np.var(variance_matrix[0], axis=0)
                    else:
                        unweighted_var = np.var(variance_matrix, axis=0)
                    
                    all_variances.append(unweighted_var)
            
            if len(all_variances) > 0:
                mean_variance = np.mean(all_variances, axis=0)
                ax.loglog(Rs, mean_variance, label=weight_type, linewidth=2, marker='o', markersize=3)
        
        ax.set_xlabel(r'Window Radius R', fontsize=12)
        ax.set_ylabel('Edge Length Variance', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"hyperuniformity_summary_{_timestamp()}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated summary plot: hyperuniformity_summary.png")

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "network_results_weights.db"
    
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        sys.exit(1)
    
    print(f"Generating group plots from database: {db_path}")
    
    create_weight_comparison_plots(db_path)
    create_summary_plot(db_path)
    
    print("Group plotting complete!")