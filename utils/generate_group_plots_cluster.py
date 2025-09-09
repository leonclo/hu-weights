#!/usr/bin/env python3
"""
Comprehensive group comparison plots from ELV analysis database
Generates multiple visualization types for cluster analysis comparison
- Faceted multi-panel plots
- Overlay comparison plots  
- Heatmaps and contour plots
- Clustering dendrograms
- Small multiple line plots
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sqlite3
import pickle
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

# Optional imports - fallback if not available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available, using basic matplotlib styling")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not available, clustering features disabled")

# Import local utilities
from db_utils import NetworkResultsDB
from backup_database import DatabaseBackupManager

# Set style for professional plots
if HAS_SEABORN:
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
else:
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def _timestamp():
    """Generate timestamp for filenames"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

class ClusterGroupPlotter:
    """Main class for generating comprehensive cluster comparison plots"""
    
    def __init__(self, db_path="../network_results_weights.db", output_dir="../outputs/plots", 
                 min_date_cutoff_days=7, N_filter=None):
        """
        Initialize plotter with database connection and filtering parameters
        
        Args:
            db_path: Path to SQLite database
            output_dir: Directory for plot output
            min_date_cutoff_days: Only include data from last N days (ensures recent/accurate data)
            N_filter: List of system sizes to include, e.g. [5, 10, 200] or None for all sizes
        """
        self.db_path = db_path
        self.output_dir = output_dir
        self.min_date_cutoff = datetime.now() - timedelta(days=min_date_cutoff_days)
        self.N_filter = N_filter
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize database connection
        self.db = NetworkResultsDB(db_path)
        
        # Load and filter data
        self.configs, self.elv_data = self._load_filtered_data()
        
        print(f"Loaded {len(self.configs)} configurations with ELV data from {self.min_date_cutoff}")
        
    def _load_filtered_data(self):
        """Load configurations and ELV data, filtering for recent entries only"""
        
        # Get recent configurations
        configs = self.db.get_configurations(dimension=2)
        if configs.empty:
            print("No configurations found in database")
            return pd.DataFrame(), {}
            
        # Filter by date - only recent data for accuracy
        configs['created_at'] = pd.to_datetime(configs['created_at'])
        configs = configs[configs['created_at'] >= self.min_date_cutoff]
        
        if configs.empty:
            print(f"No recent configurations found (since {self.min_date_cutoff})")
            return pd.DataFrame(), {}
            
        # Filter by system sizes if specified
        if self.N_filter is not None:
            original_count = len(configs)
            configs = configs[configs['N'].isin(self.N_filter)]
            filtered_count = len(configs)
            print(f"N-filter applied: {original_count} -> {filtered_count} configs (N in {self.N_filter})")
            
            if configs.empty:
                print(f"No configurations found with N in {self.N_filter}")
                return pd.DataFrame(), {}
            
        print(f"Found {len(configs)} recent configurations")
        
        # Load ELV results for each configuration
        elv_data = {}
        for config_id in configs['id']:
            try:
                elv_results = self.db.get_elv_results(config_id)
                # get_elv_results returns (window_radii, variance_matrix, num_windows, resolution)
                if elv_results and elv_results[0] is not None:  # Check if window_radii is not None
                    elv_data[config_id] = elv_results
            except Exception as e:
                print(f"Error loading ELV data for config {config_id}: {e}")
                continue
                
        print(f"Loaded ELV data for {len(elv_data)} configurations")
        
        # Only keep configs that have ELV data
        configs = configs[configs['id'].isin(elv_data.keys())]
        
        return configs, elv_data
    
    def _get_data_label(self, row):
        """Generate compact, data-accurate label for configuration"""
        weight_params = ""
        if row['weight_parameters'] and row['weight_parameters'] != 'None':
            try:
                params = eval(row['weight_parameters']) if isinstance(row['weight_parameters'], str) else row['weight_parameters']
                if isinstance(params, dict) and params:
                    weight_params = f"_p{list(params.values())[0]:.1f}" if params else ""
            except:
                pass
                
        return f"{row['config_type']}-{row['tess_type']}_N{row['N']}_{row['weight_type']}{weight_params}"
    
    def _get_density(self, config):
        """Get density value for a configuration, with fallback"""
        density = config.get('beamw', None)
        if density is None or density == 0:
            # Use normalized density = 1 for plotting
            density = 1.0
        return density
    
    def _get_elv_curve(self, config_id, use_weighted=False):
        """Extract ELV curve data for a configuration"""
        if config_id not in self.elv_data:
            return None, None
            
        elv_result = self.elv_data[config_id]
        
        try:
            # elv_result is tuple: (window_radii, variance_matrix, num_windows, resolution)
            window_radii = elv_result[0]  # Already unpickled by get_elv_results
            variance_matrix = elv_result[1]  # Already unpickled by get_elv_results
            
            # Extract ELV curve 
            # variance_matrix from Network_ELV2D.py has shape [3, Nwind, reso]
            # - tovar[0,i,j] = total edge length in window i at radius j
            # - tovar[1,i,j] = percent inside 
            # - tovar[2,i,j] = weighted edge contributions
            
            if variance_matrix.ndim == 3:
                if use_weighted and variance_matrix.shape[0] >= 3:
                    # Use pre-weighted contributions (slice 2)
                    elv_curve = np.var(variance_matrix[2, :, :], axis=0)  # Variance over windows
                else:
                    # Use unweighted edge lengths (slice 0)
                    elv_curve = np.var(variance_matrix[0, :, :], axis=0)  # Variance over windows
            elif variance_matrix.ndim == 2:
                # Legacy format: take the last row
                elv_curve = variance_matrix[-1, :] if variance_matrix.shape[0] > 1 else variance_matrix[0, :]
            else:
                # Fallback
                elv_curve = variance_matrix.flatten()
                
            return window_radii, elv_curve
            
        except Exception as e:
            print(f"Error extracting ELV curve for config {config_id}: {e}")
            print(f"  ELV result type: {type(elv_result)}")
            print(f"  ELV result length: {len(elv_result) if hasattr(elv_result, '__len__') else 'No length'}")
            if hasattr(elv_result, '__len__') and len(elv_result) > 1:
                print(f"  Window radii type: {type(elv_result[0])}, shape: {elv_result[0].shape if hasattr(elv_result[0], 'shape') else 'No shape'}")
                print(f"  Variance matrix type: {type(elv_result[1])}, shape: {elv_result[1].shape if hasattr(elv_result[1], 'shape') else 'No shape'}")
            return None, None
    
    def generate_faceted_multipanel(self):
        """Generate faceted multi-panel plots grouped by system or tessellation"""
        
        print("Generating faceted multi-panel plots...")
        
        # Group by system type and tessellation
        system_tess_groups = self.configs.groupby(['config_type', 'tess_type'])
        
        unique_systems = self.configs['config_type'].unique()
        unique_tess = self.configs['tess_type'].unique()
        
        # Create comprehensive multi-panel plot
        n_systems = len(unique_systems)
        n_tess = len(unique_tess)
        
        fig, axes = plt.subplots(n_tess, n_systems, figsize=(4*n_systems, 3*n_tess))
        if n_systems == 1 and n_tess == 1:
            axes = [[axes]]
        elif n_systems == 1:
            axes = [[ax] for ax in axes]
        elif n_tess == 1:
            axes = [axes]
            
        fig.suptitle('ELV Comparison Across Systems and Tessellations', fontsize=16, y=0.98)
        
        for i, tess in enumerate(unique_tess):
            for j, system in enumerate(unique_systems):
                ax = axes[i][j]
                
                # Get configs for this system-tessellation combination
                subset = self.configs[
                    (self.configs['config_type'] == system) & 
                    (self.configs['tess_type'] == tess)
                ]
                
                if subset.empty:
                    ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=12)
                    ax.set_title(f'{system}-{tess}')
                    continue
                    
                # Plot each weight type with different color
                if HAS_SEABORN:
                    colors = sns.color_palette("husl", len(subset))
                else:
                    colors = plt.cm.tab10(np.linspace(0, 1, len(subset)))
                
                for idx, (_, config) in enumerate(subset.iterrows()):
                    use_weighted = config['weight_type'] != 'uniform'
                radii, elv = self._get_elv_curve(config['id'], use_weighted=use_weighted)
                    if radii is not None and elv is not None:
                        # Calculate density for x-axis
                        density = self._get_density(config)
                        r_rho = radii * density
                        
                        label = f"{config['weight_type']}_N{config['N']}"
                        ax.loglog(r_rho, elv, color=colors[idx], label=label, alpha=0.8)
                
                ax.set_title(f'{system}-{tess}')
                ax.set_xlabel('Rρₗ')
                ax.set_ylabel('ELV')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
        
        plt.tight_layout()
        filename = f"faceted_multipanel_comparison_{_timestamp()}.png"
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")
        
    def generate_overlay_comparison(self):
        """Generate overlay plots comparing weighting schemes within single context"""
        
        print("Generating overlay comparison plots...")
        
        # Group by system-tessellation combinations
        system_tess_groups = self.configs.groupby(['config_type', 'tess_type'])
        
        for (system, tess), group in system_tess_groups:
            if len(group) < 2:  # Need at least 2 configs to compare
                continue
                
            plt.figure(figsize=(10, 6))
            
            # Plot each configuration in the group
            if HAS_SEABORN:
                colors = sns.color_palette("Set1", len(group))
            else:
                colors = plt.cm.Set1(np.linspace(0, 1, len(group)))
            
            for idx, (_, config) in enumerate(group.iterrows()):
                use_weighted = config['weight_type'] != 'uniform'
                radii, elv = self._get_elv_curve(config['id'], use_weighted=use_weighted)
                if radii is not None and elv is not None:
                    density = self._get_density(config)
                    r_rho = radii * density
                    
                    label = self._get_data_label(config)
                    plt.loglog(r_rho, elv, color=colors[idx], label=label, 
                             linewidth=2, alpha=0.8)
            
            plt.xlabel('Rρₗ')
            plt.ylabel('Edge Length Variance')
            plt.title(f'ELV Comparison: {system}-{tess}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            filename = f"overlay_comparison_{system}_{tess}_{_timestamp()}.png"
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved: {filename}")
    
    def generate_variance_heatmaps(self):
        """Generate heatmaps summarizing variance magnitudes"""
        
        print("Generating variance heatmaps...")
        
        # Create matrix of maximum variance values for heatmap
        systems = self.configs['config_type'].unique()
        tess_types = self.configs['tess_type'].unique()
        weight_types = self.configs['weight_type'].unique()
        
        # Heatmap 1: Max variance by system-tessellation
        max_var_matrix = np.full((len(systems), len(tess_types)), np.nan)
        
        for i, system in enumerate(systems):
            for j, tess in enumerate(tess_types):
                subset = self.configs[
                    (self.configs['config_type'] == system) & 
                    (self.configs['tess_type'] == tess)
                ]
                
                max_variances = []
                for _, config in subset.iterrows():
                    use_weighted = config['weight_type'] != 'uniform'
                radii, elv = self._get_elv_curve(config['id'], use_weighted=use_weighted)
                    if radii is not None and elv is not None:
                        max_variances.append(np.max(elv))
                
                if max_variances:
                    max_var_matrix[i, j] = np.mean(max_variances)
        
        plt.figure(figsize=(8, 6))
        
        if HAS_SEABORN:
            sns.heatmap(max_var_matrix, 
                       xticklabels=tess_types, 
                       yticklabels=systems,
                       annot=True, fmt='.2e', 
                       cmap='viridis', cbar_kws={'label': 'Max ELV'})
        else:
            # Fallback matplotlib heatmap
            im = plt.imshow(max_var_matrix, cmap='viridis', aspect='auto')
            plt.colorbar(im, label='Max ELV')
            plt.xticks(range(len(tess_types)), tess_types)
            plt.yticks(range(len(systems)), systems)
            
            # Add text annotations
            for i in range(len(systems)):
                for j in range(len(tess_types)):
                    if not np.isnan(max_var_matrix[i, j]):
                        plt.text(j, i, f'{max_var_matrix[i, j]:.2e}', 
                               ha='center', va='center', color='white', fontsize=8)
        
        plt.title('Maximum Edge Length Variance by System-Tessellation')
        plt.ylabel('System Type')
        plt.xlabel('Tessellation Type')
        
        filename = f"variance_heatmap_system_tess_{_timestamp()}.png"
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")
        
        # Heatmap 2: Variance ratios by weight type
        if len(weight_types) > 1:
            weight_var_matrix = np.full((len(systems), len(weight_types)), np.nan)
            
            for i, system in enumerate(systems):
                for j, weight in enumerate(weight_types):
                    subset = self.configs[
                        (self.configs['config_type'] == system) & 
                        (self.configs['weight_type'] == weight)
                    ]
                    
                    max_variances = []
                    for _, config in subset.iterrows():
                        use_weighted = config['weight_type'] != 'uniform'
                radii, elv = self._get_elv_curve(config['id'], use_weighted=use_weighted)
                        if radii is not None and elv is not None:
                            max_variances.append(np.max(elv))
                    
                    if max_variances:
                        weight_var_matrix[i, j] = np.mean(max_variances)
            
            plt.figure(figsize=(8, 6))
            
            if HAS_SEABORN:
                sns.heatmap(weight_var_matrix,
                           xticklabels=weight_types,
                           yticklabels=systems,
                           annot=True, fmt='.2e',
                           cmap='plasma', cbar_kws={'label': 'Max ELV'})
            else:
                # Fallback matplotlib heatmap
                im = plt.imshow(weight_var_matrix, cmap='plasma', aspect='auto')
                plt.colorbar(im, label='Max ELV')
                plt.xticks(range(len(weight_types)), weight_types)
                plt.yticks(range(len(systems)), systems)
                
                # Add text annotations
                for i in range(len(systems)):
                    for j in range(len(weight_types)):
                        if not np.isnan(weight_var_matrix[i, j]):
                            plt.text(j, i, f'{weight_var_matrix[i, j]:.2e}', 
                                   ha='center', va='center', color='white', fontsize=8)
            
            plt.title('Maximum Edge Length Variance by System-Weight Type')
            plt.ylabel('System Type')
            plt.xlabel('Weight Type')
            
            filename = f"variance_heatmap_weight_effect_{_timestamp()}.png"
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved: {filename}")
    
    def generate_clustering_dendrogram(self):
        """Generate clustering dendrograms for grouping similar variance curves"""
        
        print("Generating clustering dendrograms...")
        
        if not HAS_SKLEARN:
            print("scikit-learn not available, skipping clustering analysis")
            return
            
        if len(self.configs) < 3:
            print("Need at least 3 configurations for clustering")
            return
            
        # Extract variance curves for clustering
        curve_matrix = []
        labels = []
        
        for _, config in self.configs.iterrows():
            use_weighted = config['weight_type'] != 'uniform'
            radii, elv = self._get_elv_curve(config['id'], use_weighted=use_weighted)
            if radii is not None and elv is not None:
                # Interpolate to common grid for comparison
                log_radii = np.log10(radii)
                log_elv = np.log10(elv)
                
                # Common grid
                common_log_radii = np.linspace(log_radii.min(), log_radii.max(), 50)
                common_log_elv = np.interp(common_log_radii, log_radii, log_elv)
                
                curve_matrix.append(common_log_elv)
                labels.append(self._get_data_label(config))
        
        if len(curve_matrix) < 3:
            print("Not enough valid curves for clustering")
            return
            
        curve_matrix = np.array(curve_matrix)
        
        # Standardize the curves
        scaler = StandardScaler()
        curve_matrix_scaled = scaler.fit_transform(curve_matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(curve_matrix_scaled, method='ward')
        
        # Plot dendrogram
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix, labels=labels, orientation='left', leaf_font_size=10)
        plt.title('Hierarchical Clustering of ELV Curves')
        plt.xlabel('Distance')
        
        filename = f"clustering_dendrogram_{_timestamp()}.png"
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")
    
    def generate_small_multiples(self):
        """Generate small multiple line plots for compact side-by-side comparison"""
        
        print("Generating small multiple line plots...")
        
        # Calculate grid dimensions
        n_configs = len(self.configs)
        if n_configs == 0:
            return
            
        ncols = min(4, n_configs)  # Max 4 columns
        nrows = (n_configs + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
        if nrows == 1 and ncols == 1:
            axes = [axes]
        elif nrows == 1:
            axes = [axes]
        elif ncols == 1:
            axes = [[ax] for ax in axes]
        
        fig.suptitle('Small Multiples: Individual ELV Curves', fontsize=16)
        
        for idx, (_, config) in enumerate(self.configs.iterrows()):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row][col] if nrows > 1 else axes[col]
            
            use_weighted = config['weight_type'] != 'uniform'
            radii, elv = self._get_elv_curve(config['id'], use_weighted=use_weighted)
            if radii is not None and elv is not None:
                density = self._get_density(config)
                r_rho = radii * density
                
                ax.loglog(r_rho, elv, 'b-', linewidth=2, alpha=0.8)
            
            label = self._get_data_label(config)
            ax.set_title(label, fontsize=8)
            ax.grid(True, alpha=0.3)
            
            if row == nrows - 1:  # Bottom row
                ax.set_xlabel('Rρₗ', fontsize=8)
            if col == 0:  # Left column
                ax.set_ylabel('ELV', fontsize=8)
        
        # Hide empty subplots
        for idx in range(n_configs, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row][col] if nrows > 1 else axes[col]
            ax.axis('off')
        
        plt.tight_layout()
        filename = f"small_multiples_{_timestamp()}.png"
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")
    
    def generate_all_plots(self):
        """Generate all visualization types"""
        
        print("\n=== GENERATING COMPREHENSIVE CLUSTER PLOTS ===")
        print(f"Data cutoff date: {self.min_date_cutoff}")
        print(f"Configurations: {len(self.configs)}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 50)
        
        # Create backup before major processing
        try:
            backup_manager = DatabaseBackupManager(self.db_path)
            backup_path = backup_manager.create_backup(force=False, compress=True)
            if backup_path:
                print(f"Pre-processing backup: {backup_path.name}")
        except Exception as e:
            print(f"Backup warning: {e}")
        
        if self.configs.empty:
            print("No data to plot")
            return
            
        # Generate all plot types
        self.generate_faceted_multipanel()
        self.generate_overlay_comparison() 
        self.generate_variance_heatmaps()
        self.generate_clustering_dendrogram()
        self.generate_small_multiples()
        
        print("\n=== PLOT GENERATION COMPLETE ===")

def main():
    """
    Main execution function
    
    Usage:
        python generate_group_plots_cluster.py [cutoff_days] [N_filter]
        
    Examples:
        python generate_group_plots_cluster.py                    # Use last 7 days, all N
        python generate_group_plots_cluster.py 14                 # Use last 14 days, all N
        python generate_group_plots_cluster.py 7 "5,10,200"       # Use last 7 days, only N=5,10,200
        python generate_group_plots_cluster.py 30 "200"           # Use last 30 days, only N=200
    """
    
    # Show usage if help requested
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print(main.__doc__)
        return
    
    # Parse command line arguments
    db_path = "../network_results_weights.db"
    output_dir = "../outputs/plots"
    cutoff_days = 7  # Only use data from last 7 days
    N_filter = None  # Include all system sizes by default
    
    if len(sys.argv) > 1:
        cutoff_days = int(sys.argv[1])
    if len(sys.argv) > 2:
        # Parse N filter as comma-separated list: "5,10,200"
        N_filter = [int(n.strip()) for n in sys.argv[2].split(',')]
        print(f"System size filter: {N_filter}")
    
    # Create plotter and generate all visualizations
    plotter = ClusterGroupPlotter(
        db_path=db_path,
        output_dir=output_dir, 
        min_date_cutoff_days=cutoff_days,
        N_filter=N_filter
    )
    
    plotter.generate_all_plots()

if __name__ == "__main__":
    main()