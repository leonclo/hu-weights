import sqlite3
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
import os
from typing import List, Dict, Optional, Tuple

class NetworkResultsDB:
    def __init__(self, db_path="network_results.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with schema"""
        schema_path = os.path.join(os.path.dirname(__file__), 'database_schema.sql')
        with open(schema_path, 'r') as f:
            schema = f.read()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(schema)
    
    def save_configuration(self, N: int, a: Optional[float], simname: str, 
                          config_type: str, tess_type: str, dimension: int,
                          weight_type: str = 'uniform', weight_parameters: Dict = None,
                          beamw: Optional[float] = None, binset: Optional[float] = None,
                          filename: Optional[str] = None, point_pattern: Optional[np.ndarray] = None) -> int:
        """Save configuration and return config_id"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Serialize point pattern and weight parameters
            point_blob = pickle.dumps(point_pattern) if point_pattern is not None else None
            weight_params_json = json.dumps(weight_parameters) if weight_parameters else None
            
            cursor.execute("""
                INSERT OR REPLACE INTO configurations 
                (N, a, beamw, simname, config_type, tess_type, dimension, binset, 
                 weight_type, weight_parameters, filename, point_pattern)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (N, a, beamw, simname, config_type, tess_type, dimension, binset,
                  weight_type, weight_params_json, filename, point_blob))
            
            return cursor.lastrowid
    
    def save_elv_results(self, config_id: int, window_radii: np.ndarray, 
                        variance_matrix: np.ndarray, edge_density: float,
                        num_windows: int, resolution: int) -> int:
        """Save ELV results with edge weight support"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Serialize numpy arrays
            radii_blob = pickle.dumps(window_radii)
            variance_blob = pickle.dumps(variance_matrix)
            
            cursor.execute("""
                INSERT INTO edge_length_variance_results 
                (config_id, window_radii, variance_matrix, edge_density, num_windows, resolution)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (config_id, radii_blob, variance_blob, edge_density, num_windows, resolution))
            
            return cursor.lastrowid
    
    def save_computation_metadata(self, config_id: int, computation_type: str,
                                computation_time: float, memory_usage: float = None,
                                parameters: Dict = None):
        """Save computation metadata"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            params_json = json.dumps(parameters) if parameters else None
            
            cursor.execute("""
                INSERT INTO computation_metadata 
                (config_id, computation_type, computation_time, memory_usage, parameters)
                VALUES (?, ?, ?, ?, ?)
            """, (config_id, computation_type, computation_time, memory_usage, params_json))
    
    def get_configurations(self, dimension: Optional[int] = None, 
                          config_type: Optional[str] = None,
                          tess_type: Optional[str] = None) -> pd.DataFrame:
        """Get configurations with optional filtering"""
        query = "SELECT * FROM configurations WHERE 1=1"
        params = []
        
        if dimension:
            query += " AND dimension = ?"
            params.append(dimension)
        if config_type:
            query += " AND config_type = ?"
            params.append(config_type)
        if tess_type:
            query += " AND tess_type = ?"
            params.append(tess_type)
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_spectral_density_results(self, config_id: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """Get spectral density results for a configuration"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT wavenumber, spectral_density, phi2 
                FROM spectral_density_results 
                WHERE config_id = ?
            """, (config_id,))
            result = cursor.fetchone()
            
            if result:
                wavenumber = pickle.loads(result[0])
                spectral_density = pickle.loads(result[1])
                phi2 = result[2]
                return wavenumber, spectral_density, phi2
            return None, None, None
    
    def get_elv_results(self, config_id: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Get edge length variance results for a configuration"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT window_radii, variance_matrix, num_windows, resolution
                FROM edge_length_variance_results 
                WHERE config_id = ?
            """, (config_id,))
            result = cursor.fetchone()
            
            if result:
                window_radii = pickle.loads(result[0])
                variance_matrix = pickle.loads(result[1])
                num_windows = result[2]
                resolution = result[3]
                return window_radii, variance_matrix, num_windows, resolution
            return None, None, None, None
    
    def get_computation_metadata(self, config_id: int) -> Dict:
        """Get computation metadata for a configuration"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT computation_type, computation_time, memory_usage, parameters
                FROM computation_metadata 
                WHERE config_id = ?
            """, (config_id,))
            result = cursor.fetchone()
            
            if result:
                return {
                    'computation_type': result[0],
                    'computation_time': result[1],
                    'memory_usage': result[2],
                    'parameters': eval(result[3]) if result[3] else {}
                }
            return {}
    
    def compare_spectral_densities(self, config_ids: List[int], 
                                 labels: Optional[List[str]] = None) -> plt.Figure:
        """Compare spectral densities from multiple configurations"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if labels is None:
            labels = [f"Config {cid}" for cid in config_ids]
        
        for i, config_id in enumerate(config_ids):
            k, S, phi2 = self.get_spectral_density_results(config_id)
            if k is not None:
                ax.loglog(k, S, label=f"{labels[i]} (φ₂={phi2:.3f})")
        
        ax.set_xlabel('Wavenumber k')
        ax.set_ylabel('Spectral Density S(k)')
        ax.set_title('Spectral Density Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def compare_elv_variances(self, config_ids: List[int], 
                            labels: Optional[List[str]] = None) -> plt.Figure:
        """Compare edge length variances from multiple configurations"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if labels is None:
            labels = [f"Config {cid}" for cid in config_ids]
        
        for i, config_id in enumerate(config_ids):
            Rs, variance_matrix, _, _ = self.get_elv_results(config_id)
            if Rs is not None:
                # Calculate variance across windows for each radius
                variances = np.var(variance_matrix, axis=0)
                ax.loglog(Rs, variances, label=labels[i])
        
        ax.set_xlabel('Window Radius R')
        ax.set_ylabel('Edge Length Variance')
        ax.set_title('Edge Length Variance Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def compare_elv_var_rpl(self, config_ids: List[int], 
                           labels: Optional[List[str]] = None) -> plt.Figure:
        """Compare edge length variances vs R*ρₗ^((d-1)/d) from multiple configurations"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if labels is None:
            labels = [f"Config {cid}" for cid in config_ids]
        
        for i, config_id in enumerate(config_ids):
            Rs, variance_matrix, _, _ = self.get_elv_results(config_id)
            if Rs is not None:
                # Get configuration info to determine dimension
                config_df = self.get_configurations()
                config = config_df[config_df['id'] == config_id].iloc[0]
                d = config['dimension']  # spatial dimension
                N = config['N']
                
                # Calculate total edge length per unit volume (ρₗ)
                # For tessellations, estimate edge length density from the network structure
                # Unit volume is N×N for 2D or N×N×N for 3D box
                if d == 2:
                    unit_volume = N * N
                    # Estimate total edge length from variance matrix statistics
                    # This is an approximation - in practice you'd calculate actual total edge length
                    total_edge_length = np.sum(variance_matrix) * len(Rs) / len(variance_matrix)
                elif d == 3:
                    unit_volume = N * N * N
                    total_edge_length = np.sum(variance_matrix) * len(Rs) / len(variance_matrix)
                else:
                    unit_volume = N**d
                    total_edge_length = np.sum(variance_matrix) * len(Rs) / len(variance_matrix)
                
                # Calculate edge length per unit volume
                rho_l = total_edge_length / unit_volume
                
                # Calculate R * ρₗ^((d-1)/d) as per the paper
                x_axis = Rs * (rho_l ** ((d-1)/d))
                
                # Calculate variance across windows for each radius
                variances = np.var(variance_matrix, axis=0)
                
                ax.loglog(x_axis, variances, label=f"{labels[i]} (d={d})")
        
        ax.set_xlabel(r'$R \rho_\ell^{(d-1)/d}$')
        ax.set_ylabel('Edge Length Variance')
        ax.set_title('Edge Length Variance vs R×ρₗ^((d-1)/d)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def compare_elv_var_rpl_by_tessellation(self, dimension: Optional[int] = None, 
                                           config_type: Optional[str] = None,
                                           tessellation_types: Optional[List[str]] = None) -> plt.Figure:
        """Compare edge length variance vs R×ρₗ^((d-1)/d) grouped by tessellation type
        
        Args:
            dimension: Spatial dimension to filter (2 or 3)
            config_type: Configuration type to filter ('poi', 'URL', etc.)
            tessellation_types: List of tessellation types to include ['V', 'D', 'C', 'G']
        """
        if tessellation_types is None:
            tessellation_types = ['V', 'D', 'C', 'G']
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get all configurations matching the criteria
        configs = self.get_configurations(dimension=dimension, config_type=config_type)
        
        # Group by tessellation type
        tessellation_data = {tess_type: [] for tess_type in tessellation_types}
        
        for _, config in configs.iterrows():
            if config['tess_type'] in tessellation_types:
                config_id = config['id']
                Rs, variance_matrix, _, _ = self.get_elv_results(config_id)
                
                if Rs is not None:
                    d = config['dimension']
                    N = config['N']
                    
                    # Calculate total edge length per unit volume (ρₗ)
                    if d == 2:
                        unit_volume = N * N
                    elif d == 3:
                        unit_volume = N * N * N
                    else:
                        unit_volume = N**d
                    
                    # Estimate total edge length from variance matrix statistics
                    total_edge_length = np.sum(variance_matrix) * len(Rs) / len(variance_matrix)
                    rho_l = (total_edge_length / unit_volume)**-(1/2)
                    
                    # Calculate R * ρₗ^((d-1)/d)
                    x_axis = Rs * (rho_l ** ((d-1)/d))
                    variances = np.var(variance_matrix, axis=0)
                    
                    tessellation_data[config['tess_type']].append({
                        'x_axis': x_axis,
                        'variances': variances,
                        'config_id': config_id,
                        'N': N,
                        'config_type': config['config_type']
                    })
        
        # Plot ensemble statistics for each tessellation type
        colors = {'V': 'blue', 'D': 'red', 'C': 'green', 'G': 'orange'}
        
        for tess_type in tessellation_types:
            if tessellation_data[tess_type]:
                # Collect all data points for this tessellation type
                all_x = []
                all_y = []
                
                for data in tessellation_data[tess_type]:
                    all_x.extend(data['x_axis'])
                    all_y.extend(data['variances'])
                
                if all_x:
                    # Convert to numpy arrays for statistical analysis
                    x_array = np.array(all_x)
                    y_array = np.array(all_y)
                    
                    # Sort by x values for proper binning
                    sort_idx = np.argsort(x_array)
                    x_sorted = x_array[sort_idx]
                    y_sorted = y_array[sort_idx]
                    
                    # Create logarithmic bins for ensemble averaging
                    x_min, x_max = x_sorted[0], x_sorted[-1]
                    n_bins = 50
                    log_bins = np.logspace(np.log10(x_min), np.log10(x_max), n_bins)
                    
                    # Calculate ensemble mean and standard deviation in each bin
                    bin_centers = []
                    bin_means = []
                    bin_stds = []
                    
                    for i in range(len(log_bins)-1):
                        mask = (x_sorted >= log_bins[i]) & (x_sorted < log_bins[i+1])
                        if np.sum(mask) > 0:
                            bin_centers.append(np.sqrt(log_bins[i] * log_bins[i+1]))  # geometric mean
                            bin_means.append(np.mean(y_sorted[mask]))
                            bin_stds.append(np.std(y_sorted[mask]))
                    
                    if bin_centers:
                        bin_centers = np.array(bin_centers)
                        bin_means = np.array(bin_means)
                        bin_stds = np.array(bin_stds)
                        
                        # Plot ensemble mean
                        color = colors.get(tess_type, 'black')
                        label = f'{tess_type} (n={len(tessellation_data[tess_type])})'
                        ax.loglog(bin_centers, bin_means, 'o-', color=color, label=label, 
                                linewidth=2, markersize=4)
                        
                        # Add error bars (standard deviation)
                        ax.fill_between(bin_centers, 
                                      np.maximum(bin_means - bin_stds, bin_means/100), 
                                      bin_means + bin_stds, 
                                      alpha=0.2, color=color)
        
        # Formatting
        ax.set_xlabel(r'$R \rho_\ell^{(d-1)/d}$', fontsize=14)
        ax.set_ylabel('Edge Length Variance', fontsize=14)
        
        title_parts = []
        if config_type:
            title_parts.append(f'{config_type.upper()}')
        if dimension:
            title_parts.append(f'{dimension}D')
        title_parts.append('Edge Length Variance by Tessellation Type')
        
        ax.set_title(' '.join(title_parts), fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add text box with ensemble statistics
        info_text = []
        for tess_type in tessellation_types:
            n_configs = len(tessellation_data[tess_type])
            if n_configs > 0:
                info_text.append(f'{tess_type}: {n_configs} configs')
        
        if info_text:
            ax.text(0.02, 0.98, '\n'.join(info_text), transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig

    def get_performance_summary(self) -> pd.DataFrame:
        """Get performance summary across all computations"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT 
                    c.N, c.dimension, c.config_type, c.tess_type,
                    m.computation_type, m.computation_time, m.memory_usage,
                    c.created_at
                FROM configurations c
                JOIN computation_metadata m ON c.id = m.config_id
                ORDER BY c.created_at DESC
            """
            return pd.read_sql_query(query, conn)
    
    def export_results(self, config_id: int, output_dir: str = "."):
        """Export results for a configuration to files"""
        import os
        import json
        
        # Get configuration info
        config_df = self.get_configurations()
        config = config_df[config_df['id'] == config_id].iloc[0]
        
        base_name = f"config_{config_id}_N{config['N']}_{config['config_type']}_{config['tess_type']}"
        
        # Export spectral density if available
        k, S, phi2 = self.get_spectral_density_results(config_id)
        if k is not None:
            spectral_data = np.column_stack([k, S])
            np.savetxt(os.path.join(output_dir, f"{base_name}_spectral.txt"), 
                      spectral_data, header=f"Wavenumber Spectral_Density (phi2={phi2})")
        
        # Export ELV results if available
        Rs, variance_matrix, num_windows, resolution = self.get_elv_results(config_id)
        if Rs is not None:
            np.save(os.path.join(output_dir, f"{base_name}_Rs.npy"), Rs)
            np.save(os.path.join(output_dir, f"{base_name}_variance_matrix.npy"), variance_matrix)
            
            # Calculate and save variance
            variances = np.var(variance_matrix, axis=0)
            elv_data = np.column_stack([Rs, variances])
            np.savetxt(os.path.join(output_dir, f"{base_name}_elv_variance.txt"), 
                      elv_data, header="Window_Radius Edge_Length_Variance")
        
        # Export metadata
        metadata = self.get_computation_metadata(config_id)
        metadata['config'] = config.to_dict()
        
        with open(os.path.join(output_dir, f"{base_name}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Results exported to {output_dir} with base name: {base_name}")
    
    def export_results_ensemble(self, config_ids: List[int], output_dir: str = ".", 
                               ensemble_name: str = "ensemble"):
        """Export results for an ensemble of configurations to text files"""
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export spectral density ensemble
        spectral_data_list = []
        elv_data_list = []
        metadata_list = []
        
        for config_id in config_ids:
            # Get configuration info
            config_df = self.get_configurations()
            config = config_df[config_df['id'] == config_id].iloc[0]
            
            # Export spectral density if available
            k, S, phi2 = self.get_spectral_density_results(config_id)
            if k is not None:
                spectral_data_list.append({
                    'config_id': config_id,
                    'k': k,
                    'S': S,
                    'phi2': phi2,
                    'config': config.to_dict()
                })
            
            # Export ELV results if available
            Rs, variance_matrix, num_windows, resolution = self.get_elv_results(config_id)
            if Rs is not None:
                variances = np.var(variance_matrix, axis=0)
                elv_data_list.append({
                    'config_id': config_id,
                    'Rs': Rs,
                    'variances': variances,
                    'variance_matrix': variance_matrix,
                    'config': config.to_dict()
                })
            
            # Collect metadata
            metadata = self.get_computation_metadata(config_id)
            metadata['config'] = config.to_dict()
            metadata_list.append(metadata)
        
        # Export ensemble spectral density data
        if spectral_data_list:
            with open(os.path.join(output_dir, f"{ensemble_name}_spectral_ensemble.txt"), 'w') as f:
                f.write("# Ensemble spectral density data\n")
                f.write("# Config_ID Wavenumber Spectral_Density Phi2\n")
                for data in spectral_data_list:
                    for i, (k_val, S_val) in enumerate(zip(data['k'], data['S'])):
                        f.write(f"{data['config_id']} {k_val} {S_val} {data['phi2']}\n")
        
        # Export ensemble ELV data
        if elv_data_list:
            with open(os.path.join(output_dir, f"{ensemble_name}_elv_ensemble.txt"), 'w') as f:
                f.write("# Ensemble edge length variance data\n")
                f.write("# Config_ID Window_Radius Edge_Length_Variance\n")
                for data in elv_data_list:
                    for R_val, var_val in zip(data['Rs'], data['variances']):
                        f.write(f"{data['config_id']} {R_val} {var_val}\n")
        
        # Export metadata
        with open(os.path.join(output_dir, f"{ensemble_name}_metadata.json"), 'w') as f:
            json.dump(metadata_list, f, indent=2, default=str)
        
        print(f"Ensemble results exported to {output_dir} with base name: {ensemble_name}")
        print(f"Exported {len(spectral_data_list)} spectral density configurations")
        print(f"Exported {len(elv_data_list)} ELV configurations")
    


    def cleanup_old_results(self, days_old: int = 30):
        """Remove results older than specified days"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM configurations 
                WHERE created_at < datetime('now', '-{} days')
            """.format(days_old))
            
            rows_deleted = cursor.rowcount
            print(f"Deleted {rows_deleted} old configurations and associated results")

def main():
    """Example usage of the database utilities"""
    db = NetworkResultsDB()
    
    # Get all configurations
    configs = db.get_configurations()
    print(f"Total configurations: {len(configs)}")
    print(configs[['id', 'N', 'config_type', 'tess_type', 'dimension']].head())
    
    # Get performance summary
    perf = db.get_performance_summary()
    if not perf.empty:
        print(f"\nPerformance Summary:")
        print(f"Average computation time: {perf['computation_time'].mean():.2f} seconds")
        print(f"Average memory usage: {perf['memory_usage'].mean():.2f} MB")
    
    # Example: Compare spectral densities for 2D configurations
    configs_2d = db.get_configurations(dimension=2)
    if len(configs_2d) >= 2:
        config_ids = configs_2d['id'].head(2).tolist()
        fig = db.compare_spectral_densities(config_ids)
        fig.savefig("spectral_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Spectral density comparison saved to spectral_comparison.png")

if __name__ == "__main__":
    main()