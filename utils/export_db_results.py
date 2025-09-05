#!/usr/bin/env python3
"""
Simple script to export database results for analysis
Run this to get readable data files from your hyperuniformity database.
"""

import sys
import os
from db_utils import NetworkResultsDB

def main():
    """Export database contents to readable formats"""
    
    # Check if database exists
    db_path = "network_results_weights.db"
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found!")
        print("Looking for alternative database names...")
        
        # Look for other possible database names
        possible_names = [
            "network_results.db",
            "../network_results_weights.db", 
            "../network_results.db"
        ]
        
        found = False
        for name in possible_names:
            if os.path.exists(name):
                db_path = name
                print(f"Found database: {db_path}")
                found = True
                break
        
        if not found:
            print("No database found. Make sure your job has created some results first.")
            return 1
    
    print(f"Using database: {db_path}")
    db = NetworkResultsDB(db_path)
    
    # Create exports directory
    export_dir = "database_exports"
    os.makedirs(export_dir, exist_ok=True)
    print(f"Exporting to directory: {export_dir}")
    
    try:
        # Export summary statistics
        print("\n1. Exporting summary statistics...")
        summary_file = os.path.join(export_dir, "database_summary.txt")
        db.export_summary_statistics(summary_file)
        
        # Export configurations
        print("\n2. Exporting configurations...")
        config_file = os.path.join(export_dir, "all_configurations.csv")
        db.export_configurations_csv(config_file)
        
        # Export weight comparison data
        print("\n3. Exporting weight comparison data...")
        
        configs = db.get_configurations()
        if not configs.empty:
            # All data
            all_file = os.path.join(export_dir, "weight_comparison_all.csv")
            db.export_weight_comparison_data(all_file)
            
            # By configuration type if multiple exist
            config_types = configs['config_type'].unique()
            for config_type in config_types:
                if config_type:  # Skip None values
                    type_file = os.path.join(export_dir, f"weight_comparison_{config_type}.csv")
                    db.export_weight_comparison_data(type_file, config_type=config_type)
                    print(f"   - Exported {config_type} configurations")
            
            # By tessellation type if multiple exist  
            tess_types = configs['tess_type'].unique()
            for tess_type in tess_types:
                if tess_type:  # Skip None values
                    tess_file = os.path.join(export_dir, f"weight_comparison_{tess_type}.csv")
                    db.export_weight_comparison_data(tess_file, tess_type=tess_type)
                    print(f"   - Exported {tess_type} tessellations")
        
        # Export individual results for first few configurations (for debugging)
        print("\n4. Exporting individual result samples...")
        sample_configs = configs.head(3)  # Export first 3 as examples
        for _, config in sample_configs.iterrows():
            try:
                config_id = config['id']
                individual_dir = os.path.join(export_dir, f"individual_results")
                os.makedirs(individual_dir, exist_ok=True)
                db.export_results(config_id, individual_dir)
                print(f"   - Exported config {config_id}: N={config['N']} {config['config_type']} {config['tess_type']} {config['weight_type']}")
            except Exception as e:
                print(f"   - Could not export config {config_id}: {e}")
        
        print(f"\n=== EXPORT COMPLETE ===")
        print(f"All files exported to: {export_dir}/")
        print(f"\nKey files:")
        print(f"  - {summary_file}: Database overview and statistics")
        print(f"  - {config_file}: All configuration parameters")
        print(f"  - weight_comparison_*.csv: ELV data ready for plotting")
        print(f"  - individual_results/: Sample individual results")
        
        # Quick data preview
        print(f"\n=== QUICK PREVIEW ===")
        print(f"Total configurations in database: {len(configs)}")
        if not configs.empty:
            print("Configuration breakdown:")
            print(f"  Config types: {list(configs['config_type'].value_counts().to_dict().items())}")
            print(f"  Weight types: {list(configs['weight_type'].value_counts().to_dict().items())}")
            print(f"  System sizes: {list(configs['N'].value_counts().to_dict().items())}")
        
        return 0
        
    except Exception as e:
        print(f"\nError during export: {e}")
        print("This might happen if the database is empty or corrupted.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
