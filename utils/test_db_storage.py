#!/usr/bin/env python3
"""
Test script for dual edge density database storage
"""
import sys
import os
import numpy as np
import subprocess

def test_small_run():
    """Test with very small parameters to verify database storage"""
    
    print("Testing dual edge density storage...")
    
    # Small test parameters
    N = 5
    a = 0.1  
    simtag = "test_dual_density"
    ctype = "URL"
    ntype = "D"
    weight_type = "uniform"
    
    # Run the main script
    cmd = [
        "python", "Network_ELV2D.py",
        str(N), str(a), simtag, ctype, ntype, weight_type
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("Script completed successfully!")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
        else:
            print("Script failed!")
            print("STDERR:", result.stderr)
            print("STDOUT:", result.stdout)
            return False
            
    except subprocess.TimeoutExpired:
        print("Script timed out!")
        return False
    except Exception as e:
        print(f"Error running script: {e}")
        return False
    
    # Test database query
    try:
        from db_utils import NetworkResultsDB
        
        db = NetworkResultsDB("network_results_weights.db")
        configs = db.get_configurations()
        
        if len(configs) > 0:
            print(f"Found {len(configs)} configurations in database")
            
            # Test the new columns
            import sqlite3
            with sqlite3.connect(db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT config_id, weighted_edge_density, unweighted_edge_density 
                    FROM edge_length_variance_results 
                    ORDER BY config_id DESC 
                    LIMIT 1
                """)
                result = cursor.fetchone()
                
                if result:
                    config_id, weighted_density, unweighted_density = result
                    print(f"Latest result:")
                    print(f"  Config ID: {config_id}")
                    print(f"  Weighted density: {weighted_density}")
                    print(f"  Unweighted density: {unweighted_density}")
                    
                    if weighted_density is not None and unweighted_density is not None:
                        print("Both edge densities stored successfully!")
                        return True
                    else:
                        print("Missing edge density values")
                        return False
                else:
                    print("No ELV results found")
                    return False
        else:
            print("No configurations found in database")
            return False
            
    except Exception as e:
        print(f"Database test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_small_run()
    if success:
        print("\nAll tests passed! Dual edge density storage is working.")
    else:
        print("\nTest failed. Check the errors above.")
        sys.exit(1)