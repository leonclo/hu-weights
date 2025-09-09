#!/usr/bin/env python3
"""
Update existing database to add new edge density columns
"""
import sqlite3
import os

def update_database_schema(db_path="network_results_weights.db"):
    """Add new columns to existing database"""
    
    if not os.path.exists(db_path):
        print(f"Database {db_path} does not exist!")
        return False
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check current schema
            cursor.execute("PRAGMA table_info(edge_length_variance_results)")
            columns = cursor.fetchall()
            
            print(f"Current columns in edge_length_variance_results:")
            for col in columns:
                print(f"  {col[1]} ({col[2]})")
            
            # Add new columns if they don't exist
            has_weighted = any(col[1] == 'weighted_edge_density' for col in columns)
            has_unweighted = any(col[1] == 'unweighted_edge_density' for col in columns)
            
            if not has_weighted:
                print("Adding weighted_edge_density column...")
                cursor.execute("ALTER TABLE edge_length_variance_results ADD COLUMN weighted_edge_density REAL")
                print("  Added weighted_edge_density")
            else:
                print("  weighted_edge_density already exists")
                
            if not has_unweighted:
                print("Adding unweighted_edge_density column...")
                cursor.execute("ALTER TABLE edge_length_variance_results ADD COLUMN unweighted_edge_density REAL")
                print("  Added unweighted_edge_density")
            else:
                print("  unweighted_edge_density already exists")
            
            # Copy existing edge_density to new columns if they were just created
            if not has_weighted or not has_unweighted:
                print("Copying existing edge_density values to new columns...")
                cursor.execute("""
                    UPDATE edge_length_variance_results 
                    SET weighted_edge_density = edge_density,
                        unweighted_edge_density = edge_density 
                    WHERE weighted_edge_density IS NULL OR unweighted_edge_density IS NULL
                """)
                print("  Copied existing values")
            
            conn.commit()
            
            # Verify the update
            cursor.execute("PRAGMA table_info(edge_length_variance_results)")
            new_columns = cursor.fetchall()
            
            print(f"\nUpdated columns:")
            for col in new_columns:
                print(f"  {col[1]} ({col[2]})")
            
            print(f"\nDatabase schema updated successfully!")
            return True
            
    except Exception as e:
        print(f"Error updating database: {e}")
        return False

if __name__ == "__main__":
    success = update_database_schema()
    if success:
        print("\nSchema update completed!")
    else:
        print("\nSchema update failed!")