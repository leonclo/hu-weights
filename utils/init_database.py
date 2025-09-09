#!/usr/bin/env python3
"""
Initialize database with proper schema
Run this to create a new database or verify existing schema
"""

import sqlite3
import os
from pathlib import Path

def init_database(db_path="../network_results_weights.db"):
    """Initialize database with schema from db_schema.sql"""
    
    # Read schema from file
    schema_path = Path(__file__).parent / "db_schema.sql"
    
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    
    # Create database and apply schema
    print(f"Initializing database: {db_path}")
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Execute schema
        cursor.executescript(schema_sql)
        
        # Verify tables were created
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print("Created tables:")
        for table in tables:
            print(f"  - {table[0]}")
            
        print("Database initialization complete!")
        
        # Show some basic stats if data exists
        cursor.execute("SELECT COUNT(*) FROM configurations")
        config_count = cursor.fetchone()[0]
        
        if config_count > 0:
            print(f"\nExisting data found: {config_count} configurations")
            
            cursor.execute("SELECT DISTINCT config_type FROM configurations")
            config_types = [row[0] for row in cursor.fetchall()]
            print(f"Configuration types: {config_types}")
            
            cursor.execute("SELECT DISTINCT weight_type FROM configurations") 
            weight_types = [row[0] for row in cursor.fetchall()]
            print(f"Weight types: {weight_types}")
        else:
            print("\nDatabase is empty - ready for new data")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize network results database")
    parser.add_argument("--db-path", default="../network_results_weights.db",
                       help="Path to database file (default: ../network_results_weights.db)")
    
    args = parser.parse_args()
    
    try:
        init_database(args.db_path)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())