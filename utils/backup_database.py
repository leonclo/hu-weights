#!/usr/bin/env python3
"""
Database Backup Utility for Network Results
Provides automatic, periodic, compressed backups with cleanup
Cluster-compatible with minimal overhead
"""

import os
import sys
import shutil
import gzip
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
import argparse

class DatabaseBackupManager:
    """Handles database backups with compression and cleanup"""
    
    def __init__(self, db_path="../network_results_weights.db", 
                 backup_dir="../outputs/db_backups"):
        """
        Initialize backup manager
        
        Args:
            db_path: Path to main database
            backup_dir: Directory for backup storage
        """
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir)
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup settings
        self.compression_level = 6  # Good balance of speed vs compression
        self.max_backups = 10       # Keep last 10 backups
        self.min_interval_hours = 1 # Minimum time between backups
        
    def _get_database_info(self):
        """Get basic database information"""
        if not self.db_path.exists():
            return None
            
        try:
            stat = self.db_path.stat()
            size_mb = stat.st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(stat.st_mtime)
            
            # Get record counts
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM configurations")
                config_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM edge_length_variance_results") 
                elv_count = cursor.fetchone()[0]
                
            return {
                'size_mb': size_mb,
                'modified': mod_time,
                'config_count': config_count,
                'elv_count': elv_count
            }
        except Exception as e:
            print(f"Warning: Could not get database info: {e}")
            return None
    
    def _should_backup(self, force=False):
        """Check if backup is needed"""
        if force:
            return True
            
        if not self.db_path.exists():
            print("Database does not exist - no backup needed")
            return False
            
        # Check if minimum time has passed since last backup
        backups = list(self.backup_dir.glob("backup_*.db.gz"))
        if backups:
            latest_backup = max(backups, key=lambda x: x.stat().st_mtime)
            last_backup_time = datetime.fromtimestamp(latest_backup.stat().st_mtime)
            time_since = datetime.now() - last_backup_time
            
            if time_since.total_seconds() < self.min_interval_hours * 3600:
                print(f"Last backup was {time_since} ago - skipping (min interval: {self.min_interval_hours}h)")
                return False
                
        return True
    
    def create_backup(self, force=False, compress=True):
        """
        Create database backup with optional compression
        
        Args:
            force: Force backup even if not needed
            compress: Apply gzip compression (recommended)
            
        Returns:
            Path to backup file or None if backup not created
        """
        if not self._should_backup(force):
            return None
            
        # Get database info
        db_info = self._get_database_info()
        if not db_info:
            print("Could not analyze database")
            return None
            
        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}.db"
        if compress:
            backup_name += ".gz"
            
        backup_path = self.backup_dir / backup_name
        
        print(f"Creating backup: {backup_path.name}")
        print(f"  Source: {self.db_path} ({db_info['size_mb']:.1f} MB)")
        print(f"  Records: {db_info['config_count']} configs, {db_info['elv_count']} ELV results")
        
        start_time = time.time()
        
        try:
            if compress:
                # Create compressed backup
                with open(self.db_path, 'rb') as src:
                    with gzip.open(backup_path, 'wb', compresslevel=self.compression_level) as dst:
                        shutil.copyfileobj(src, dst)
            else:
                # Create uncompressed backup
                shutil.copy2(self.db_path, backup_path)
                
            backup_time = time.time() - start_time
            backup_size_mb = backup_path.stat().st_size / (1024 * 1024)
            
            if compress:
                compression_ratio = backup_size_mb / db_info['size_mb']
                print(f"  Backup created: {backup_size_mb:.1f} MB ({compression_ratio:.1%} of original)")
            else:
                print(f"  Backup created: {backup_size_mb:.1f} MB")
                
            print(f"  Time: {backup_time:.1f} seconds")
            
            # Cleanup old backups
            self._cleanup_old_backups()
            
            return backup_path
            
        except Exception as e:
            print(f"Backup failed: {e}")
            # Clean up failed backup
            if backup_path.exists():
                backup_path.unlink()
            return None
    
    def _cleanup_old_backups(self):
        """Remove old backups keeping only the most recent ones"""
        backups = list(self.backup_dir.glob("backup_*.db*"))
        
        if len(backups) <= self.max_backups:
            return
            
        # Sort by modification time (oldest first)
        backups.sort(key=lambda x: x.stat().st_mtime)
        
        # Remove oldest backups
        to_remove = backups[:-self.max_backups]
        
        print(f"Cleaning up {len(to_remove)} old backups (keeping {self.max_backups} most recent)")
        
        for backup in to_remove:
            try:
                backup.unlink()
                print(f"  Removed: {backup.name}")
            except Exception as e:
                print(f"  Warning: Could not remove {backup.name}: {e}")
    
    def list_backups(self):
        """List all available backups"""
        backups = list(self.backup_dir.glob("backup_*.db*"))
        
        if not backups:
            print("No backups found")
            return
            
        # Sort by modification time (newest first)  
        backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        print(f"Found {len(backups)} backups in {self.backup_dir}:")
        print()
        
        for backup in backups:
            stat = backup.stat()
            size_mb = stat.st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(stat.st_mtime)
            compressed = backup.suffix == '.gz'
            
            print(f"  {backup.name}")
            print(f"    Created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"    Size: {size_mb:.1f} MB {'(compressed)' if compressed else ''}")
            print()
    
    def restore_backup(self, backup_name, target_path=None):
        """
        Restore database from backup
        
        Args:
            backup_name: Name of backup file to restore
            target_path: Target path for restored database (default: original location)
        """
        if target_path is None:
            target_path = self.db_path
        else:
            target_path = Path(target_path)
            
        backup_path = self.backup_dir / backup_name
        
        if not backup_path.exists():
            print(f"Backup not found: {backup_path}")
            return False
            
        print(f"Restoring backup: {backup_name}")
        print(f"  Target: {target_path}")
        
        try:
            if backup_path.suffix == '.gz':
                # Decompress and restore
                with gzip.open(backup_path, 'rb') as src:
                    with open(target_path, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
            else:
                # Direct copy
                shutil.copy2(backup_path, target_path)
                
            print("Restore completed successfully")
            return True
            
        except Exception as e:
            print(f"Restore failed: {e}")
            return False

def main():
    """Main backup utility function"""
    parser = argparse.ArgumentParser(description="Database backup utility")
    parser.add_argument("--db-path", default="../network_results_weights.db",
                       help="Path to database (default: ../network_results_weights.db)")
    parser.add_argument("--backup-dir", default="../outputs/db_backups", 
                       help="Backup directory (default: ../outputs/db_backups)")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Create backup')
    backup_parser.add_argument('--force', action='store_true',
                              help='Force backup even if not needed')
    backup_parser.add_argument('--no-compress', action='store_true',
                              help='Disable compression')
    
    # List command
    subparsers.add_parser('list', help='List backups')
    
    # Restore command  
    restore_parser = subparsers.add_parser('restore', help='Restore from backup')
    restore_parser.add_argument('backup_name', help='Name of backup to restore')
    restore_parser.add_argument('--target', help='Target path for restored database')
    
    # Auto command (for periodic use)
    auto_parser = subparsers.add_parser('auto', help='Automatic backup (for cron/periodic use)')
    auto_parser.add_argument('--quiet', action='store_true', help='Suppress output except errors')
    
    args = parser.parse_args()
    
    # Default to auto mode if no command specified
    if args.command is None:
        args.command = 'auto'
        args.quiet = False
    
    # Initialize backup manager
    manager = DatabaseBackupManager(args.db_path, args.backup_dir)
    
    try:
        if args.command == 'backup':
            backup_path = manager.create_backup(
                force=args.force, 
                compress=not args.no_compress
            )
            if backup_path:
                print(f"Backup successful: {backup_path}")
            else:
                print("Backup was not created")
                return 1
                
        elif args.command == 'list':
            manager.list_backups()
            
        elif args.command == 'restore':
            success = manager.restore_backup(args.backup_name, args.target)
            return 0 if success else 1
            
        elif args.command == 'auto':
            # Automatic backup for periodic use
            if not args.quiet:
                print(f"Auto backup check at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
            backup_path = manager.create_backup(force=False, compress=True)
            
            if backup_path and not args.quiet:
                print(f"Auto backup created: {backup_path.name}")
            elif not backup_path and not args.quiet:
                print("No backup needed")
                
    except KeyboardInterrupt:
        print("\nOperation cancelled")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())