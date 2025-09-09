# Database Backup Procedures

This document describes the comprehensive backup system for network results databases.

## Overview

The backup system provides:
- **Automatic periodic backups** with intelligent scheduling
- **Gzip compression** (typically 70% space savings)
- **Integrated backup calls** in main analysis scripts
- **Cluster-compatible** minimal overhead operations
- **Automatic cleanup** of old backups
- **Complete restore functionality**

## Quick Start

### Manual Backup
```bash
cd utils
python backup_database.py backup --force
```

### List Backups
```bash
python backup_database.py list
```

### Automatic Backup (for periodic use)
```bash
python backup_database.py auto
```

### Restore Database
```bash
python backup_database.py restore backup_20250909_093708.db.gz
```

## Backup System Components

### 1. `backup_database.py` - Main Backup Utility

**Features:**
- Command-line interface for all backup operations
- Intelligent scheduling (minimum 1-hour intervals)
- Gzip compression (level 6 - balanced speed/size)
- Automatic cleanup (keeps 10 most recent backups)
- Database integrity verification

**Commands:**
- `backup` - Create new backup
- `list` - Show available backups
- `restore` - Restore from backup
- `auto` - Automatic backup with smart scheduling

### 2. Integrated Backup Calls

**Automatic backups are triggered by:**
- `Network_ELV2D.py` - After each analysis completion
- `generate_group_plots.py` - Before major plotting operations
- `generate_group_plots_cluster.py` - Before comprehensive analysis

**Integration benefits:**
- No manual intervention required
- Backups created at critical points
- Minimal performance impact (smart scheduling)

### 3. Backup Storage Structure

```
outputs/db_backups/
├── backup_20250909_093708.db.gz    # Compressed backups
├── backup_20250909_110245.db.gz
└── backup_20250909_143022.db.gz
```

**Naming convention:** `backup_YYYYMMDD_HHMMSS.db.gz`

## Usage Examples

### For Cluster Operations

**1. SLURM Script Integration:**
```bash
# Before major analysis
python backup_database.py auto

# Run your analysis
python Network_ELV2D.py 200 0.1 test URL V length_power_law

# After completion (automatic backup will be created)
```

**2. Periodic Backup (Cron-style):**
```bash
# Add to cluster job or cron
*/6 * * * * cd /path/to/utils && python backup_database.py auto --quiet
```

**3. Pre-deployment Backup:**
```bash
# Before starting large cluster job
python backup_database.py backup --force
```

### For Development/Testing

**1. Backup Before Experiments:**
```bash
python backup_database.py backup --force
```

**2. Quick Status Check:**
```bash
python backup_database.py list
```

**3. Restore for Testing:**
```bash
python backup_database.py restore backup_20250909_093708.db.gz --target test_db.db
```

## Advanced Configuration

### Customizing Backup Settings

Edit `backup_database.py` to modify:
```python
self.compression_level = 6  # 1-9 (speed vs compression)
self.max_backups = 10       # Number of backups to keep
self.min_interval_hours = 1 # Minimum time between backups
```

### Custom Backup Locations
```bash
python backup_database.py --db-path /custom/path/db.db --backup-dir /custom/backups backup
```

## Performance Characteristics

### Typical Performance (35MB database):
- **Backup time:** ~0.5 seconds
- **Compressed size:** ~10MB (70% reduction)
- **CPU overhead:** Minimal (level 6 compression)
- **I/O overhead:** Single sequential read/write

### Cluster Compatibility:
- **No interactive input** required
- **Exit codes** for script integration (0=success, 1=error)
- **Quiet mode** available (`--quiet` flag)
- **Robust error handling** won't crash main processes

## Backup Schedule Recommendations

### Development Environment:
- **Manual backups** before major code changes
- **Automatic backups** enabled in scripts

### Cluster Environment:
- **Periodic backups** every 6-12 hours via cron
- **Automatic backups** in all analysis scripts (enabled by default)
- **Pre-job backups** for large computational runs

### Production Environment:
- **Daily backups** via automated system
- **Pre-deployment backups** before updates
- **Multiple backup locations** for redundancy

## Troubleshooting

### Common Issues:

**1. "Database locked" error:**
```bash
# Wait for current operations to complete, then retry
python backup_database.py backup --force
```

**2. Backup directory permissions:**
```bash
# Ensure backup directory is writable
chmod 755 outputs/db_backups
```

**3. Disk space issues:**
```bash
# Clean up old backups manually if needed
python backup_database.py list
rm outputs/db_backups/backup_OLDEST_*.db.gz
```

**4. Corruption during restore:**
```bash
# Verify backup integrity
python -c "
import gzip
with gzip.open('outputs/db_backups/backup_*.db.gz', 'rb') as f:
    data = f.read()
print(f'Backup size: {len(data)} bytes - OK')
"
```

## Integration with Cluster Workflows

### Example SLURM Integration:
```bash
#!/bin/bash
#SBATCH --job-name=ELV_analysis
#SBATCH --time=48:00:00

# Create backup before starting
cd utils
python backup_database.py backup --force

# Run analysis with automatic backups
for config in config_list; do
    python Network_ELV2D.py $config
done

# Final backup after completion
python backup_database.py backup --force

# Generate summary
python backup_database.py list >> ../slrm_logs/backup_summary.log
```

## Security Considerations

- **No credentials stored** in backup files
- **Local filesystem only** (no network transfers)
- **Standard gzip compression** (widely compatible)
- **Atomic operations** (backup completes or fails cleanly)
- **Non-destructive** (original database never modified during backup)

## Monitoring and Alerts

### Check Backup Health:
```bash
# Verify recent backups exist
find outputs/db_backups -name "backup_*.db.gz" -mtime -1 | wc -l
```

### Backup Size Monitoring:
```bash
# Check if backup sizes are reasonable
du -h outputs/db_backups/backup_*.db.gz | tail -5
```