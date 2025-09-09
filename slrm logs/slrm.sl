#!/bin/bash
#SBATCH --job-name=ELV_power_vs_uniform
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=elv_cluster_analysis_%j.out
#SBATCH --error=elv_cluster_analysis_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leonleon@email.unc.edu

# Load required modules
module load python/3.12.2
module load gcc/12.2.0

# Change to utils directory
cd utils

# Initialize progress tracking
echo "Starting ELV Cluster Analysis: Power Law vs Uniform Weighting" > ../cluster_progress.log
echo "Start time: $(date)" >> ../cluster_progress.log
echo "======================================================" >> ../cluster_progress.log

# Function to run analysis and track progress
run_analysis() {
    local job_num=$1
    local N=$2
    local a=$3
    local simtag=$4
    local ctype=$5
    local ntype=$6
    local weight=$7
    local alpha=${8:-2.0}
    
    echo "[$job_num/30] Starting: $ctype N=$N $ntype $weight ($(date))" >> ../cluster_progress.log
    
    if [ "$weight" = "length_power_law" ]; then
        python Network_ELV2D.py $N $a $simtag $ctype $ntype $weight "alpha=$alpha"
    else
        python Network_ELV2D.py $N $a $simtag $ctype $ntype $weight
    fi
    
    if [ $? -eq 0 ]; then
        echo "[$job_num/30] ✓ Completed: $ctype N=$N $ntype $weight ($(date))" >> ../cluster_progress.log
    else
        echo "[$job_num/30] ✗ FAILED: $ctype N=$N $ntype $weight ($(date))" >> ../cluster_progress.log
    fi
}

# =============================================================================
# POISSON CONFIGURATIONS (N=200, a=0) - Creating Hyperuniformity
# =============================================================================

echo "Phase 1: Poisson Point Patterns (N=200)" >> ../cluster_progress.log

# Voronoi tessellations
run_analysis 1  200 0.0 "poi_uniform_v1" poi V uniform
run_analysis 2  200 0.0 "poi_power15_v1" poi V length_power_law 1.5
run_analysis 3  200 0.0 "poi_power20_v1" poi V length_power_law 2.0
run_analysis 4  200 0.0 "poi_power25_v1" poi V length_power_law 2.5

# Gabriel graph tessellations
run_analysis 5  200 0.0 "poi_uniform_g1" poi G uniform
run_analysis 6  200 0.0 "poi_power15_g1" poi G length_power_law 1.5
run_analysis 7  200 0.0 "poi_power20_g1" poi G length_power_law 2.0
run_analysis 8  200 0.0 "poi_power25_g1" poi G length_power_law 2.5

# Circumcenter tessellations
run_analysis 9  200 0.0 "poi_uniform_c1" poi C uniform
run_analysis 10 200 0.0 "poi_power20_c1" poi C length_power_law 2.0

# =============================================================================
# URL CONFIGURATIONS (N=300, a=0.5) - Breaking Hyperuniformity
# =============================================================================

echo "Phase 2: URL Point Patterns (N=300, a=0.5)" >> ../cluster_progress.log

# Voronoi tessellations
run_analysis 11 300 0.5 "url_uniform_v1" URL V uniform
run_analysis 12 300 0.5 "url_power15_v1" URL V length_power_law 1.5
run_analysis 13 300 0.5 "url_power20_v1" URL V length_power_law 2.0
run_analysis 14 300 0.5 "url_power25_v1" URL V length_power_law 2.5

# Gabriel graph tessellations  
run_analysis 15 300 0.5 "url_uniform_g1" URL G uniform
run_analysis 16 300 0.5 "url_power15_g1" URL G length_power_law 1.5
run_analysis 17 300 0.5 "url_power20_g1" URL G length_power_law 2.0
run_analysis 18 300 0.5 "url_power25_g1" URL G length_power_law 2.5

# Circumcenter tessellations
run_analysis 19 300 0.5 "url_uniform_c1" URL C uniform
run_analysis 20 300 0.5 "url_power20_c1" URL C length_power_law 2.0

# Delaunay tessellations
run_analysis 21 300 0.5 "url_uniform_d1" URL D uniform
run_analysis 22 300 0.5 "url_power20_d1" URL D length_power_law 2.0

# =============================================================================
# MODERATE URL CONFIGURATIONS (N=250, a=0.1) - Intermediate Disorder
# =============================================================================

echo "Phase 3: Moderate URL Point Patterns (N=250, a=0.1)" >> ../cluster_progress.log

# Voronoi tessellations
run_analysis 23 250 0.1 "url01_uniform_v1" URL V uniform  
run_analysis 24 250 0.1 "url01_power20_v1" URL V length_power_law 2.0

# Gabriel graph tessellations
run_analysis 25 250 0.1 "url01_uniform_g1" URL G uniform
run_analysis 26 250 0.1 "url01_power20_g1" URL G length_power_law 2.0

# =============================================================================
# ADDITIONAL REALIZATIONS FOR STATISTICS
# =============================================================================

echo "Phase 4: Additional Realizations" >> ../cluster_progress.log

# Key configurations with second realizations
run_analysis 27 200 0.0 "poi_uniform_v2" poi V uniform
run_analysis 28 200 0.0 "poi_power20_v2" poi V length_power_law 2.0
run_analysis 29 300 0.5 "url_uniform_g2" URL G uniform  
run_analysis 30 300 0.5 "url_power20_g2" URL G length_power_law 2.0

# =============================================================================
# POST-PROCESSING: GENERATE ANALYSIS PLOTS
# =============================================================================

echo "======================================================" >> ../cluster_progress.log
echo "Post-Processing: Generating Group Analysis Plots" >> ../cluster_progress.log
echo "Start time: $(date)" >> ../cluster_progress.log

# Generate comprehensive comparison plots
python generate_group_plots.py ../network_results_weights.db

if [ $? -eq 0 ]; then
    echo "✓ Group plots generated successfully ($(date))" >> ../cluster_progress.log
else
    echo "✗ Group plots generation FAILED ($(date))" >> ../cluster_progress.log
fi

# Export database summary for verification
python export_db_results.py > ../final_database_summary.txt

echo "======================================================" >> ../cluster_progress.log
echo "Cluster Analysis Complete: $(date)" >> ../cluster_progress.log
echo "Total configurations analyzed: 30" >> ../cluster_progress.log
echo "Focus: Power Law (α=1.5,2.0,2.5) vs Uniform weighting" >> ../cluster_progress.log
echo "Tessellations: Voronoi, Gabriel, Circumcenter, Delaunay" >> ../cluster_progress.log
echo "Point patterns: Poisson (N=200), URL a=0.1 (N=250), URL a=0.5 (N=300)" >> ../cluster_progress.log
echo "Results: plots/ directory, database summary in final_database_summary.txt" >> ../cluster_progress.log

# Final database statistics
echo "======================================================" >> ../cluster_progress.log
echo "DATABASE SUMMARY:" >> ../cluster_progress.log
python -c "
from utils.db_utils import NetworkResultsDB
db = NetworkResultsDB('../network_results_weights.db')
configs = db.get_configurations(dimension=2)
print(f'Total configurations: {len(configs)}')
print(f'Weight types: {sorted(configs[\"weight_type\"].unique())}')
print(f'Configuration types: {sorted(configs[\"config_type\"].unique())}')
print(f'Tessellation types: {sorted(configs[\"tess_type\"].unique())}')
print(f'System sizes: {sorted(configs[\"N\"].unique())}')
" >> ../cluster_progress.log

echo "All analysis complete! Check cluster_progress.log for detailed status." >> ../cluster_progress.log