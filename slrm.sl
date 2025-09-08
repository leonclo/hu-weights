#!/bin/bash
#SBATCH --job-name=weighted_hyperuniformity_plots_
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=elv_weights_db_%j.out
#SBATCH --error=elv_weights_db_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=youremail@email.unc.edu

module load python/3.12.2
module load gcc/12.2.0

cd "$SLURM_DIR"

python ../export_db_results.py > progress_$(date +%H%M).txt
python Network_ELV2D.py 200 0.0 "POI_uniform_V" poi V uniform

python ../export_db_results.py > progress_$(date +%H%M).txt
python Network_ELV2D.py 200 0.0 "POI_power_V" poi V length_power_law

python ../export_db_results.py > progress_$(date +%H%M).txt
python Network_ELV2D.py 200 0.0 "POI_inverse_V" poi V length_inverse

python ../export_db_results.py > progress_$(date +%H%M).txt
python Network_ELV2D.py 200 0.0 "POI_centrality_V" poi V betweenness_centrality

python ../export_db_results.py > progress_$(date +%H%M).txt
python Network_ELV2D.py 300 0.5 "URL_uniform_V" URL V uniform

python ../export_db_results.py > progress_$(date +%H%M).txt
python Network_ELV2D.py 300 0.5 "URL_power_V" URL V length_power_law

python ../export_db_results.py > progress_$(date +%H%M).txt
python Network_ELV2D.py 300 0.5 "URL_inverse_V" URL V length_inverse

python ../export_db_results.py > progress_$(date +%H%M).txt
python Network_ELV2D.py 300 0.5 "URL_centrality_V" URL V betweenness_centrality

python generate_group_plots.py "network_results_weights.db"