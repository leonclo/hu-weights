# Network ELV Analysis

**Quick Navigation:** [Usage](#usage) | [Weight Structure](#edge-weight-structure) | [Examples](#examples-for-usage) | [SLURM](#examples-for-slrm-usage) | [Structure](#structure)

## Usage
 >  run with ```bash Network_ELV2D.py args```
 >  plot weight comparisons/summary plots with ```bash generate_group_plots.py```
 >  plot cluster analysis plots (aggregate data) with ```bash generate_group_plots_cluster.py args``` 
 > results stored in network_results_weights.db
 > automatic backups in outputs/db_backups (see BACKUP_PROCEDURES.md)
## Edge weight structure

### todos
* [x] edge weight calculation : 
* [x] db added to store results in a nicely accessible format
* [x] still need to check input for weighting arguments (should be easier to test with command line changes for weights e.g. )

### loop through edge lengths  > 
* store in tovar slice 1 : ~edge length in window~ total edge length (althoughh calculate_edge_weights should return the "edge weight" I'll double check that I'm not duplicating weight calculations)
* store in tovar slice 2: percent of edge lengths in window (inside/total edge length) 
* store in tovar slice 3: percent size increase from edge weights

### calculate_edge_weights returns edge_length:
* store in edge_weights slice 1: edge length
* store in edge_weights slice 2: edge weight
* store in edge_weights slice 3: edge length/weighted edge

### density
* calculate weighted_edge_density using edge_density_function with weighted edges (uselen_w line 491)
* calculate edge_density


# input
## main weighting schemes
* uniform
* length_power_law
* length_inverse

## arguments
* N = int(sys.argv[1]) #Square root of the number of particles in the system
* a = float(sys.argv[2]) #Parameter describing the degree of local translational disorder in the URL point patterns
* simtag = str(sys.argv[3]) #Specific identifying name for particular simulation
* ctype = str(sys.argv[4]) #Configuration type
* ntype = str(sys.argv[5]) #Tessellation type (C,V,G,D)
* weight_type = str(sys.argv[6]) if len(sys.argv)  >  6 else 'uniform' #beam vcharacteristics for function calls
* params_str = str(sys.argv[7]) if len(sys.argv)  >  7 else None
optional: fn = str(sys.argv[8])

## Examples for usage

**Individual ELV Analysis:**
```bash
# Basic syntax: N a simtag ctype ntype weight_type [params]
python Network_ELV2D.py 200 0.1 test URL V uniform
python Network_ELV2D.py 200 0.1 test URL V length_power_law alpha=2.0
python Network_ELV2D.py 200 0.1 test poi V length_inverse alpha=3.0
python Network_ELV2D.py 10 0.0 comp URL G length_power_law alpha=1.5,beta=2.0
```

**Group Plot Generation:**
```bash
python generate_group_plots.py
```

**Comprehensive Cluster Analysis:**
```bash
# Syntax: [cutoff_days] [N_filter]
python generate_group_plots_cluster.py                    # All sizes, last 7 days
python generate_group_plots_cluster.py 14                 # All sizes, last 14 days  
python generate_group_plots_cluster.py 7 "200,300"        # Large systems only
python generate_group_plots_cluster.py 14 "5,8,10"        # Small systems, 14 days
python generate_group_plots_cluster.py 30 "200"           # Single size, 30 days
```

## Examples for slrm usage
### Generation
        ```slrm
        ### Generation
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
        ```
### Plotting
        ```slrm
        #!/bin/bash
        #SBATCH --job-name=ELV_power_vs_uniform
        #SBATCH --partition=general
        #SBATCH --nodes=1
        #SBATCH --ntasks-per-node=4
        #SBATCH --cpus-per-task=1
        #SBATCH --mem=8G
        #SBATCH --time=01:00:00
        #SBATCH --output=elv_cluster_analysis_%j.out
        #SBATCH --error=elv_cluster_analysis_%j.err
        #SBATCH --mail-type=BEGIN,END,FAIL
        #SBATCH --mail-user=onyen@email.unc.edu

        module load python/3.12.2
        module load gcc/12.2.0

        # Install required packages
        pip install --user scikit-learn seaborn

        cd utils
        python generate_group_plots_cluster.py 2 "200,300"
        ```


## Structure

### Output Structure:
```
cluster-run-1/
├── outputs/
│   ├── plots/          # all visualization files                                                                              |
│   ├── data/           # npy config files + variance data                                                                     |
│   ├── db_backups/     # automatic database backups                                                                           |            
│   └── logs/           # analysis logs                                                                                        |
├── slrm logs/          # SLURM job logs                                                                                       |
├── utils/              # analysis scripts                                                                                     |
├── network_results_weights.db  # main database                                                                                |
└── README.md                                                                                                                  |
```
### Code structure:
        ```
        main()
        |-CLI argument parsing                         > N, a, simtag, ctype, ntype, weight_type, params                       |
        |-Point pattern generation:                                                                                            |
        | |-ctype='poi'                                > np.random.rand(N*2,2) * N                                             |
        | |-ctype='load'                               > read from file                                                        |
        | |-ctype='URL'                                > Z2_gen(N) > lat[N,N,2] > URL(a, flatlat)                              |
        |-wrap(N, flatlat)                             > periodic boundary enforcement                                         |
        |-tile(flatlat, N)                             > tURL[N*2  x 9, 2] (3 x 3 tiling)                                      |
        |-Tessellation generation:                                                                                             |
        | |-ntype='V'                                  > sp.spatial.Voronoi(tURL) > vor.ridge_vertices                         |
        | |-ntype='D'                                  > delaunay_graph_2d(tURL) > using[~edges,2]                             |
        | |-ntype='G'                                  > gabriel_graph_2d(tURL) > using[~edges,2]                              |
        | |-ntype='C'                                  > Delaunay + centroids > using[~edges,2]                                |
        |-calculate_edge_weights()                     > edge_weights[~edges,3] (if not uniform)                               |
        |-Edge length calculation                      > uselen[~edges], uselen_w[~edges]                                      |
        |-ELV computation:                                                                                                     |
        | |-250 x  window loop (Nwind):                                                                                        |
        | | |-mindist()                                > distto[~edges] (edge-to-window distances)                             |
        | | |-500 x  radius loop (reso):                                                                                       |
        | | | |-np.argwhere(distto < Rs[j])            > etc indices                                                           |
        | | | |-len(etc) x  edge loop:                                                                                         |
        | | |     |-eclen()                            > circle_line_segment_intersection()                                    |
        | | |-tovar[3,i,j] accumulation (slice 0: length, 1: percent, 2: weights)                                              |
        |-edge_density()                               > weighted_edge_density, edge_density_n                                 |
        |-Data persistence:                                                                                                    |
        | |-.npy saves                                 > tovar, Rs, density files                                              |
        | |-plot_elv_variance()                        > .png visualization                                                    |
        | |-NetworkResultsDB.save_*()                  > SQLite database storage                                               |
        |-Computation metadata logging
        ```