# Edge weight structure
* db added to store results in a nicely accessible format

* still need to check input for weighting arguments (should be easier to test with command line changes for weights e.g. )

## loop through edge lengths >
* store in tovar slice 1 : edge length in window
* store in tovar slice 2: percent of edge lengths in window (inside/total edge length) 
* store in tovar slice 3: percent size increase from edge weights

## calculate_edge_weights returns edge_length:
* store in edge_weights slice 1: edge length
* store in edge_weights slice 2: edge weight
* store in edge_weights slice 3: edge length/weighted edge

## density
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
* weight_type = str(sys.argv[6]) if len(sys.argv) > 6 else 'uniform' #beam vcharacteristics for function calls
* params_str = str(sys.argv[7]) if len(sys.argv) > 7 else None
optional: fn = str(sys.argv[8])