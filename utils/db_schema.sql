-- Network Results Database Schema
-- Database schema for ELV analysis results storage

-- Main configuration table
CREATE TABLE IF NOT EXISTS configurations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    N INTEGER,                              -- System size (square root of particle count)
    a REAL,                                 -- Disorder parameter for URL patterns  
    beamw REAL,                             -- Beam width parameter / density
    simname TEXT,                           -- Simulation identifier/tag
    config_type TEXT,                       -- Configuration type: 'poi', 'URL', 'load'
    tess_type TEXT,                         -- Tessellation type: 'V', 'D', 'G', 'C'
    dimension INTEGER,                      -- Spatial dimension (typically 2)
    binset REAL,                           -- Binning parameter
    weight_type TEXT DEFAULT 'uniform',    -- Edge weighting: 'uniform', 'length_power_law', 'length_inverse'
    weight_parameters TEXT,                 -- JSON string of weight parameters (alpha, beta, etc.)
    filename TEXT,                          -- Associated data filename
    point_pattern BLOB,                     -- Pickled point coordinates array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Edge Length Variance results table
CREATE TABLE IF NOT EXISTS edge_length_variance_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id INTEGER,                      -- Foreign key to configurations
    window_radii BLOB,                      -- Pickled array of analysis window radii
    variance_matrix BLOB,                   -- Pickled 3D variance matrix [3, Nwind, reso]
    edge_density REAL,                      -- Base edge density
    weighted_edge_density REAL,            -- Weighted edge density  
    unweighted_edge_density REAL,          -- Unweighted edge density
    num_windows INTEGER,                    -- Number of analysis windows (typically 250)
    resolution INTEGER,                     -- Resolution of radius sampling (typically 500)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (config_id) REFERENCES configurations(id)
);

-- Spectral density results table  
CREATE TABLE IF NOT EXISTS spectral_density_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id INTEGER,                      -- Foreign key to configurations
    wavenumber BLOB,                        -- Pickled wavenumber array
    spectral_density BLOB,                  -- Pickled spectral density S(k)
    phi2 REAL,                             -- Hyperuniformity parameter φ₂
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (config_id) REFERENCES configurations(id)
);

-- Computation metadata table
CREATE TABLE IF NOT EXISTS computation_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id INTEGER,                      -- Foreign key to configurations
    computation_type TEXT,                  -- Type: 'ELV2D_weighted', 'spectral_analysis', etc.
    computation_time REAL,                  -- Wall clock time in seconds
    memory_usage REAL,                      -- Peak memory usage in MB
    parameters TEXT,                        -- JSON string of computation parameters
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (config_id) REFERENCES configurations(id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_config_type ON configurations(config_type);
CREATE INDEX IF NOT EXISTS idx_tess_type ON configurations(tess_type);
CREATE INDEX IF NOT EXISTS idx_weight_type ON configurations(weight_type);
CREATE INDEX IF NOT EXISTS idx_dimension ON configurations(dimension);
CREATE INDEX IF NOT EXISTS idx_created_at ON configurations(created_at);
CREATE INDEX IF NOT EXISTS idx_elv_config_id ON edge_length_variance_results(config_id);
CREATE INDEX IF NOT EXISTS idx_spectral_config_id ON spectral_density_results(config_id);
CREATE INDEX IF NOT EXISTS idx_metadata_config_id ON computation_metadata(config_id);