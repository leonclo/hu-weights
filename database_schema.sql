-- Database schema for Network Structure Characterization
-- Supports storing configurations, parameters, and results for all three modules

CREATE TABLE IF NOT EXISTS configurations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    N INTEGER NOT NULL,                    -- Square/cube root of particle number
    a REAL,                               -- Disorder parameter (URL only)
    beamw REAL,                           -- Beam width (2DXv only)
    simname TEXT NOT NULL,                -- Simulation name/tag
    config_type TEXT NOT NULL,            -- 'URL', 'poi', 'load'
    tess_type TEXT NOT NULL,              -- 'V', 'D', 'C', 'G'
    dimension INTEGER NOT NULL,           -- 2 or 3
    binset REAL,                          -- Bin size parameter (2DXv only)
    weight_type TEXT DEFAULT 'uniform',   -- Edge weighting scheme
    weight_parameters TEXT,               -- JSON string of weight parameters
    filename TEXT,                        -- Original filename if loaded
    point_pattern BLOB,                   -- Serialized numpy array of points
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(N, a, beamw, simname, config_type, tess_type, dimension, binset, weight_type)
);

CREATE TABLE IF NOT EXISTS spectral_density_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id INTEGER NOT NULL,
    wavenumber BLOB NOT NULL,             -- Numpy array of wavenumbers
    spectral_density BLOB NOT NULL,      -- Numpy array of spectral density values
    phi2 REAL NOT NULL,                   -- Average phase fraction
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (config_id) REFERENCES configurations(id)
);

CREATE TABLE IF NOT EXISTS edge_length_variance_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id INTEGER NOT NULL,
    window_radii BLOB NOT NULL,           -- Numpy array of window radii
    variance_matrix BLOB NOT NULL,        -- Numpy array of variance data [unweighted, percent_inside, weights]
    edge_density REAL,                    -- Weighted edge density (edges per unit area)
    num_windows INTEGER NOT NULL,        -- Number of windows used
    resolution INTEGER NOT NULL,         -- Resolution parameter
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (config_id) REFERENCES configurations(id)
);

CREATE TABLE IF NOT EXISTS computation_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id INTEGER NOT NULL,
    computation_type TEXT NOT NULL,      -- '2DXv', 'ELV2D', 'ELV3D'
    computation_time REAL,               -- Time in seconds
    memory_usage REAL,                   -- Peak memory usage in MB
    parameters TEXT,                     -- JSON string of additional parameters
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (config_id) REFERENCES configurations(id)
);

-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_config_params ON configurations(N, config_type, tess_type, dimension, weight_type);
CREATE INDEX IF NOT EXISTS idx_config_created ON configurations(created_at);
CREATE INDEX IF NOT EXISTS idx_config_weights ON configurations(weight_type);
CREATE INDEX IF NOT EXISTS idx_results_config ON spectral_density_results(config_id);
CREATE INDEX IF NOT EXISTS idx_elv_config ON edge_length_variance_results(config_id);
CREATE INDEX IF NOT EXISTS idx_metadata_config ON computation_metadata(config_id);