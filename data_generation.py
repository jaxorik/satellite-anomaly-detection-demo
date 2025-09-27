import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, sobel
from scipy.signal import convolve2d

np.random.seed(42)

def generate_realistic_satellite_data(grid_size=64, noise_level=0.05):
    """
    Generate realistic synthetic satellite imagery with multiple spectral bands
    and spatially-structured anomalies relevant to defense and disaster response.
    
    Spectral Bands (10 total):
    - RGB: Red, Green, Blue (visible spectrum)
    - NIR: Near-Infrared
    - SWIR1, SWIR2: Short-Wave Infrared (penetrates haze, detects moisture)
    - TIR1, TIR2: Thermal Infrared (heat detection)
    - NDVI: Normalized Difference Vegetation Index
    - NDWI: Normalized Difference Water Index
    
    Anomaly Types:
    1. New Construction: High reflectance, geometric edges, low vegetation
    2. Camouflaged Structures: Visible/NIR mismatch (fake vegetation)
    3. Fires: Extreme thermal signature with smoke (low visible, high thermal)
    4. Flooding: Water signature in normally dry areas
    5. Vehicle Concentrations: Small bright spots with thermal signatures
    """
    
    # === STEP 1: Generate base terrain with spatial structure ===
    
    # Create spatially correlated base terrain using multiple octaves of noise
    def generate_perlin_like(shape, scale=10):
        """Simple multi-scale noise for terrain generation"""
        result = np.zeros(shape)
        for octave in range(4):
            freq = 2 ** octave
            noise = np.random.randn(shape[0] // freq + 1, shape[1] // freq + 1)
            noise = gaussian_filter(noise, sigma=1)
            # Upsample
            from scipy.ndimage import zoom
            noise_scaled = zoom(noise, freq, order=1)[:shape[0], :shape[1]]
            result += noise_scaled / (2 ** octave)
        return (result - result.min()) / (result.max() - result.min())
    
    terrain_elevation = generate_perlin_like((grid_size, grid_size))
    terrain_type = generate_perlin_like((grid_size, grid_size), scale=5)
    
    # Classify terrain: water (0-0.3), vegetation (0.3-0.6), urban/bare (0.6-1.0)
    water_mask = terrain_type < 0.3
    veg_mask = (terrain_type >= 0.3) & (terrain_type < 0.6)
    urban_mask = terrain_type >= 0.6
    
    # === STEP 2: Generate spectral bands based on terrain ===
    
    # Initialize bands
    n_pixels = grid_size * grid_size
    
    # Visible bands (RGB)
    red = np.zeros((grid_size, grid_size))
    green = np.zeros((grid_size, grid_size))
    blue = np.zeros((grid_size, grid_size))
    
    # Water: low red, moderate green/blue
    red[water_mask] = 0.1 + 0.1 * np.random.random(water_mask.sum())
    green[water_mask] = 0.2 + 0.15 * np.random.random(water_mask.sum())
    blue[water_mask] = 0.3 + 0.2 * np.random.random(water_mask.sum())
    
    # Vegetation: low red/blue, high green
    red[veg_mask] = 0.15 + 0.1 * np.random.random(veg_mask.sum())
    green[veg_mask] = 0.4 + 0.2 * np.random.random(veg_mask.sum())
    blue[veg_mask] = 0.1 + 0.1 * np.random.random(veg_mask.sum())
    
    # Urban/bare: moderate all channels
    red[urban_mask] = 0.3 + 0.2 * np.random.random(urban_mask.sum())
    green[urban_mask] = 0.25 + 0.2 * np.random.random(urban_mask.sum())
    blue[urban_mask] = 0.2 + 0.15 * np.random.random(urban_mask.sum())
    
    # Smooth spatial transitions
    red = gaussian_filter(red, sigma=1.5)
    green = gaussian_filter(green, sigma=1.5)
    blue = gaussian_filter(blue, sigma=1.5)
    
    # NIR (Near-Infrared) - vegetation reflects strongly
    nir = np.zeros((grid_size, grid_size))
    nir[water_mask] = 0.05 + 0.05 * np.random.random(water_mask.sum())
    nir[veg_mask] = 0.6 + 0.2 * np.random.random(veg_mask.sum())
    nir[urban_mask] = 0.2 + 0.15 * np.random.random(urban_mask.sum())
    nir = gaussian_filter(nir, sigma=1.5)
    
    # SWIR bands (Short-Wave Infrared) - moisture detection
    swir1 = nir * 0.7 + 0.1 * np.random.random((grid_size, grid_size))
    swir2 = nir * 0.6 + 0.1 * np.random.random((grid_size, grid_size))
    swir1[water_mask] *= 0.3  # Water absorbs SWIR
    swir2[water_mask] *= 0.3
    
    # Thermal bands (heat signature)
    tir1 = 0.3 + 0.2 * terrain_elevation + 0.1 * np.random.random((grid_size, grid_size))
    tir2 = tir1 * 0.9 + 0.05 * np.random.random((grid_size, grid_size))
    tir1[urban_mask] += 0.15  # Urban heat island effect
    tir2[urban_mask] += 0.15
    
    # Computed indices
    ndvi = (nir - red) / (nir + red + 1e-8)  # Vegetation index
    ndwi = (green - nir) / (green + nir + 1e-8)  # Water index
    
    # === STEP 3: Add edge/texture features ===
    
    # Edge magnitude from visible bands
    edge_x = sobel(green, axis=0)
    edge_y = sobel(green, axis=1)
    edge_magnitude = np.sqrt(edge_x**2 + edge_y**2)
    edge_magnitude = (edge_magnitude - edge_magnitude.min()) / (edge_magnitude.max() - edge_magnitude.min() + 1e-8)
    
    # Local texture variance (using a sliding window approximation)
    texture_variance = gaussian_filter(green**2, sigma=2) - gaussian_filter(green, sigma=2)**2
    texture_variance = (texture_variance - texture_variance.min()) / (texture_variance.max() - texture_variance.min() + 1e-8)
    
    # === STEP 4: Inject realistic anomalies ===
    
    anomaly_mask = np.zeros((grid_size, grid_size), dtype=bool)
    anomaly_types = np.zeros((grid_size, grid_size), dtype=int)
    
    # Anomaly 1: New Construction (5-8 instances)
    n_construction = np.random.randint(5, 9)
    for _ in range(n_construction):
        # Place in urban or bare areas
        valid_locations = np.argwhere(urban_mask)
        if len(valid_locations) > 0:
            center = valid_locations[np.random.randint(len(valid_locations))]
            size = np.random.randint(3, 7)
            y, x = center
            y_slice = slice(max(0, y-size), min(grid_size, y+size))
            x_slice = slice(max(0, x-size), min(grid_size, x+size))
            
            # Create rectangular structure
            structure = np.ones((y_slice.stop - y_slice.start, x_slice.stop - x_slice.start))
            
            # High reflectance in visible/SWIR, geometric edges, no vegetation
            red[y_slice, x_slice] = np.maximum(red[y_slice, x_slice], 0.6 + 0.1 * structure)
            green[y_slice, x_slice] = np.maximum(green[y_slice, x_slice], 0.55 + 0.1 * structure)
            blue[y_slice, x_slice] = np.maximum(blue[y_slice, x_slice], 0.5 + 0.1 * structure)
            nir[y_slice, x_slice] = np.minimum(nir[y_slice, x_slice], 0.2)  # No vegetation
            ndvi[y_slice, x_slice] = -0.2
            edge_magnitude[y_slice, x_slice] = 0.9  # Sharp edges
            
            anomaly_mask[y_slice, x_slice] = True
            anomaly_types[y_slice, x_slice] = 1
    
    # Anomaly 2: Camouflaged Structures (3-5 instances)
    n_camouflage = np.random.randint(3, 6)
    for _ in range(n_camouflage):
        valid_locations = np.argwhere(veg_mask & ~anomaly_mask)
        if len(valid_locations) > 0:
            center = valid_locations[np.random.randint(len(valid_locations))]
            size = np.random.randint(2, 5)
            y, x = center
            y_slice = slice(max(0, y-size), min(grid_size, y+size))
            x_slice = slice(max(0, x-size), min(grid_size, x+size))
            
            # Visible looks like vegetation, but NIR is wrong
            green[y_slice, x_slice] = 0.5  # Painted green
            red[y_slice, x_slice] = 0.15
            nir[y_slice, x_slice] = 0.25  # But NIR reveals it's not real vegetation
            ndvi[y_slice, x_slice] = 0.1  # Abnormally low for "vegetation"
            
            anomaly_mask[y_slice, x_slice] = True
            anomaly_types[y_slice, x_slice] = 2
    
    # Anomaly 3: Fires (2-4 instances)
    n_fires = np.random.randint(2, 5)
    for _ in range(n_fires):
        valid_locations = np.argwhere((veg_mask | urban_mask) & ~anomaly_mask)
        if len(valid_locations) > 0:
            center = valid_locations[np.random.randint(len(valid_locations))]
            size = np.random.randint(2, 4)
            y, x = center
            y_slice = slice(max(0, y-size), min(grid_size, y+size))
            x_slice = slice(max(0, x-size), min(grid_size, x+size))
            
            # Extreme thermal signature
            tir1[y_slice, x_slice] = 0.9 + 0.1 * np.random.random()
            tir2[y_slice, x_slice] = 0.85 + 0.1 * np.random.random()
            
            # Smoke reduces visible reflectance
            red[y_slice, x_slice] *= 0.5
            green[y_slice, x_slice] *= 0.5
            blue[y_slice, x_slice] *= 0.5
            
            # Destroyed vegetation
            nir[y_slice, x_slice] = 0.1
            ndvi[y_slice, x_slice] = -0.3
            
            anomaly_mask[y_slice, x_slice] = True
            anomaly_types[y_slice, x_slice] = 3
    
    # Anomaly 4: Flooding (2-3 instances)
    n_floods = np.random.randint(2, 4)
    for _ in range(n_floods):
        valid_locations = np.argwhere((veg_mask | urban_mask) & ~anomaly_mask)
        if len(valid_locations) > 0:
            center = valid_locations[np.random.randint(len(valid_locations))]
            size = np.random.randint(4, 8)
            y, x = center
            y_slice = slice(max(0, y-size), min(grid_size, y+size))
            x_slice = slice(max(0, x-size), min(grid_size, x+size))
            
            # Water signature in areas that should be dry
            red[y_slice, x_slice] = 0.1
            green[y_slice, x_slice] = 0.2
            blue[y_slice, x_slice] = 0.35
            nir[y_slice, x_slice] = 0.05
            swir1[y_slice, x_slice] = 0.02
            swir2[y_slice, x_slice] = 0.02
            ndwi[y_slice, x_slice] = 0.5  # High water content
            ndvi[y_slice, x_slice] = -0.4
            
            anomaly_mask[y_slice, x_slice] = True
            anomaly_types[y_slice, x_slice] = 4
    
    # Anomaly 5: Vehicle Concentrations (3-6 instances)
    n_vehicles = np.random.randint(3, 7)
    for _ in range(n_vehicles):
        valid_locations = np.argwhere(urban_mask & ~anomaly_mask)
        if len(valid_locations) > 0:
            center = valid_locations[np.random.randint(len(valid_locations))]
            # Small cluster of 1-2 pixel vehicles
            n_veh = np.random.randint(3, 8)
            for _ in range(n_veh):
                offset_y = np.random.randint(-3, 4)
                offset_x = np.random.randint(-3, 4)
                vy, vx = center[0] + offset_y, center[1] + offset_x
                if 0 <= vy < grid_size and 0 <= vx < grid_size:
                    # Bright in visible, moderate thermal
                    red[vy, vx] = 0.7
                    green[vy, vx] = 0.65
                    blue[vy, vx] = 0.6
                    tir1[vy, vx] += 0.2
                    tir2[vy, vx] += 0.2
                    
                    anomaly_mask[vy, vx] = True
                    anomaly_types[vy, vx] = 5
    
    # === STEP 5: Add realistic noise and atmospheric effects ===
    
    # Sensor noise (varies by band)
    red += noise_level * np.random.randn(grid_size, grid_size)
    green += noise_level * np.random.randn(grid_size, grid_size)
    blue += noise_level * np.random.randn(grid_size, grid_size)
    nir += noise_level * 0.8 * np.random.randn(grid_size, grid_size)
    swir1 += noise_level * 1.2 * np.random.randn(grid_size, grid_size)
    swir2 += noise_level * 1.2 * np.random.randn(grid_size, grid_size)
    tir1 += noise_level * 0.5 * np.random.randn(grid_size, grid_size)
    tir2 += noise_level * 0.5 * np.random.randn(grid_size, grid_size)
    
    # Atmospheric haze (affects visible more than IR)
    haze_pattern = generate_perlin_like((grid_size, grid_size), scale=20) * 0.1
    red += haze_pattern
    green += haze_pattern * 0.8
    blue += haze_pattern * 0.6
    
    # Clip all values to valid range
    red = np.clip(red, 0, 1)
    green = np.clip(green, 0, 1)
    blue = np.clip(blue, 0, 1)
    nir = np.clip(nir, 0, 1)
    swir1 = np.clip(swir1, 0, 1)
    swir2 = np.clip(swir2, 0, 1)
    tir1 = np.clip(tir1, 0, 1)
    tir2 = np.clip(tir2, 0, 1)
    ndvi = np.clip(ndvi, -1, 1)
    ndwi = np.clip(ndwi, -1, 1)
    
    # === STEP 6: Flatten to dataframe ===
    
    x_coords = np.repeat(np.arange(grid_size), grid_size)
    y_coords = np.tile(np.arange(grid_size), grid_size)
    
    df = pd.DataFrame({
        'x': x_coords,
        'y': y_coords,
        'red': red.flatten(),
        'green': green.flatten(),
        'blue': blue.flatten(),
        'nir': nir.flatten(),
        'swir1': swir1.flatten(),
        'swir2': swir2.flatten(),
        'tir1': tir1.flatten(),
        'tir2': tir2.flatten(),
        'ndvi': ndvi.flatten(),
        'ndwi': ndwi.flatten(),
        'edge_magnitude': edge_magnitude.flatten(),
        'texture_variance': texture_variance.flatten(),
        'anomaly': anomaly_mask.flatten().astype(int),
        'anomaly_type': anomaly_types.flatten()
    })
    
    # Create feature matrix
    feature_names = ['red', 'green', 'blue', 'nir', 'swir1', 'swir2', 
                     'tir1', 'tir2', 'ndvi', 'ndwi', 'edge_magnitude', 'texture_variance']
    X = df[feature_names].values
    y = df['anomaly'].values
    
    return df, X, y, feature_names

# Test the generator
if __name__ == "__main__":
    df, X, y, feature_names = generate_realistic_satellite_data(grid_size=64)
    print(f"Generated {len(df)} pixels")
    print(f"Features: {feature_names}")
    print(f"Anomalies: {y.sum()} ({100*y.sum()/len(y):.1f}%)")
    print(f"\nAnomaly type breakdown:")
    for i in range(1, 6):
        count = (df['anomaly_type'] == i).sum()
        types = ['Construction', 'Camouflage', 'Fire', 'Flood', 'Vehicles']
        print(f"  {types[i-1]}: {count} pixels")
