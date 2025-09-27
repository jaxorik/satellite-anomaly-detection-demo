import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_realistic_satellite_data(df, title, predictions=None, ax=None, show_type=False):
    """
    Visualize realistic satellite data as a grid with anomalies highlighted.
    
    Parameters:
    - df: DataFrame containing x, y coordinates and anomaly labels
    - title: Plot title
    - predictions: Optional array of predicted anomalies
    - ax: Optional matplotlib axis to plot on
    - show_type: If True, color by anomaly type instead of prediction accuracy
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    grid_size_x = df['x'].max() + 1
    grid_size_y = df['y'].max() + 1
    
    if show_type and 'anomaly_type' in df.columns:
        # Show anomaly types with distinct colors
        grid = np.zeros((grid_size_y, grid_size_x, 3))
        type_colors = {
            0: [0.2, 0.2, 0.2],      # Normal - dark gray
            1: [1.0, 0.5, 0.0],       # Construction - orange
            2: [0.6, 0.0, 0.6],       # Camouflage - purple
            3: [1.0, 0.0, 0.0],       # Fire - red
            4: [0.0, 0.4, 0.8],       # Flood - blue
            5: [1.0, 1.0, 0.0]        # Vehicles - yellow
        }
        
        for idx, row in df.iterrows():
            x, y = int(row['x']), int(row['y'])
            if 0 <= x < grid_size_x and 0 <= y < grid_size_y:
                anomaly_type = int(row['anomaly_type'])
                grid[y, x] = type_colors.get(anomaly_type, [0.5, 0.5, 0.5])
        
        ax.imshow(grid, interpolation='nearest')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=type_colors[0], label='Normal'),
            Patch(facecolor=type_colors[1], label='Construction'),
            Patch(facecolor=type_colors[2], label='Camouflage'),
            Patch(facecolor=type_colors[3], label='Fire'),
            Patch(facecolor=type_colors[4], label='Flood'),
            Patch(facecolor=type_colors[5], label='Vehicles')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        
    elif predictions is None:
        # Just show ground truth (anomaly vs normal)
        grid = np.zeros((grid_size, grid_size, 3))
        for idx, row in df.iterrows():
            x, y = int(row['x']), int(row['y'])
            if 0 <= x < grid_size_x and 0 <= y < grid_size_y:
                if row['anomaly'] == 1:
                    grid[y, x] = [1, 0, 0]  # Red for anomalies
                else:
                    grid[y, x] = [0.8, 0.8, 0.8]  # Light gray for normal
        
        ax.imshow(grid, interpolation='nearest')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        
    else:
        # Show predictions vs ground truth
        grid = np.zeros((grid_size_y, grid_size_x, 3))
        for idx, row in df.iterrows():
            x, y = int(row['x']), int(row['y'])
            if 0 <= x < grid_size_x and 0 <= y < grid_size_y:
                true_label = row['anomaly']
                pred_label = predictions[idx]
                
                if true_label == 1 and pred_label == 1:
                    grid[y, x] = [0, 0.8, 0]  # Green for true positives
                elif true_label == 0 and pred_label == 1:
                    grid[y, x] = [1, 0.8, 0]  # Yellow for false positives
                elif true_label == 1 and pred_label == 0:
                    grid[y, x] = [0.8, 0, 0.8]  # Purple for false negatives
                else:
                    grid[y, x] = [0, 0.6, 0.8]  # Blue for true negatives
        
        ax.imshow(grid, interpolation='nearest')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    
    return ax


def plot_spectral_bands(df, sample_idx=None, figsize=(20, 10)):
    """
    Visualize individual spectral bands and computed indices.
    
    Parameters:
    - df: DataFrame with satellite data
    - sample_idx: Optional index to highlight a specific pixel
    - figsize: Figure size
    """
    # Get the original grid dimensions from the x,y coordinates
    grid_size_x = df['x'].max() + 1
    grid_size_y = df['y'].max() + 1
    
    bands = ['red', 'green', 'blue', 'nir', 'swir1', 'swir2', 
             'tir1', 'tir2', 'ndvi', 'ndwi', 'edge_magnitude', 'texture_variance']
    
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    axes = axes.flatten()
    
    for i, band in enumerate(bands):
        # Create empty grid and fill with available data
        data = np.full((grid_size_y, grid_size_x), np.nan)
        for idx, row in df.iterrows():
            x, y = int(row['x']), int(row['y'])
            data[y, x] = row[band]
        
        im = axes[i].imshow(data, cmap='viridis', interpolation='nearest')
        axes[i].set_title(band.upper(), fontsize=10)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        
        # Highlight specific pixel if requested
        if sample_idx is not None:
            x = df.loc[sample_idx, 'x']
            y = df.loc[sample_idx, 'y']
            axes[i].plot(x, y, 'r*', markersize=10)
        
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig


def plot_rgb_composite(df, figsize=(12, 10)):
    """
    Create RGB composite and false-color composites commonly used in remote sensing.
    
    Parameters:
    - df: DataFrame with satellite data
    - figsize: Figure size
    """
    grid_size_x = df['x'].max() + 1
    grid_size_y = df['y'].max() + 1
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Helper function to create grid from dataframe
    def create_band_grid(df, band_name):
        grid = np.full((grid_size_y, grid_size_x), np.nan)
        for idx, row in df.iterrows():
            x, y = int(row['x']), int(row['y'])
            grid[y, x] = row[band_name]
        return grid
    
    # True color (RGB)
    rgb = np.dstack([
        create_band_grid(df, 'red'),
        create_band_grid(df, 'green'),
        create_band_grid(df, 'blue')
    ])
    rgb = np.clip(rgb, 0, 1)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('True Color (RGB)')
    axes[0, 0].axis('off')
    
    # False color (NIR-R-G) - vegetation appears red
    false_color = np.dstack([
        create_band_grid(df, 'nir'),
        create_band_grid(df, 'red'),
        create_band_grid(df, 'green')
    ])
    false_color = np.clip(false_color, 0, 1)
    axes[0, 1].imshow(false_color)
    axes[0, 1].set_title('False Color (NIR-R-G)')
    axes[0, 1].axis('off')
    
    # SWIR composite (SWIR2-SWIR1-NIR) - good for geology and urban
    swir_composite = np.dstack([
        create_band_grid(df, 'swir2'),
        create_band_grid(df, 'swir1'),
        create_band_grid(df, 'nir')
    ])
    swir_composite = np.clip(swir_composite, 0, 1)
    axes[1, 0].imshow(swir_composite)
    axes[1, 0].set_title('SWIR Composite')
    axes[1, 0].axis('off')
    
    # Thermal
    thermal = create_band_grid(df, 'tir1')
    im = axes[1, 1].imshow(thermal, cmap='hot', interpolation='nearest')
    axes[1, 1].set_title('Thermal (TIR1)')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig


def plot_anomaly_detection_details(df, y_pred, y_test, figsize=(20, 12)):
    """
    Create comprehensive visualization of anomaly detection results.
    
    Parameters:
    - df: DataFrame with satellite data (test set)
    - y_pred: Predicted anomaly labels
    - y_test: True anomaly labels
    - figsize: Figure size
    """
    grid_size_x = df['x'].max() + 1
    grid_size_y = df['y'].max() + 1
    
    # Helper function to create grid from dataframe
    def create_band_grid(df, band_name):
        grid = np.full((grid_size_y, grid_size_x), np.nan)
        for idx, row in df.iterrows():
            x, y = int(row['x']), int(row['y'])
            grid[y, x] = row[band_name]
        return grid
    
    fig = plt.figure(figsize=figsize)
    
    # Create a 3x3 grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: RGB, Ground Truth, Predictions
    ax1 = fig.add_subplot(gs[0, 0])
    rgb = np.dstack([
        create_band_grid(df, 'red'),
        create_band_grid(df, 'green'),
        create_band_grid(df, 'blue')
    ])
    rgb = np.clip(rgb, 0, 1)
    ax1.imshow(rgb)
    ax1.set_title('RGB Composite')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_realistic_satellite_data(df, 'Ground Truth by Type', ax=ax2, show_type=True)
    
    ax3 = fig.add_subplot(gs[0, 2])
    plot_realistic_satellite_data(df, 'Model Predictions', predictions=y_pred, ax=ax3)
    
    # Row 2: Key spectral bands
    ax4 = fig.add_subplot(gs[1, 0])
    ndvi = create_band_grid(df, 'ndvi')
    im = ax4.imshow(ndvi, cmap='RdYlGn', interpolation='nearest', vmin=-1, vmax=1)
    ax4.set_title('NDVI (Vegetation Index)')
    ax4.axis('off')
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    
    ax5 = fig.add_subplot(gs[1, 1])
    thermal = create_band_grid(df, 'tir1')
    im = ax5.imshow(thermal, cmap='hot', interpolation='nearest')
    ax5.set_title('Thermal (TIR1)')
    ax5.axis('off')
    plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
    
    ax6 = fig.add_subplot(gs[1, 2])
    edges = create_band_grid(df, 'edge_magnitude')
    im = ax6.imshow(edges, cmap='gray', interpolation='nearest')
    ax6.set_title('Edge Magnitude')
    ax6.axis('off')
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
    
    # Row 3: Analysis plots
    ax7 = fig.add_subplot(gs[2, 0])
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax7)
    ax7.set_xlabel('Predicted')
    ax7.set_ylabel('Actual')
    ax7.set_title('Confusion Matrix')
    ax7.set_xticklabels(['Normal', 'Anomaly'])
    ax7.set_yticklabels(['Normal', 'Anomaly'])
    
    ax8 = fig.add_subplot(gs[2, 1])
    # Per-type detection rates
    if 'anomaly_type' in df.columns:
        type_names = ['Construction', 'Camouflage', 'Fire', 'Flood', 'Vehicles']
        detection_rates = []
        
        for i in range(1, 6):
            type_mask = df['anomaly_type'].values == i
            if type_mask.sum() > 0:
                detected = (y_pred[type_mask] == 1).sum()
                rate = detected / type_mask.sum()
                detection_rates.append(rate)
            else:
                detection_rates.append(0)
        
        ax8.bar(type_names, detection_rates, color='steelblue')
        ax8.set_ylabel('Detection Rate')
        ax8.set_title('Detection Rate by Anomaly Type')
        ax8.set_ylim(0, 1.1)
        ax8.tick_params(axis='x', rotation=45)
        
        for i, rate in enumerate(detection_rates):
            ax8.text(i, rate + 0.02, f'{rate:.2f}', ha='center', fontsize=9)
    
    ax9 = fig.add_subplot(gs[2, 2])
    # Class balance
    true_counts = [len(y_test) - y_test.sum(), y_test.sum()]
    pred_counts = [len(y_pred) - y_pred.sum(), y_pred.sum()]
    
    x = np.arange(2)
    width = 0.35
    ax9.bar(x - width/2, true_counts, width, label='Ground Truth', color='lightblue')
    ax9.bar(x + width/2, pred_counts, width, label='Predicted', color='coral')
    ax9.set_ylabel('Count')
    ax9.set_title('Class Distribution')
    ax9.set_xticks(x)
    ax9.set_xticklabels(['Normal', 'Anomaly'])
    ax9.legend()
    
    return fig
