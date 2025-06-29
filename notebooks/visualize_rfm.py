import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add project root to path (Jupyter-safe)
sys.path.append(str(Path().resolve().parent.parent))

from src.data_processing import build_pipeline

def visualize_clusters(data_path):
    """Visualize RFM clusters"""
    # Load data
    data = pd.read_csv(data_path)

    # Convert all datetime columns to pandas datetime and make them tz-naive
    for col in data.columns:
        if pd.api.types.is_object_dtype(data[col]) or pd.api.types.is_datetime64_any_dtype(data[col]):
            try:
                data[col] = pd.to_datetime(data[col], errors='ignore')
            except Exception:
                continue
            # If column is tz-aware, make it tz-naive
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                if hasattr(data[col].dt, 'tz') and data[col].dt.tz is not None:
                    data[col] = data[col].dt.tz_localize(None)

    # Process data
    pipeline = build_pipeline()
    processed = pipeline.fit_transform(data)
    
    # Create visualization
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        processed['Recency'],
        processed['Frequency'], 
        processed['Monetary'],
        c=processed['is_high_risk'],
        cmap='viridis',
        s=50,
        alpha=0.6
    )
    
    ax.set_xlabel('Recency (days)')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Monetary (UGX)')
    plt.title('Customer Segments by RFM')
    plt.colorbar(scatter, label='Risk Level')
    
    # Save figure
    fig_dir = Path('reports/figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / 'rfm_clusters.png', dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {fig_dir/'rfm_clusters.png'}")

if __name__ == "__main__":
    visualize_clusters("./data/raw/data.csv")