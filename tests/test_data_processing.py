import pytest
import pandas as pd
import numpy as np
from src.data_processing import DataCleaner, RFMTransformer
from src.data_processing import build_pipeline, RiskLabelGenerator
@pytest.fixture
def sample_config():
    return {
        'outlier_caps': {'Amount': 1000000},
        'rfm': {'snapshot_date': '2019-12-31'}
    }
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'CustomerId': [1, 1, 2, 3],
        'Amount': [100, 200, 50, 10000],
        'TransactionStartTime': ['2019-12-01', '2019-12-15', '2019-11-01', '2019-10-01'],
        'ProductCategory': ['A', 'B', 'A', 'C']
    })

def test_full_pipeline(sample_data):
    """Test the complete pipeline without visualizations"""
    from src.data_processing import build_pipeline
    
    pipeline = build_pipeline()
    processed = pipeline.fit_transform(sample_data)
    
    # Verify key outputs
    assert 'is_high_risk' in processed.columns
    assert set(processed['is_high_risk'].unique()).issubset({0, 1})
    assert all(col in processed.columns for col in ['Recency', 'Frequency', 'Monetary'])
def test_data_cleaner(sample_config):
    test_data = pd.DataFrame({
        'Amount': [100, np.nan, 2000000],
        'ProductCategory': ['A', None, 'B']
    })
    cleaned = DataCleaner(sample_config).transform(test_data)
    
    assert not cleaned['Amount'].isna().any()
    assert cleaned['Amount'].max() == 1000000
    assert (cleaned['ProductCategory'] == 'missing').sum() == 1

def test_rfm_transformer(sample_config):
    test_data = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'Amount': [100, 200, 50],
        'TransactionStartTime': ['2019-12-01', '2019-12-15', '2019-11-01']
    })
    rfm = RFMTransformer(sample_config['rfm']['snapshot_date']).transform(test_data)
    
    assert rfm.loc[rfm['CustomerId'] == 1, 'Recency'].iloc[0] == 16
    assert rfm.loc[rfm['CustomerId'] == 1, 'Frequency'].iloc[0] == 2
    assert rfm.loc[rfm['CustomerId'] == 1, 'Monetary'].iloc[0] == 300
def test_risk_label_generator():
    test_data = pd.DataFrame({
        'Recency': [100, 10, 200],
        'Frequency': [1, 20, 2],
        'Monetary': [500, 5000, 300]
    })
    labeled = RiskLabelGenerator().fit_transform(test_data)
    assert 'is_high_risk' in labeled.columns
    assert set(labeled['is_high_risk'].unique()).issubset({0, 1})
pipeline = build_pipeline()

# Example: Load or define raw_data before using it
# Replace this with your actual data loading logic
raw_data = pd.DataFrame({
    'CustomerId': [1, 1, 2, 3],
    'Amount': [100, 200, 50, 400],
    'TransactionStartTime': ['2019-12-01', '2019-12-15', '2019-11-01', '2019-10-10'],
    'ProductCategory': ['A', 'A', 'B', 'C']
})

df_processed = pipeline.fit_transform(raw_data)
print(df_processed['is_high_risk'].value_counts())

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_processed['Recency'], df_processed['Frequency'], df_processed['Monetary'], 
           c=df_processed['is_high_risk'], cmap='viridis')
plt.savefig('reports/figures/rfm_clusters.png')

def test_model_training():
    """Test model training pipeline"""
    from src.train import load_data
    data = load_data()
    assert 'is_high_risk' in data.columns
    assert not data.isnull().any().any()