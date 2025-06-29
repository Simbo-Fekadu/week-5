import pytest
import pandas as pd
import numpy as np
from src.data_processing import DataCleaner, RFMTransformer

@pytest.fixture
def sample_config():
    return {
        'outlier_caps': {'Amount': 1000000},
        'rfm': {'snapshot_date': '2019-12-31'}
    }

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
