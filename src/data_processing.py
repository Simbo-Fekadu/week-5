import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class DataCleaner(BaseEstimator, TransformerMixin):
    """Modernized without in-place operations"""
    def __init__(self, config):
        self.config = config
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()

        # Cap Amount column
        X['Amount'] = X['Amount'].clip(upper=self.config['outlier_caps']['Amount'])
        
        # Fill missing Amount values with median after capping
        X['Amount'] = X['Amount'].fillna(X['Amount'].median())
        
        # Fill missing ProductCategory with 'missing'
        X['ProductCategory'] = X['ProductCategory'].fillna('missing')
        
        return X

class RFMTransformer(BaseEstimator, TransformerMixin):
    """Simplified to work with test data"""
    def __init__(self, snapshot_date='2019-12-31'):
        self.snapshot_date = pd.to_datetime(snapshot_date)
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        X['TransactionDate'] = pd.to_datetime(X['TransactionStartTime'])
        
        rfm = X.groupby('CustomerId').agg({
            'TransactionDate': lambda x: (self.snapshot_date - x.max()).days,
            'Amount': ['count', 'sum', 'mean']
        })
        
        rfm.columns = ['Recency', 'Frequency', 'Monetary', 'AvgAmount']
        return rfm.reset_index()

def build_pipeline(config_path="configs/config.yml"):
    """Load config and build pipeline"""
    with open(Path(__file__).parent.parent / config_path) as f:
        config = yaml.safe_load(f)
    
    return Pipeline([
        ('cleaner', DataCleaner(config)),
        ('rfm', RFMTransformer(config.get('rfm', {}).get('snapshot_date', '2019-12-31')))
    ])
