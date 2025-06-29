import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
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
    """Timezone-safe RFM feature generator"""
    def __init__(self, snapshot_date='2019-12-31'):
        # Keep snapshot_date timezone-naive
        self.snapshot_date = pd.to_datetime(snapshot_date)
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        # Convert to timezone-naive datetime
        X['TransactionDate'] = pd.to_datetime(X['TransactionStartTime']).dt.tz_localize(None)
        
        rfm = X.groupby('CustomerId').agg({
            'TransactionDate': lambda x: (self.snapshot_date - x.max()).days,
            'Amount': ['count', 'sum', 'mean']
        })
        rfm.columns = ['Recency', 'Frequency', 'Monetary', 'AvgAmount']
        return rfm.reset_index()
    """Generates RFM features with timezone handling"""
    def __init__(self, snapshot_date='2019-12-31'):
        # Make snapshot_date timezone-aware (UTC)
        self.snapshot_date = pd.to_datetime(snapshot_date).tz_localize('UTC')
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        # Convert to timezone-aware datetime (UTC)
        X['TransactionDate'] = pd.to_datetime(X['TransactionStartTime']).dt.tz_localize('UTC')
        
        rfm = X.groupby('CustomerId').agg({
            'TransactionDate': lambda x: (self.snapshot_date - x.max()).days,
            'Amount': ['count', 'sum', 'mean']
        })
        rfm.columns = ['Recency', 'Frequency', 'Monetary', 'AvgAmount']
        return rfm.reset_index()
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
        ('rfm', RFMTransformer(config['rfm']['snapshot_date'])),
        ('risk_labeler', RiskLabelGenerator())
    ])
class RiskLabelGenerator(BaseEstimator, TransformerMixin):
    """Creates is_high_risk labels via RFM clustering"""
    def __init__(self, n_clusters=3, risk_cluster=0):
        self.n_clusters = n_clusters
        self.risk_cluster = risk_cluster
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    def fit(self, X, y=None):
        self.kmeans.fit(X[['Recency', 'Frequency', 'Monetary']])
        return self

    def transform(self, X):
        X['Cluster'] = self.kmeans.predict(X[['Recency', 'Frequency', 'Monetary']])
        X['is_high_risk'] = (X['Cluster'] == self.risk_cluster).astype(int)
        return X.drop('Cluster', axis=1)