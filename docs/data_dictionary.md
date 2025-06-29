## Engineered Features Documentation

### RFM Features

| Feature   | Calculation                                    | Description                     |
| --------- | ---------------------------------------------- | ------------------------------- |
| Recency   | `(snapshot_date - last_transaction_date).days` | Days since last activity        |
| Frequency | `COUNT(transactions)`                          | Total transactions per customer |
| Monetary  | `SUM(Amount)`                                  | Total spend per customer        |

### Data Quality Rules

1. **Outliers**:
   - Amounts > 1M UGX capped at 1M (99th percentile)
2. **Missing Values**:
   - `ProductCategory`: Filled with 'missing'
   - `Amount`: Filled with median
