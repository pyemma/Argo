# Label Leakage

Label leakage occurs when information that wouldn't be available at prediction time is accidentally included in training data, leading to artificially high model performance that doesn't generalize to real-world scenarios.

## Example from MovieLens Dataset

In our [MovieLens dataset preparation](../data/dataset.py), we prevent label leakage by carefully selecting which user ratings to use for training:

```python
df = (
    ratings.groupby("userId")
    .apply(lambda x: x.sort_values("timestamp").iloc[history_seq_length:])  # use sequence that is not part of the sequence feature to prevent label leakage
    .reset_index(drop=True)
)
```

This code ensures we only use ratings that come *after* the sequence features used for prediction, maintaining temporal integrity.

## Common Types of Label Leakage

1. **Temporal leakage**: Using future information to predict past events
2. **Target leakage**: Including variables that contain target information
3. **Train-test contamination**: Test data influencing training process

## Prevention Strategies

- **Respect time order**: For time-series data, ensure training features precede prediction targets
- **Feature engineering caution**: Avoid features that implicitly contain target information
- **Proper data splitting**: Split data before any preprocessing or feature engineering
- **Cross-validation**: Use time-based splits for sequential data
- **Data isolation**: Keep test data completely separate until final evaluation

Careful data preparation and strict separation between features and targets are essential to building models that generalize well to new, unseen data. 