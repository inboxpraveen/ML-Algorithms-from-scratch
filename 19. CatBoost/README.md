# CatBoost - Algorithm #19

## Overview

CatBoost (Categorical Boosting) is a gradient boosting framework developed by Yandex that excels at handling categorical features and uses symmetric (oblivious) trees for better generalization and robustness.

## Files in this Directory

- **`_19_catboost.py`** - Complete implementation of CatBoost with 9 comprehensive usage examples
- **`_19_catboost.md`** - Detailed guide covering theory, math, implementation, and applications

## Key Features

### What Makes CatBoost Special?

1. **Symmetric (Oblivious) Trees**
   - All nodes at the same level split on the same feature and threshold
   - Natural regularization and faster prediction
   - Creates 2^depth leaves with balanced structure

2. **Ordered Boosting**
   - Prevents prediction shift and target leakage
   - More robust to overfitting
   - Each sample's gradient computed using models trained on other samples

3. **Native Categorical Handling**
   - Automatically handles categorical features (in full CatBoost library)
   - No need for manual one-hot encoding
   - Uses ordered target statistics to prevent leakage

4. **Strong Default Parameters**
   - Works well out-of-the-box
   - Lower learning rate (0.03) with higher regularization (3.0)
   - Less sensitive to hyperparameter tuning

## Quick Start

```python
import numpy as np
from _19_catboost import CatBoost

# Generate sample data
X = np.random.randn(200, 5)
y = 2 * X[:, 0] - 3 * X[:, 1] + X[:, 2] + np.random.randn(200) * 0.5

# Train-test split
X_train, X_test = X[:150], X[50:]
y_train, y_test = y[:150], y[50:]

# Create and train model
model = CatBoost(
    n_estimators=100,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0
)
model.fit(X_train, y_train)

# Evaluate
train_rmse = -model.score(X_train, y_train)
test_rmse = -model.score(X_test, y_test)
print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

# Make predictions
predictions = model.predict(X_test)
```

## Usage Examples Included

The `_19_catboost.py` file includes 9 detailed examples:

1. **Simple Regression** - Basic usage with non-linear data
2. **Binary Classification** - Credit risk, customer churn
3. **Early Stopping** - Prevent overfitting with validation
4. **Feature Importance** - Understand which features matter
5. **Tree Depth Comparison** - Effect of model complexity
6. **Learning Rate Effects** - Trade-off between speed and accuracy
7. **L2 Regularization** - Control overfitting strength
8. **Customer Churn Prediction** - Real-world classification example
9. **Configuration Comparison** - Fast vs Accurate vs Regularized

## When to Use CatBoost

✅ **Best for:**
- Datasets with many categorical features
- When you want good results without extensive tuning
- When robustness and generalization are priorities
- Production systems requiring stable models
- Small to medium-sized datasets

❌ **Consider alternatives:**
- Very large datasets (>1M samples) → Use LightGBM for speed
- Pure numerical features with speed priority → Use LightGBM
- When you need industry-standard ecosystem → Use XGBoost

## Key Parameters

```python
CatBoost(
    n_estimators=100,      # Number of trees
    learning_rate=0.03,    # Shrinkage (conservative default)
    depth=6,               # Tree depth (2^6 = 64 leaves)
    l2_leaf_reg=3.0,       # L2 regularization (strong default)
    border_count=128,      # Feature quantization bins
    objective='regression' # or 'binary' for classification
)
```

## Comparison with Other Boosting Algorithms

| Feature | XGBoost | LightGBM | **CatBoost** |
|---------|---------|----------|--------------|
| Tree Structure | Asymmetric | Asymmetric | **Symmetric** |
| Default LR | 0.3 | 0.1 | **0.03** |
| Categorical Support | Manual | Manual | **Native** |
| Overfitting Risk | Medium | Higher | **Lower** |
| Default Performance | Good | Good | **Best** |
| Best Use Case | General | Large data | **Categoricals** |

## Mathematical Foundation

### Symmetric Tree Structure
```
All nodes at the same level use the SAME split:

                [Feature 2 <= 5]
               /                \
        [Feature 0 <= 3]    [Feature 0 <= 3]
         /        \          /        \
      Leaf 0   Leaf 1    Leaf 2   Leaf 3
```

### Leaf Value Calculation
```
value = -sum(gradients) / (count + l2_leaf_reg)

- Stronger regularization for smaller leaves
- Shrinks values toward zero
- Prevents overfitting to small groups
```

### Fast Prediction
```
O(depth) instead of O(depth × branches)

Using binary indexing:
1. Start with leaf_index = 0
2. For each level: if goes right, add 2^(remaining_depth)
3. Return leaf_value[leaf_index]
```

## Performance Tips

**If underfitting:**
```python
model = CatBoost(
    n_estimators=200,    # More trees
    depth=8,             # Deeper trees
    l2_leaf_reg=1.0      # Less regularization
)
```

**If overfitting:**
```python
model = CatBoost(
    depth=4,             # Shallower trees
    l2_leaf_reg=10.0,    # More regularization
    random_strength=2.0  # More randomness
)
```

## Learn More

- Read the comprehensive guide: `_19_catboost.md`
- Study the implementation: `_19_catboost.py`
- Run the 9 usage examples included in the Python file
- Official CatBoost: https://catboost.ai/

## Implementation Notes

This educational implementation includes:
- ✅ Symmetric (oblivious) tree structure
- ✅ Feature quantization (histogram-based)
- ✅ L2 regularization in leaf values
- ✅ Binary classification and regression
- ✅ Early stopping
- ✅ Feature importance
- ✅ Fast prediction with binary indexing

Not included (available in full CatBoost library):
- Native categorical feature handling
- Full ordered boosting implementation
- GPU acceleration
- Distributed training
- Advanced loss functions

---

**Next Algorithm:** Isolation Forest (Coming Soon)

**Previous Algorithm:** [LightGBM](../18.%20LightGBM/_18_lightgbm.md)
