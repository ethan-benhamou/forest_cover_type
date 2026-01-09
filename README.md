# Forest Cover Type Prediction

A machine learning solution for the [Kaggle Forest Cover Type Prediction](https://www.kaggle.com/c/forest-cover-type-prediction) competition, achieving competitive scores through advanced feature engineering and ensemble methods.

## Competition Overview

The goal is to predict the forest cover type (7 categories) for 30m x 30m cells in Roosevelt National Forest, Colorado, using only cartographic variables—no remotely sensed data.

**Cover Types:**
1. Spruce/Fir
2. Lodgepole Pine
3. Ponderosa Pine
4. Cottonwood/Willow
5. Aspen
6. Douglas-fir
7. Krummholz

## Solution Approach

### Feature Engineering

| Feature Category | Description |
|------------------|-------------|
| **Distance Metrics** | Euclidean & Manhattan distances to hydrology |
| **Distance Ratios** | Hydrology/Road, Hydrology/Fire, Road/Fire ratios |
| **Elevation Interactions** | Combined elevation with vertical hydrology distance |
| **Hillshade Statistics** | Mean and range across time points (9am, noon, 3pm) |
| **Aspect Decomposition** | North-South and East-West components (cos/sin transform) |
| **Categorical IDs** | Wilderness Area ID, Soil Type ID from one-hot columns |
| **Domain Features** | Climate zone, stony soil indicator |

### Target Encoding

Cross-validation target encoding with smoothing to prevent data leakage:
- Encoding computed only on training folds
- Test set receives averaged encodings across all folds
- Smoothing blends category mean with global mean for rare categories

### Models

Three gradient boosting models with optimized hyperparameters:

| Model | Key Strengths |
|-------|---------------|
| **LightGBM** | Fast training, efficient memory usage |
| **XGBoost** | Robust regularization, histogram-based |
| **CatBoost** | Excellent categorical handling, symmetric trees |

### Ensemble Strategies

1. **Stacking**: Logistic Regression meta-learner trained on out-of-fold predictions
2. **Blending**: Weighted average (LightGBM: 45%, CatBoost: 35%, XGBoost: 20%)

## Project Structure

```
├── forest_cover_type_ensemble.ipynb  # Main notebook (run this)
├── stacking.py                       # Original Python script
├── train.csv                         # Training data
├── test-full.csv                     # Test data
├── full_submission.csv               # Sample submission format
├── submission_stacking_*.csv         # Stacking predictions
├── submission_blending_*.csv         # Blending predictions
└── README.md
```

## Requirements

```
numpy
pandas
scikit-learn
lightgbm
xgboost
catboost
tqdm
```

### Installation

```bash
pip install numpy pandas scikit-learn lightgbm xgboost catboost tqdm
```

## Usage

### Option 1: Jupyter Notebook (Recommended)

```bash
jupyter notebook forest_cover_type_ensemble.ipynb
```

Run cells sequentially. The notebook includes detailed explanations for each step.

### Option 2: Python Script

```bash
python stacking.py
```

## Configuration

Key parameters in the notebook/script:

```python
N_FOLDS = 7      # Cross-validation folds
SEED = 42        # Random seed for reproducibility
```

Model hyperparameters are pre-tuned for optimal performance. Training takes approximately 15-30 minutes depending on hardware.

## Results

The solution generates two submission files:

| Method | Description |
|--------|-------------|
| `submission_stacking_*.csv` | Meta-learner ensemble |
| `submission_blending_*.csv` | Weighted average ensemble |

Submit both to Kaggle to compare leaderboard performance.

## Potential Improvements

- Hyperparameter optimization with Optuna/GridSearch
- Additional models (ExtraTrees, Neural Networks)
- Feature selection using permutation importance
- Pseudo-labeling with high-confidence test predictions
- More aggressive feature interactions

## References

- [Kaggle Competition Page](https://www.kaggle.com/c/forest-cover-type-prediction)
- [UCI ML Repository - Covertype Dataset](https://archive.ics.uci.edu/ml/datasets/covertype)

## License

This project is for educational purposes as part of HEC coursework.
