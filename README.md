# Weather Forecasting with Machine Learning

A temperature prediction project using ensemble weather forecasts and ground-based observations to build and compare machine learning models for improved local weather forecasting accuracy.

## Project Overview

Weather prediction accuracy is critical for agriculture, energy management, and daily planning. This project tackles the challenge of improving local temperature forecasts by combining ensemble numerical weather prediction (NWP) data with ground observations. I developed and compared multiple machine learning approaches—Linear Regression, Random Forest, and Neural Networks—to post-process ensemble forecasts and reduce prediction errors.

**Context**: Independent research project (2024)  
**My Role**: Principal investigator and implementer

## My Role & Contributions

- **Data Integration**: Built pipeline to merge ensemble forecast data (88,450+ ensemble members) with ground truth observations from VLINDER network
- **Feature Engineering**: Converted temperature units (Kelvin to Celsius), handled missing data, performed temporal alignment of forecast-observation pairs
- **Model Development**: Implemented and tuned three ML approaches using scikit-learn and TensorFlow
- **Feature Selection**: Applied correlation analysis, Recursive Feature Elimination with Cross-Validation (RFECV), and domain knowledge to identify optimal predictors
- **Validation Strategy**: Designed 60/20/20 train/validation/test split with proper temporal ordering
- **Performance Analysis**: Conducted comprehensive model comparison using MSE metrics and prediction visualization

## Data

- **Ensemble Forecasts**: 58 CSV files containing ~88,450 forecast ensemble members from February-July 2023
  - Source: ECMWF-style numerical weather prediction ensemble
  - Features: 26 meteorological variables (temperature, pressure, wind, humidity, radiation, soil properties)
  - Temporal resolution: 6-hourly forecasts
  - Geographic coverage: Single location (50.75°N, 4.25°E)

- **Ground Truth**: VLINDER observation network data (vlinder19_2023.csv)
  - High-quality temperature measurements for model training/validation
  - Temporal alignment with forecast valid times

- **Final Dataset**: 14,934 matched forecast-observation pairs after preprocessing
- **Key preprocessing**: Temperature unit conversion, missing value removal, temporal sorting

## Methods & Models

### Linear Regression
- **Baseline approach** with full feature set (26 variables)
- **Feature selection variants**:
  - Top 10 features by correlation with temperature: `['t2m', 'mx2t6', 'skt', 'st', 'mn2t6', 'sm', 'd2m', 'tcw', 'sshf', 'ssr']`
  - RFECV selected features: `['t2m', 'sf', 'st', 'sd']`
  - Coefficient-based top 10: `['t2m', 'st', 'sd', 'v10', 'u10', 'sp', 'msl', 'mx2t6', 'd2m', 'sf']`

### Random Forest
- **Hyperparameter tuning** via GridSearchCV (5-fold CV)
- **Parameter grid**: n_estimators (50,100,200), max_depth (10,20,30,None), min_samples_split (2,5,10), min_samples_leaf (1,2,4), bootstrap (True/False)
- **Best configuration**: 200 estimators, max_depth=10, min_samples_leaf=4, min_samples_split=10

### Neural Networks (Feedforward)
- **Architecture**: Input → Dense(64, ReLU) → Dense(32, ReLU) → Output(1)
- **Training**: Adam optimizer, MSE loss, 50 epochs, batch_size=32
- **Feature standardization**: StandardScaler for neural network inputs
- **Hyperparameter optimization**: RandomizedSearchCV over neurons (32,64,128), batch_size (16,32,64), epochs (50,100), optimizers (adam, rmsprop)

## Experiments & Results

| Model | Features | Validation MSE | Best Configuration |
|-------|----------|----------------|-------------------|
| **Linear Regression** | Top 10 (correlation) | **7.68** | Standard linear model |
| Linear Regression | Top 10 (coefficients) | 7.94 | Standard linear model |
| Linear Regression | All features | 8.45 | Standard linear model |
| Linear Regression | RFECV selected | 9.49 | 4 features selected |
| Random Forest | All features | 39.13 | 50 estimators, depth=10 |
| Random Forest | Top 10 | 46.33 | 200 estimators, depth=10 |
| Neural Network | Top 10 | 55.42 | 64-32 architecture |
| Neural Network | All features | 40.73 | 64-32 architecture |

**Best Model**: Linear Regression with top 10 correlation-selected features  
**Test MSE**: 5.97 (best model performance on held-out test set)

### Key Findings
- **Linear models excel** for this temperature forecasting task, significantly outperforming ensemble methods
- **Feature selection critical**: Correlation-based selection outperformed automated methods (RFECV)
- **Most important predictors**: t2m (2m temperature), mx2t6 (6h max temperature), skt (skin temperature), st (soil temperature)
- **Ensemble forecasts showed high variance** in Random Forest and Neural Network approaches, potentially due to limited training data relative to ensemble size

## How to Run

### Prerequisites
- Python 3.7.16
- Conda/Mamba package manager

### Setup
```bash
# Create environment from requirements
conda env create -f requirment.yml
conda activate weather-notebook

# Verify key packages installed
python -c "import pandas, numpy, sklearn, tensorflow, matplotlib, seaborn; print('Environment ready')"
```

### Data Preparation
```bash
# Ensure data structure:
mkdir -p Forecast_2023/
# Place 58 forecast CSV files in Forecast_2023/
# Place vlinder19_2023.csv in project root (referenced in notebook but missing from repo)
```

### Running the Analysis
```bash
# Launch Jupyter notebook
jupyter notebook weather-notebook.ipynb

# Or run all cells programmatically
jupyter nbconvert --to notebook --execute weather-notebook.ipynb
```

### Reproduce Best Model
The optimal Linear Regression model uses correlation-selected top 10 features. Key code from `weather-notebook.ipynb:cell-19`:

```python
# Select best features
top_10_features = correlation_with_temp.abs().nlargest(10).index.tolist()
selected_train_features = trainset[top_10_features]

# Train optimal model
model = LinearRegression()
model.fit(selected_train_features, train_target)
```

## Project Structure

```
├── README.md                    # This documentation
├── requirment.yml              # Conda environment specification
├── weather-notebook.ipynb      # Main analysis notebook (all ML experiments)
├── notebook_pdf.pdf           # Static PDF export of notebook results
├── Forecast_2023/             # Directory containing 58 ensemble forecast CSV files
├── Forecast_27_04_2024.csv    # April 2024 forecast data for operational prediction demo
└── .git/                      # Git repository
```

**Key Files**:
- `weather-notebook.ipynb`: Complete implementation with data loading, preprocessing, model training, and evaluation
- `Forecast_2023/*.csv`: Training/validation ensemble forecast data (58 files, ~22MB total)
- TODO: `vlinder19_2023.csv` - Ground truth observations (missing from repo)

## Decisions, Trade-offs & Challenges

### Technical Challenges
- **Missing ground truth data**: VLINDER observations file not in repository, limiting reproducibility
- **Ensemble size vs. training data**: 88K+ ensemble members but only ~15K matched observations created potential overfitting
- **Temporal alignment**: Ensuring proper matching between 6-hourly forecasts and observation timestamps
- **Feature scaling**: Neural networks required standardization while tree methods worked with raw features

### Model Selection Rationale
- **Linear Regression chosen** over complex models due to superior performance and interpretability
- **Feature selection approach**: Correlation-based selection outperformed automated RFECV, suggesting domain knowledge value
- **Avoided ensemble methods**: Random Forest performed poorly, likely due to high variance in ensemble forecast features

### Alternative Approaches Considered
- **Time series methods**: Could incorporate temporal dependencies not captured in current approach
- **Ensemble post-processing**: Statistical methods like Model Output Statistics (MOS) or Ensemble Model Output Statistics (EMOS)
- **Deep learning**: LSTM/GRU for sequential modeling, but limited data size made this impractical

## Ethics & Responsible AI

### Bias Considerations
- **Geographic bias**: Model trained on single location (Belgium), may not generalize to other climates/regions
- **Temporal bias**: Training data from Feb-July 2023 may not capture full seasonal variability
- **Observational bias**: VLINDER network quality and siting may introduce systematic errors

### Mitigations Implemented
- **Proper validation split**: Temporal ordering maintained to avoid data leakage
- **Feature interpretation**: Clear documentation of most important predictors for model transparency
- **Performance reporting**: Honest assessment of model limitations and failure modes

## Future Work

1. **Expand temporal coverage**: Include full annual cycle (winter months) for seasonal robustness
2. **Multi-location validation**: Test model performance across different geographic regions and climate zones
3. **Probabilistic forecasting**: Develop uncertainty quantification using ensemble spread information
4. **Operational deployment**: Implement real-time forecast processing pipeline with automated model updates
5. **Advanced architectures**: Explore transformer models for spatial-temporal weather pattern recognition

## Acknowledgments

**Data Sources**: 
- Ensemble forecasts: ECMWF-style numerical weather prediction
- Ground observations: VLINDER atmospheric measurement network

**Technical Stack**: scikit-learn, TensorFlow/Keras, pandas, numpy, matplotlib, seaborn


---

*This project demonstrates practical machine learning application to meteorological forecasting, emphasizing model selection based on performance over complexity and the importance of domain-informed feature engineering.*