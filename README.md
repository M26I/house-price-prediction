# House Price Prediction

## Overview
This project builds a machine learning model to predict house prices based on various features. It uses the **XGBoost** algorithm and evaluates performance using metrics like **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **R² score**. The dataset is preprocessed with feature engineering, outlier removal, and scaling.

## Project Structure
```
├── data/                # Raw dataset
├── notebooks/           # Jupyter Notebooks for exploration & evaluation
│   ├── 01_data_cleaning.ipynb
│   ├── 02_model_evaluation.ipynb
├── src/                 # Source code
│   ├── train.py         # Model training script
├── reports/             # Saved models and reports
│   ├── xgboost_model.pkl
├── .gitignore           # Ignored files
├── README.md            # Project documentation
```

## Dataset
- The dataset (`Housing.csv`) contains various features such as **area, bedrooms, location**, etc.
- Outliers were removed based on the **Interquartile Range (IQR)** method.
- Feature engineering added:
  - `price_per_sqft = price / (area + 1)`
  - `rooms_per_sqft = bedrooms / (area + 1)`

## Model Training
- The model is trained using **XGBoost** with **GridSearchCV** for hyperparameter tuning.
- Best hyperparameters:
  ```
  {'colsample_bytree': 1.0, 'gamma': 0, 'learning_rate': 0.05, 'max_depth': 4, 'min_child_weight': 1, 'n_estimators': 500, 'reg_alpha': 0, 'subsample': 0.8}
  ```
- Trained model is saved as `xgboost_model.pkl` in the `reports/` directory.

## Model Evaluation
| Metric  | Value |
|---------|-------------|
| MAE     | 162,685.57 |
| RMSE    | 228,625.48 |
| R²      | 0.98       |

- **Visualization:** Actual vs. Predicted prices, residuals distribution, and feature importance are analyzed in `02_model_evaluation.ipynb`.

## How to Run
1. **Set up the environment**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the model**
   ```bash
   python src/train.py
   ```
3. **Evaluate the model**
   Run `02_model_evaluation.ipynb` in Jupyter Notebook.

## Future Improvements
- Add more features (e.g., location-based metrics)
- Try ensemble methods for better predictions
- Deploy the model as an API

---
📌 **Author:** [M26I](https://github.com/M26I)  




