# 🏡 Housing Price Prediction

## 📌 Overview
This project predicts **housing prices** based on factors like **area, number of bedrooms, and other property features**. The goal is to build a regression model that estimates housing prices given property data.

## 💋 Dataset
- **Source:** [Kaggle Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset?)
- **Features:**
  - `price`: Price of the house (Target variable)
  - `area`: Area of the house in square feet
  - `bedrooms`: Number of bedrooms in the house
  - `location`: Location of the house (Encoded as categorical variable)
  - `year_built`: Year the house was built
  - `sqft_living`: Square footage of living space
  - `sqft_lot`: Square footage of the lot

## 🔧 Steps
1. **Data Loading**: Loaded the dataset from a CSV file.
2. **Outlier Removal**: Removed outliers in the target variable (`price`) using the Interquartile Range (IQR) method.
3. **Feature Engineering**: Created new features such as `price_per_sqft` and `rooms_per_sqft`.
4. **Data Preprocessing**: Standardized numerical features and performed one-hot encoding (OHE) for categorical features.
5. **Model Training**: Used **XGBoost** for regression with hyperparameter tuning via **GridSearchCV**.
6. **Evaluation**: Measured performance using **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **R²**.

## 🖥️ Installation & Usage
Clone the repository:
```bash
git clone https://github.com/M26I/housing-price-prediction


