import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import cross_val_score


#  Load Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "data", "Housing.csv")
df = pd.read_csv(data_path)

# OHE
df = pd.get_dummies(df, drop_first=True)

#  Price Distribution Before Removing Outliers
plt.figure(figsize=(10, 5))
sns.histplot(df["price"], bins=50, kde=True)
plt.title("Price Distribution Before Removing Outliers")
plt.show()

#  Remove Outliers using IQR
Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df["price"] >= lower_bound) & (df["price"] <= upper_bound)]
print(f"Dataset size after removing outliers: {df.shape}")

#  Price Distribution After Removing Outliers
plt.figure(figsize=(10, 5))
sns.histplot(df["price"], bins=50, kde=True)
plt.title("Price Distribution After Removing Outliers")
plt.show()

#  Feature Engineering
df["price_per_sqft"] = df["price"] / (df["area"] + 1)
df["rooms_per_sqft"] = df["bedrooms"] / (df["area"] + 1)

#  Define Features and Target
X = df.drop(columns=["price"])
y = np.log1p(df["price"])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize Numerical Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'max_depth': [4, 5],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [200, 500],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [1, 2],
    'gamma': [0, 0.1],
    'reg_alpha': [0, 0.1]
}
xgb_model = xgb.XGBRegressor(random_state=42)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best model from GridSearchCV
print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Predictions & Evaluate
xgb_pred = np.expm1(best_model.predict(X_test_scaled))
y_test_actual = np.expm1(y_test)
xgb_mae = mean_absolute_error(y_test_actual, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test_actual, xgb_pred))
r2 = r2_score(y_test_actual, xgb_pred)

print(f"Best XGBoost - MAE: {xgb_mae:.2f}, RMSE: {xgb_rmse:.2f}, R²: {r2:.2f}")

# Perform cross-validation using XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, random_state=42)
cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=5, scoring='neg_root_mean_squared_error')
print(f"Cross-validated RMSE: {-cv_scores.mean():.2f}")

# Residual Plot
residuals = y_test_actual - xgb_pred
plt.figure(figsize=(8, 6))
sns.scatterplot(x=xgb_pred, y=residuals, alpha=0.6)
plt.hlines(0, xmin=min(xgb_pred), xmax=max(xgb_pred), colors='red')
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Prices")
plt.show()

#feature importance score, for future improving or tuning
xgb.plot_importance(best_model, importance_type='weight')
plt.show()
