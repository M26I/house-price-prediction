🏡 Housing Price Prediction
📌 Overview
This project predicts housing prices based on factors like area, number of bedrooms, and other property features. The goal is to build a regression model that estimates housing prices given property data.

💋 Dataset
Source: [Kaggle Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset?)
Features:
price: Price of the house (Target variable)
area: Area of the house in square feet
bedrooms: Number of bedrooms in the house
location: Location of the house (Encoded as categorical variable)
year_built: Year the house was built
sqft_living: Square footage of living space
sqft_lot: Square footage of the lot

🔧 Steps
Data Loading: Loaded the dataset from a CSV file.
Outlier Removal: Removed outliers in the target variable (price) using the Interquartile Range (IQR) method.
Feature Engineering: Created new features such as price_per_sqft and rooms_per_sqft.
Data Preprocessing: Standardized numerical features and performed one-hot encoding (OHE) for categorical features.
Model Training: Used XGBoost for regression with hyperparameter tuning via GridSearchCV.
Evaluation: Measured performance using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R².

🖥️ Installation & Usage
Clone the repository:

git clone https://github.com/M26I/housing-price-prediction


📊 Results
The model achieved:

Mean Absolute Error (MAE): 162685.57
Root Mean Squared Error (RMSE): 228625.48
R²: 0.98
Cross-validated RMSE: 0.06


🔗 Contributing
Feel free to fork this repository, make improvements, and submit pull requests.

✨ Acknowledgments
Thanks to the creators of the dataset and the libraries used in this project.

Author: M26I