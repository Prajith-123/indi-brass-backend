import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Read data from CSV (adjust if you have the path)
data = pd.read_csv('../DatasetCollection1.csv')

# Features preparation
features = data[['IP Qty', 'MP Qty', 'QUANTITY (PCS)']].copy()

# Create new ratios
features.loc[:, 'Ip_to_Pcs_Ratio'] = features['IP Qty'] / features['QUANTITY (PCS)']
features.loc[:, 'Mp_to_Pcs_Ratio'] = features['MP Qty'] / features['QUANTITY (PCS)']

# Target variable
target = features['Ip_to_Pcs_Ratio']  # This is the labor efficiency ratio

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Set up XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')

# Hyperparameter grid for XGBoost
xgb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

# Perform GridSearchCV to find the best parameters for XGBoost
grid_search_xgb = GridSearchCV(xgb_model, xgb_param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_xgb.fit(X_train, y_train)

# Best XGBoost model
best_xgb = grid_search_xgb.best_estimator_

# Predictions and evaluation
y_pred_xgb = best_xgb.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost RMSE: {rmse_xgb}")
print(f"XGBoost R²: {r2_xgb}")

# Cross-validation for XGBoost
cv_scores_xgb = cross_val_score(best_xgb, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"XGBoost Cross-validation scores: {cv_scores_xgb}")

# Save the trained model using pickle
with open('../pickles/labor_model.pkl', 'wb') as f:
    pickle.dump(best_xgb, f)

print("Labor model saved successfully!")