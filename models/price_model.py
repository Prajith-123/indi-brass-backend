import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Read the dataset from the CSV file
df = pd.read_csv('../DatasetCollection1.csv')

# Clean the column names (strip spaces and normalize to title case)
df.columns = df.columns.str.strip()
df.columns = df.columns.str.title()

# Select relevant features for predicting Total Amount ($)
features = df[['Material', 'Item Type', 'Quantity (Pcs)', 'Price ($)']]

# One-hot encode categorical variables: Material and Item Type
features_encoded = pd.get_dummies(features, columns=['Material', 'Item Type'], drop_first=True)

# Target variable: Total Amount ($)
target = df['Total Amount ($)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)

# Train Gradient Boosting Regressor model
gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr.fit(X_train, y_train)

# Predictions
predictions = gbr.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# Output evaluation metrics
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')

# Save the trained model using pickle
with open('../pickles/price_model.pkl', 'wb') as f:
    pickle.dump(gbr, f)

print("Model saved successfully!")
