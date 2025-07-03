import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import pickle

# Read the dataset from the CSV file
df = pd.read_csv('../DatasetCollection1.csv')

# Define the features and target variable
ip_qty = df['IP Qty'].values
mp_qty = df['MP Qty'].values
item_length = df['Item Length'].values
item_width = df['Item Width'].values
item_height = df['Item Height'].values
quantity_pcs = df['QUANTITY (PCS)'].values
completion_days = df['COMPLETION DAYS'].values

# Prepare the dataset for training
X = np.array(list(zip(ip_qty, mp_qty, item_length, item_width, item_height, quantity_pcs)))
y = np.array(completion_days)

# Train the model
model = DecisionTreeRegressor()
model.fit(X, y)

# Save the trained model
with open('../pickles/timeline_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully!")
