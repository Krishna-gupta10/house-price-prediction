import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained XGBoost model
xgb_model = joblib.load('xgboost_model.joblib')

# Load the entire train.csv dataset
file_path = 'train.csv'
data = pd.read_csv(file_path)

# Preprocess the data
X = data.drop('TARGET(PRICE_IN_LACS)', axis=1)
y_actual = data['TARGET(PRICE_IN_LACS)']

X = X.drop('ADDRESS', axis=1)

label_encoder = LabelEncoder()
X['BHK_OR_RK'] = label_encoder.fit_transform(X['BHK_OR_RK'])
X['POSTED_BY'] = label_encoder.fit_transform(X['POSTED_BY'])

X = pd.get_dummies(X, columns=['BHK_OR_RK', 'POSTED_BY'], drop_first=True)

# Predict using the XGBoost model
y_pred = xgb_model.predict(X)

# Calculate the absolute differences between actual and predicted prices
differences = np.abs(y_actual - y_pred)

# Calculate and print the average difference
average_difference = np.mean(differences)
print(f'Average Difference between Actual and Predicted Prices: {average_difference} Lacs')

# Display the predicted prices alongside actual prices
plt.figure(figsize=(10, 5))

# Plot actual prices
plt.scatter(range(len(y_actual)), y_actual, color='blue', label='Actual Prices')

# Plot predicted prices
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Prices')

plt.xlabel('Data Points')
plt.ylabel('Price in Lacs')
plt.title('Actual vs. Predicted House Prices using XGBoost')
plt.legend()
plt.show()
