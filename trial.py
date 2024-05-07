import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import time
import matplotlib.pyplot as plt

# Load the dataset (adjust the file path accordingly)
file_path = 'train.csv'
data = pd.read_csv(file_path)

# Preprocess the data
X = data.drop('TARGET(PRICE_IN_LACS)', axis=1)
y = data['TARGET(PRICE_IN_LACS)']

X = X.drop('ADDRESS', axis=1)

label_encoder = LabelEncoder()
X['BHK_OR_RK'] = label_encoder.fit_transform(X['BHK_OR_RK'])
X['POSTED_BY'] = label_encoder.fit_transform(X['POSTED_BY'])

X = pd.get_dummies(X, columns=['BHK_OR_RK', 'POSTED_BY'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Save the best model
joblib.dump(best_model, 'random_forest_model_analysis.joblib')

# Measure training time
start_time = time.time()
best_model.fit(X_train, y_train)
end_time = time.time()
training_time = end_time - start_time

# Print the training time
print(f'Training Time: {training_time} seconds')

# Plotting the results
plt.figure(figsize=(10, 5))

# Bar chart for Mean Squared Error
plt.subplot(1, 2, 1)
plt.bar('Random Forest', mse, color='blue')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error for Random Forest')

# Bar chart for R-squared
plt.subplot(1, 2, 2)
plt.bar('Random Forest', r2, color='green')
plt.ylabel('R-squared')
plt.title('R-squared for Random Forest')

plt.tight_layout()
plt.show()
