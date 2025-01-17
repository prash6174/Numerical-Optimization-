import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('real_estate_dataset.csv')

# Extract dataset dimensions
num_rows, num_columns = data.shape

# Save column names to a text file
columns = data.columns
np.savetxt('columns.txt', columns, fmt='%s')

# Select specific features and target variable
features = data[['Square_Feet', 'Garage_Size', 'Location_Score', 'Distance_to_Center']]
target = data['Price']

# Print feature matrix details
print(f"Feature matrix shape: {features.shape}")
print(f"Feature data types:\n{features.dtypes}\n")

# Initialize coefficients for the linear model
num_features = features.shape[1]
coefficients = np.ones(num_features + 1)

# Compute predictions
intercept = coefficients[0]
weights = coefficients[1:]
predicted_values = features @ weights + intercept

# Add a bias term (column of ones) to the feature matrix
features_with_bias = np.hstack((np.ones((num_rows, 1)), features))

# Predict using augmented feature matrix
predicted_values_aug = features_with_bias @ coefficients

# Compare predictions from both approaches
if np.allclose(predicted_values, predicted_values_aug):
    print("Both prediction methods yield identical results.")

# Calculate errors and relative errors
errors = target - predicted_values_aug
relative_errors = errors / target

# Compute mean squared error
mse = (errors.T @ errors) / num_rows

print(f"Error vector shape: {errors.shape}")
print(f"Error norm: {np.linalg.norm(errors)}")
print(f"Relative error norm: {np.linalg.norm(relative_errors)}")

# Solve for optimal coefficients using the normal equation
optimal_coefficients = np.linalg.inv(features_with_bias.T @ features_with_bias) @ features_with_bias.T @ target
np.savetxt('optimal_coefficients.txt', optimal_coefficients, fmt='%s')

# Compute predictions with optimal coefficients
optimal_predictions = features_with_bias @ optimal_coefficients
optimal_errors = target - optimal_predictions

# Print error norms for the optimized model
print(f"Optimal solution error norm: {np.linalg.norm(optimal_errors)}")
print(f"Optimal solution relative error norm: {np.linalg.norm(optimal_errors / target)}")

# Use all features for model training
all_features = data.drop('Price', axis=1).values
all_features_with_bias = np.hstack((np.ones((num_rows, 1)), all_features))
all_target = data['Price'].values

# Solve normal equation for all features
coefficients_all = np.linalg.inv(all_features_with_bias.T @ all_features_with_bias) @ all_features_with_bias.T @ all_target
np.savetxt('all_features_coefficients.txt', coefficients_all, fmt='%s')

# Predict and compute errors using all features
all_predictions = all_features_with_bias @ coefficients_all
all_errors = all_target - all_predictions

# Print norms for errors with all features
print(f"All features error norm: {np.linalg.norm(all_errors)}")
print(f"All features relative error norm: {np.linalg.norm(all_errors / all_target)}")

# QR decomposition
Q, R = np.linalg.qr(all_features_with_bias)
np.savetxt('R_matrix.csv', R, delimiter=',')
Q_TQ = Q.T @ Q
np.savetxt('Q_TQ_matrix.csv', Q_TQ, delimiter=',')

# Solve using QR decomposition
b = Q.T @ all_target
coefficients_qr = np.zeros(all_features_with_bias.shape[1])
for i in range(coefficients_qr.size - 1, -1, -1):
    coefficients_qr[i] = b[i]
    for j in range(i + 1, coefficients_qr.size):
        coefficients_qr[i] -= R[i, j] * coefficients_qr[j]
    coefficients_qr[i] /= R[i, i]
np.savetxt('qr_coefficients.txt', coefficients_qr, fmt='%s')

# SVD decomposition
U, S, V_T = np.linalg.svd(all_features_with_bias, full_matrices=False)
coefficients_svd = V_T.T @ np.linalg.inv(np.diag(S)) @ U.T @ all_target
np.savetxt('svd_coefficients.txt', coefficients_svd, fmt='%s')
