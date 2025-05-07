import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('D:/Users/veres/UEF/Thesis/fadsdiet-analysis/data/processed/fadsdiet_preprocessed_dropped.csv')
# data = pd.read_csv('D:/Users/veres/UEF/Thesis/fadsdiet-analysis/data/processed/fadsdiet_synthetic.csv')

target = 'fad8_chol'

print("Initial Data shape:", data.shape)

data.fillna(data.mean(), inplace=True)

X = data.drop(columns=[target])
y = data[target]

# Scale features (Gaussian processes can benefit from feature scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the Gaussian Process kernel
kernel = C(1.0) * RBF(length_scale=1.0)

# Initialize the Gaussian Process Regressor
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Fit the model to the training data
gp.fit(X_train, y_train)

y_pred, sigma = gp.predict(X_test, return_std=True)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R² Score: {r2:.3f}")

cv_scores = cross_val_score(gp, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
print("\nCross-Validation Results (RMSE):", cv_rmse)
print(f"Mean RMSE from CV: {cv_rmse.mean():.3f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k', label="Perfect Prediction")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("True vs Predicted Cholesterol Levels")
plt.legend()
plt.show()