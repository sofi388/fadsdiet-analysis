import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

data = pd.read_csv('D:/Users/veres/UEF/Thesis/fadsdiet-analysis/data/processed/fadsdiet_preprocessed.csv')

target = 'fad8_chol'

columns_to_drop = data.filter(regex='^fad8').columns.difference([target]).tolist()
columns_to_drop2 = data.filter(regex='^fad8').columns.difference([target]).tolist()

columns_to_drop2.extend(['genotype', 'group'])

# Drop the specified columns
data_dropped = data.drop(columns=columns_to_drop)

# Save the dataset with dropped columns
# data_dropped.to_csv('D:/Users/veres/UEF/Thesis/fadsdiet-analysis/data/processed/fadsdiet_preprocessed_dropped.csv', index=False)

# Initialize lists to store mean squared errors
mse_with_genotype_group = []
mse_without_genotype_group = []

for _ in range(10):
    # Dataset with genotype and group
    data_dropped = data.drop(columns=columns_to_drop)
    features_with = data_dropped.drop(columns=[target]).columns
    X_train_with, X_test_with, y_train_with, y_test_with = train_test_split(data[features_with], data[target], test_size=0.2, random_state=42)
    rf_with = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_with.fit(X_train_with, y_train_with)
    y_pred_with = rf_with.predict(X_test_with)
    mse_with_genotype_group.append(mean_squared_error(y_test_with, y_pred_with))

    # Dataset without genotype and group
    data_dropped = data.drop(columns=columns_to_drop2)
    features_without = data_dropped.drop(columns=[target]).columns
    X_train_without, X_test_without, y_train_without, y_test_without = train_test_split(data_dropped[features_without], data_dropped[target], test_size=0.2, random_state=42)
    rf_without = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_without.fit(X_train_without, y_train_without)
    y_pred_without = rf_without.predict(X_test_without)
    mse_without_genotype_group.append(mean_squared_error(y_test_without, y_pred_without))

# Calculate the average mean squared error for each case
avg_mse_with_genotype_group = np.mean(mse_with_genotype_group)
avg_mse_without_genotype_group = np.mean(mse_without_genotype_group)
difference_percentage = (avg_mse_without_genotype_group - avg_mse_with_genotype_group) / avg_mse_with_genotype_group * 100

print(f'Average Mean Squared Error with genotype and group: {avg_mse_with_genotype_group}')
print(f'Average Mean Squared Error without genotype and group: {avg_mse_without_genotype_group}')
print(f'Difference in MSE: {difference_percentage:.2f}%')