import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np

data = pd.read_csv('D:/Users/veres/UEF/Thesis/fadsdiet-analysis/data/processed/fadsdiet_preprocessed.csv')

targets = [ #'fad8_glauc', # 'fad8_chol', 'fad8_ldlc', 'fad8_hdlc', 'fad8_tg', 
            'fad8_crp', 'fad8_hba1c'] 
           #'fad8_glauc', 'fad8_gluc0']

for target in targets:

    columns_to_drop = data.filter(regex='^fad8').columns.difference([target]).tolist()
    columns_to_drop2 = data.filter(regex='^fad8').columns.difference([target]).tolist()

    columns_to_drop2.extend(['genotype', 'group'])

    if target == 'fad8_crp':
        columns_to_drop.extend(['Lnfad8_crp'])
        columns_to_drop2.extend(['Lnfad8_crp'])

    data_dropped = data.drop(columns=columns_to_drop)

    # Save the dataset with dropped columns
    # data_dropped.to_csv('D:/Users/veres/UEF/Thesis/fadsdiet-analysis/data/processed/fadsdiet_preprocessed_dropped.csv', index=False)

    # Initialize lists to store mean squared errors
    mse_genotype = []
    mae_genotype = []
    r2_genotype = []

    mse = []
    mae = []
    r2 = []
    

    for _ in range(10):
        # Dataset with genotype and group
        data_dropped = data.drop(columns=columns_to_drop)
        features_with = data_dropped.drop(columns=[target]).columns
        X_train_with, X_test_with, y_train_with, y_test_with = train_test_split(data[features_with], data[target], test_size=0.2, random_state=42)
        rf_with = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_with.fit(X_train_with, y_train_with)
        y_pred_with = rf_with.predict(X_test_with)

        # with genotype errors
        mse_genotype.append(mean_squared_error(y_test_with, y_pred_with))
        mae_genotype.append(mean_absolute_error(y_test_with, y_pred_with))
        r2_genotype.append(r2_score(y_test_with, y_pred_with))

        # Dataset without genotype and group
        data_dropped = data.drop(columns=columns_to_drop2)
        features_without = data_dropped.drop(columns=[target]).columns
        X_train_without, X_test_without, y_train_without, y_test_without = train_test_split(data_dropped[features_without], data_dropped[target], test_size=0.2, random_state=42)
        rf_without = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_without.fit(X_train_without, y_train_without)
        y_pred_without = rf_without.predict(X_test_without)

        # without genotype errors
        mse.append(mean_squared_error(y_test_without, y_pred_without))
        mae.append(mean_absolute_error(y_test_without, y_pred_without))
        r2.append(r2_score(y_test_without, y_pred_without))

    # Calculate the average errors for each case
    avg_mse_genotype = np.mean(mse_genotype)
    avg_mse = np.mean(mse)

    avg_mae_genotype = np.mean(mae_genotype)
    avg_mae = np.mean(mae)

    avg_r2_genotype = np.mean(r2_genotype)
    avg_r2 = np.mean(r2)

    difference_mse = (avg_mse - avg_mse_genotype) / avg_mse_genotype * 100

    # Print the results
    print(f"Results for {target}:")
    print(f'Average MSE with genotype and group: {round(avg_mse_genotype, 5)}')
    print(f'Average MSE without genotype and group: {round(avg_mse, 5)}')
    print(f'Average MAE with genotype and group: {round(avg_mae_genotype, 2)}')
    print(f'Average MAE without genotype and group: {round(avg_mae, 2)}')
    print(f'Average R² with genotype and group: {round(avg_r2_genotype, 2)}')
    print(f'Average R² without genotype and group: {round(avg_r2, 2)}')
    print(f'Difference in MSE: {round(difference_mse, 2)}%')
    print("-" * 50)