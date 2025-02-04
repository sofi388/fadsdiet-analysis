import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Set the directory for subsets (use absolute path)
subset_dir = 'D:/Users/veres/UEF/Thesis/fadsdiet-analysis/data/subsets'

if not os.path.exists(subset_dir):
    print(f"Error: The directory {subset_dir} does not exist.")
else:
    # Find all relevant subset files
    files = [f for f in os.listdir(subset_dir) if f.startswith('subset') and f.endswith('.csv')]

    if len(files) < 2:
        print("Error: At least two subset files are required for merging.")
    else:
        # Load all subsets into a list of DataFrames
        dfs = [pd.read_csv(os.path.join(subset_dir, file)) for file in files]
        total_columns = sum([df.shape[1] for df in dfs])
        print(f"Total columns across all subsets (including duplicates): {total_columns}")

        # Find the common columns across all subsets
        intersecting_columns = set(dfs[0].columns)
        for df in dfs[1:]:
            intersecting_columns &= set(df.columns)
        intersecting_columns = list(intersecting_columns)

        print(f"Common columns identified for kNN: {intersecting_columns}")

        # Initialize the merged DataFrame with all columns
        merged_df_full = pd.concat(dfs, axis=1)
        print(f"Shape of merged DataFrame with all columns: {merged_df_full.shape}")

        # Use the first two subsets for kNN-based merging
        subset1 = dfs[0]
        subset2 = dfs[1]
        # Use only intersecting columns for kNN, handling missing values
        X1 = subset1[intersecting_columns].fillna(0).values  # Replace NaN with 0 for kNN computation
        X2 = subset2[intersecting_columns].fillna(0).values

        # Fit kNN to find the nearest neighbors from subset2 for each row in subset1
        knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        knn.fit(X2)
        distances, indices = knn.kneighbors(X1)

        # Filter rows with valid neighbors
        subset1_matched = subset1.iloc[np.arange(len(X1))]  # Keep the same number of rows as X1
        subset2_matched = subset2.iloc[indices.flatten()]

        # Merge all columns for matched rows
        merged_knn_rows = pd.concat([subset1_matched.reset_index(drop=True),
                                    subset2_matched.reset_index(drop=True)], axis=1)

        # Include unmatched columns from all subsets
        final_columns = list(set(merged_df_full.columns))
        # Remove duplicate columns
        merged_knn_rows = merged_knn_rows.loc[:, ~merged_knn_rows.columns.duplicated()]

        # Reindex with the final set of columns
        merged_knn_rows = merged_knn_rows.reindex(columns=final_columns, fill_value=np.nan)

        merged_knn_rows = merged_knn_rows.reindex(columns=final_columns, fill_value=np.nan)

        print(f"Shape of merged DataFrame after kNN matching: {merged_knn_rows.shape}")

        # Output the final merged DataFrame
        merged_knn_rows.to_csv("merged_knn_output.csv", index=False)
        print("Merged DataFrame with kNN matching saved to 'merged_knn_output.csv'")