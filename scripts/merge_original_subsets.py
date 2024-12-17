import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

subset_dir = 'D:/Users/veres/UEF/Thesis/fadsdiet-analysis/data/subsets' # Use an absolute path

if not os.path.exists(subset_dir):
    print(f"Error: The directory {subset_dir} does not exist.")
else:
    files = [f for f in os.listdir(subset_dir) if f.startswith('subset') and f.endswith('.csv')]

    if files:
        dfs = [pd.read_csv(os.path.join(subset_dir, file)) for file in files]
        merged_df = dfs[0] 

        # Merge the remaining dataframes by index (avoid using 'on' parameter)
        for df in dfs[1:]:
            merged_df = pd.concat([merged_df, df], axis=1, join='outer')

        correlation_matrix = merged_df.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Merged DataFrames')
        plt.show()
    else:
        print(f"No CSV files found starting with 'subset' in {subset_dir}")
