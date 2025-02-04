import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = 'D:/Users/veres/UEF/Thesis/fadsdiet-analysis/data/processed/fadsdiet_preprocessed.csv'
df = pd.read_csv(file_path)

target_features = ['fad8_chol']  # Add target features if they exist

# Take a random sample of 5 rows
sample_df = df.sample(n=5, random_state=1)


print("Logging ")
# Plot timeline changes between fad0_chol and fad8_chol
plt.figure(figsize=(10, 6))
for index, row in sample_df.iterrows():
    plt.plot(['fad0_chol', 'fad8_chol'], [row['fad0_chol'], row['fad8_chol']], marker='o', label=f'Sample {index}')

plt.xlabel('Time')
plt.ylabel('Cholesterol Levels')
plt.title('Timeline Changes in Cholesterol Levels')
plt.legend()
plt.savefig('timeline_changes.png')
plt.show()

