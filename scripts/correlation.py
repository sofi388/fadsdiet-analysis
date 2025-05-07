import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


path = 'D:/Users/veres/UEF/Thesis/fadsdiet-analysis/data/processed/fadsdiet_preprocessed.csv'
data = pd.read_csv(path)

correlation_matrix = data.corr()

plt.figure(figsize=(25, 20))  
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('FADSDIET data correlation Heatmap')
plt.show()