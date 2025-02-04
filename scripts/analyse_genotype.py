import pandas as pd

"""
Highest correlation, all features
"""
# Load the preprocessed data
path = 'D:/Users/veres/UEF/Thesis/fadsdiet-analysis/data/processed/fadsdiet_preprocessed.csv'
data = pd.read_csv(path)

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Get the correlations with 'genotype'
genotype_correlations = correlation_matrix['genotype']

# Filter for columns starting with 'fad8'
fad8_correlations = genotype_correlations.filter(like='fad8')

# Sort the correlations and get the 10 highest correlated variables
top_10_fad8_correlated = fad8_correlations.abs().sort_values(ascending=False).head(10)

print("Top 10 variables starting with 'fad8' most correlated with genotype:")
print(top_10_fad8_correlated)



"""
Correlation with target features
"""
# Target features
target_features = ['fad8_chol', 'fad8_crp', 'fad8_gluc120', 'fad8_ins30', 'fad8_ldlc', 'fad8_gluc0', 'fad8_hba1c', 'fad8_ins120', 'fad8_hdlc', 'fad8_gluc30', 'fad8_ins0', 'fad8_glauc']

# Correlation of target features with genotype
target_correlations = correlation_matrix.loc[target_features, 'genotype']

# Print the correlation of target features with genotype
print("\nCorrelation of target features with genotype:")
print(target_correlations)


"""
Split dataset into subsets based on diet type
"""

data_la_diet = data[data['diet'] == 0]
data_ala_diet = data[data['diet'] == 1]

print("\nNumber of samples for LA diet:", len(data_la_diet))
print("Number of samples for ALA diet:", len(data_ala_diet))

# Calculate correlations target features with genotype for each diet type
correlation_matrix_la_diet = data_la_diet.corr()
correlation_matrix_ala_diet = data_ala_diet.corr()

target_correlations_la_diet = correlation_matrix_la_diet.loc[target_features, 'genotype']
target_correlations_ala_diet = correlation_matrix_ala_diet.loc[target_features, 'genotype']

print("\nCorrelation of target features with genotype for LA diet:")
print(target_correlations_la_diet)

print("\nCorrelation of target features with genotype for ALA diet:")
print(target_correlations_ala_diet)



"""
Split subsets based on diet type into subsets based on genotype
"""
# Split the data into subsets based on genotype
#data_la_genotype1 = data_la_diet[data_la_diet['genotype'] == 1]
#data_la_genotype2 = data_la_diet[data_la_diet['genotype'] == 0]
#data_ala_genotype1 = data_ala_diet[data_ala_diet['genotype'] == 1]
#data_ala_genotype2 = data_ala_diet[data_ala_diet['genotype'] == 0]


# Calculate correlations of target features with genotype for each subset
#correlation_matrix_la_genotype1 = data_la_genotype1.corr()
#correlation_matrix_la_genotype2 = data_la_genotype2.corr()
#correlation_matrix_ala_genotype1 = data_ala_genotype1.corr()
#correlation_matrix_ala_genotype2 = data_ala_genotype2.corr()

#target_correlations_la_genotype1 = correlation_matrix_la_genotype1.loc[target_features, 'genotype']
#target_correlations_la_genotype2 = correlation_matrix_la_genotype2.loc[target_features, 'genotype']
#target_correlations_ala_genotype1 = correlation_matrix_ala_genotype1.loc[target_features, 'genotype']
#target_correlations_ala_genotype2 = correlation_matrix_ala_genotype2.loc[target_features, 'genotype']

#print("\nCorrelation of target features with genotype for LA diet, genotype 1:")
#print(target_correlations_la_genotype1)

#print("\nCorrelation of target features with genotype for LA diet, genotype 2:")
#print(target_correlations_la_genotype2)

#print("\nCorrelation of target features with genotype for ALA diet, genotype 1:")
#print(target_correlations_ala_genotype1)

#print("\nCorrelation of target features with genotype for ALA diet, genotype 2:")
#print(target_correlations_ala_genotype2)




"""
Split subsets based on diet type into subsets based on genotype
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Data provided
data = {
    'Feature': [
        'fad8_chol', 'fad8_crp', 'fad8_gluc120', 'fad8_ins30', 'fad8_ldlc', 
        'fad8_gluc0', 'fad8_hba1c', 'fad8_ins120', 'fad8_hdlc', 'fad8_gluc30', 
        'fad8_ins0', 'fad8_glauc'
    ],
    'genotype, LA Diet': [
        -0.044932, 0.145451, -0.038643, 0.221436, -0.050838, 
        0.047447, 0.017127, -0.041382, 0.097825, 0.057071, 
        0.065440, 0.010798
    ],
    'genotype, ALA Diet': [
        -0.022976, 0.042347, 0.109617, -0.179570, -0.104134, 
        -0.074406, 0.322956, -0.250969, 0.202241, -0.067506, 
        -0.170113, 0.010808
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Set the index to 'Feature'
df.set_index('Feature', inplace=True)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Correlation Heatmap: LA Diet vs ALA Diet')
plt.show()