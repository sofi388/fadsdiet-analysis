import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ParameterEstimator, MaximumLikelihoodEstimator
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import numpy as np


data_path = 'D:/Users/veres/UEF/Thesis/fadsdiet-analysis/data/processed/fadsdiet_preprocessed.csv'
data = pd.read_csv(data_path)

columns = ["genotype", "diet", "fad8_crp"]
data = data[columns]

data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

data["genotype"] = data["genotype"].astype(str)
data["diet"] = data["diet"].astype(str)

model = BayesianNetwork([("genotype", "fad8_crp"), ("diet", "fad8_crp")])
model.fit(data, estimator=MaximumLikelihoodEstimator)

inference = VariableElimination(model)

result = inference.query(variables=["fad8_crp"], evidence={"genotype": "0", "diet": "0"})

print(result)

# Visualizations
genotype1_values = [
    0.0455, 0.0000, 0.0000, 0.0000, 0.0000, 0.2273, 0.0909, 
    0.0455, 0.0455, 0.2727, 0.0909, 0.0455, 0.0455, 0.0455, 0.0455
]
genotype2_values = [
    0.0000, 0.0000, 0.0263, 0.0263, 0.0000, 0.0526, 0.1842, 
    0.1316, 0.1316, 0.1842, 0.0789, 0.0789, 0.1053, 0.0000, 0.0000
]


x = np.arange(29, 44)

plt.figure(figsize=(10, 6))

plt.plot(x, genotype1_values, label='Genotype 1', color='b', marker='o')
plt.plot(x, genotype2_values, label='Genotype 2', color='r', marker='x')

plt.title('Compare phi(fad8_hba1c) for two genotypes, high-ALA Diet', fontsize=14)
plt.xlabel('Individual (number)', fontsize=12)
plt.ylabel('phi(fad8_crp)', fontsize=12)

plt.legend()

plt.grid(True)
plt.tight_layout()
plt.show()
