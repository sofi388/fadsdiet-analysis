import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ParameterEstimator, MaximumLikelihoodEstimator
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import numpy as np

data_path = 'D:/Users/veres/UEF/Thesis/fadsdiet-analysis/data/processed/fadsdiet_preprocessed.csv'
data = pd.read_csv(data_path)
columns = ["genotype", "diet", "fad8_hba1c"]
data = data[columns]

data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

data["genotype"] = data["genotype"].astype(str)
data["diet"] = data["diet"].astype(str)


model = BayesianNetwork([("genotype", "fad8_hba1c"), ("diet", "fad8_hba1c")])
model.fit(data, estimator=MaximumLikelihoodEstimator)

inference = VariableElimination(model)

result = inference.query(variables=["fad8_hba1c"], evidence={"genotype": "0", "diet": "0"})

print(result)



# Данные для двух различных генотипов
genotype1_values = [
    0.0455, 0.0000, 0.0000, 0.0000, 0.0000, 0.2273, 0.0909, 
    0.0455, 0.0455, 0.2727, 0.0909, 0.0455, 0.0455, 0.0455, 0.0455
]
genotype2_values = [
    0.0000, 0.0000, 0.0263, 0.0263, 0.0000, 0.0526, 0.1842, 
    0.1316, 0.1316, 0.1842, 0.0789, 0.0789, 0.1053, 0.0000, 0.0000
]

# Индексы (например, идентификаторы пациентов или просто индексы наблюдений)
x = np.arange(29, 44)

# Построение графика
plt.figure(figsize=(10, 6))

# Линии для двух генотипов
plt.plot(x, genotype1_values, label='Genotype 1', color='b', marker='o')
plt.plot(x, genotype2_values, label='Genotype 2', color='r', marker='x')

# Добавление заголовков и подписей
plt.title('Сравнение phi(fad8_hba1c) для двух генотипов', fontsize=14)
plt.xlabel('Индивидуальные наблюдения (номер)', fontsize=12)
plt.ylabel('phi(fad8_hba1c)', fontsize=12)

# Показать легенду
plt.legend()

# Показать график
plt.grid(True)
plt.tight_layout()
plt.show()
