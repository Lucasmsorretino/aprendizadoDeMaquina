import os.path
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

from itertools import combinations

import statsmodels.api as sm

import matplotlib.pyplot as plt 
import seaborn as sns
import joblib

import warnings
warnings.filterwarnings("ignore")

scaler = MinMaxScaler()
le = preprocessing.LabelEncoder()
plt.rcParams["figure.figsize"] = [22,8]




# Gerar 1000 campos aleatórios
random_data = []
for _ in range(1000):
    row = np.random.choice(['leite','pao','bolacha','suco','ovos','café'], size=4, replace=False)
    random_data.append(row)

# Convertendo a lista de campos aleatórios para um dataframe
df = pd.DataFrame(random_data, columns=['prod1','prod2','prod3','prod4'])

print(df.head())


df_onehot = pd.get_dummies(df[['prod1','prod2','prod3','prod4']])
print(df_onehot.head())

# Encontrar itemsets frequentes usando o algoritmo Apriori
frequent_itemsets = apriori(df_onehot, min_support=0.01, use_colnames=True)

# Mostrar itemsets frequentes
print(frequent_itemsets)

# Gerar regras de associação
regras = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

# Exibir as regras geradas
print("\nRegras de associação geradas:")
print(regras[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Salvar em arquivo CSV
regras.to_csv("/Users/jaimewojciechowski/Dropbox/Jaime/AA-UFPR/EspecializacaoIAA2026/Praticas Python/12 - Regras de Associacao - Praticas/12 - Regras de Associacao - Praticas – 1 - Lista de Compras/Lista de Compras - Regras Geradas.csv", index=False)
print("\nRegras salvas em '/Users/jaimewojciechowski/Dropbox/Jaime/AA-UFPR/EspecializacaoIAA2026/Praticas Python/12 - Regras de Associacao - Praticas/12 - Regras de Associacao - Praticas – 1 - Lista de Compras/Lista de Compras - Regras Geradas.csv'")


