import os.path
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

file  = '/Users/jaimewojciechowski/Dropbox/Jaime/AA-UFPR/EspecializacaoIAA2026/Praticas Python/11 - Agrupamento - Praticas/11 - Agrupamento - Praticas - 2 - Moveis/Moveis - Dados.csv'

df = pd.read_csv(file, encoding='latin1')

print(' ')
print('###################')
print('Conteúdo do arquivo - Móveis')
print(df.head())



dfg = df

dfg['categoria'] = le.fit_transform(df['categoria'])
dfg['cor'] = le.fit_transform(df['cor'])
dfg['estilo'] = le.fit_transform(df['estilo'])
print(dfg.head())

inertia = []

# Loop para diferentes valores de k
for k in range(1, 11):  # Testando de 1 a 10 clusters
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(dfg)
    inertia.append(kmeans.inertia_)

# Plotar o gráfico
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 11), inertia, marker='o')
# plt.title('Método do Cotovelo')
# plt.xlabel('Número de clusters')
# plt.ylabel('Inércia')
# plt.show()

# # Loop para diferentes valores de k
# for k in range(2, 11):  # Tem que ter pelo menos 2 clusters para calcular o silhouette_score
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(dfg)
#     labels = kmeans.labels_
#     sil_score = silhouette_score(dfg, labels)
#     sil_scores.append(sil_score)

# # Plotar o gráfico
# plt.figure(figsize=(10, 6))
# plt.plot(range(2, 11), sil_scores, marker='o')
# plt.title('Método Silhouette')
# plt.xlabel('Número de clusters')
# plt.ylabel('Silhouette Score')
# plt.show()


kmeans = KMeans(n_clusters=5)  # Colocar o número ideal
kmeans.fit(dfg)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

df['cluster'] = labels

print(df.head(30))



#Exemplo de agrupamento  			
# plt.scatter( df['categoria'], df['cor'], c=df['cluster'], cmap='rainbow')
# plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200)
# plt.xlabel('categoria')
# plt.ylabel('cor')
# plt.title('K-means Clustering')
# plt.show()




