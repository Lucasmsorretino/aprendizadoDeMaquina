import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#%%
warnings.filterwarnings("ignore")

le = preprocessing.LabelEncoder()
plt.rcParams["figure.figsize"] = [22,8]

file  = r"C:\Users\lcast\OneDrive\Documents\Especialização UFPR\IAA08 - Aprendizado de Máquina\aprendizadoDeMaquina\6Veiculos\dados\veiculos.csv"
df = pd.read_csv(file, sep = ',')
df = df.drop('a', axis = 1)

print(' ')
print('###################')
print('Conteúdo do arquivo - Veículos')
print(df.head())

#%% Preparar os dados para o Agrupamento
# Criamos um novo DataFrame 'X' contendo apenas as features que serão usadas para agrupar.
X = df.drop('tipo', axis=1)

#%% Executar o K-Means
# Usamos n_clusters=10.
# Adicionar random_state e n_init garante que os resultados sejam reprodutíveis.
kmeans = KMeans(n_clusters=10, random_state=202526, n_init=10)
kmeans.fit(X)

# Adicionar os clusters encontrados de volta ao DataFrame original para análise
df['cluster'] = kmeans.labels_

print('\nDataFrame com os 10 clusters atribuídos:')
print(df.head(10))

#%% Visualização dos Clusters com PCA (Principal Component Analysis)
# PCA reduz as 18 features para 2 dimensões para que possamos plotar em um gráfico 2D.

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Transforma as coordenadas dos centroides para o mesmo espaço 2D do PCA
centroids = kmeans.cluster_centers_
centroids_pca = pca.transform(centroids)

#%%
# Plotar o gráfico de dispersão
plt.figure(figsize=(14, 10))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='tab20', alpha=0.8, edgecolors='k', s=50)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=250, c='red', label='Centroides')
plt.title('Visualização dos 10 Clusters (K-Means) com PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
legend_labels = [f'Cluster {i}' for i in range(10)]
plt.legend(handles=scatter.legend_elements(num=10)[0], labels=legend_labels, title="Clusters")
plt.grid(True)
plt.show()

#%%
# Salvar resultados em um csv
df.to_csv('Veiculos agrupados kmeans = 10.csv', index=False)
