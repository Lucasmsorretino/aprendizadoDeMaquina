import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report, log_loss, jaccard_score, cohen_kappa_score, roc_auc_score, average_precision_score, hamming_loss

from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm

import matplotlib.pyplot as plt 
import seaborn as sns
import joblib

# Definada as sementes para reprodutibilidade
random_seed = 196572
np.random.seed(random_seed)

scaler = MinMaxScaler()
plt.rcParams["figure.figsize"] = [22,8]
le = preprocessing.LabelEncoder()

##############################################
# Abre o arquivo e mostra o conteúdo

df = pd.read_csv('/Users/jaimewojciechowski/Dropbox/Jaime/AA-UFPR/EspecializacaoIAA2026/Praticas Python/6 - Veiculos/Veiculos - Dados.csv',sep=',')
df = df.drop('a', axis = 1)

df.head()

print(' ')
print('###################')
print('Conteúdo do arquivo - Veículos')
print(df.head())

##############################################
# Mostra gráfico de correlação
# corr = df.corr(method='pearson')
# sns.heatmap(corr,cmap='seismic',annot=True, fmt=".3f")
# plt.show()


##############################################
# EXPERIMENTO - Separa as bases
y = df['tipo']
X = df.drop('tipo', axis = 1)

columns = list(X.columns)
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 13)
#X.head()


##############################################
# EXPERIMENTO SVM
# Demorado
# param_grid = {
#     'C': [0.1, 1, 10],
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'gamma': ['scale', 'auto']  
# }
param_grid = {
    'C': [0.1, 1],
    'kernel': ['rbf'],
    'gamma': ['auto']  
}
grid = GridSearchCV(SVC(), param_grid, n_jobs= -1)
grid.fit(X_train, y_train)

print(' ')
print('###########################')
print('EXPERIMENTO SVM - Veículos')
print(' ')
print('Melhores parâmetros:')
print(' ')

print(grid.best_params_) 
y_pred = grid.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
jaccard = jaccard_score(y_test, y_pred, average=None)
cohen_kappa = cohen_kappa_score(y_test, y_pred)
hamming = hamming_loss(y_test, y_pred)

# Imprimindo as métricas
print("Accuracy:", accuracy)
print("Jaccard Index:", jaccard)
print("Cohen's Kappa:", cohen_kappa)
print("Hamming Loss:", hamming)
print("Classification Report:\n", class_report)

conf_matrix = confusion_matrix(y_test, y_pred)

# Define os rótulos das classes para um gráfico
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.xlabel('Previsto')
# plt.ylabel('Corretos')
# plt.title('Matrix de confusão - SVM')
# class_names = np.unique(y_test)
# tick_marks = np.arange(len(class_names))
# plt.xticks(tick_marks, class_names, rotation=45)
# plt.yticks(tick_marks, class_names)
# plt.show()


print(' ')
print('SVM - Veiculos - Matriz de Confusão')
print(conf_matrix)


##############################################
# Predição de Novos Casos
#

# Salvando o modelo e scaler
joblib.dump(grid.best_estimator_, "svm_modelo_treinado.pkl")
joblib.dump(scaler, "svm_scaler_treinado.pkl")

# Caminhos dos arquivos
modelo_path = "svm_modelo_treinado.pkl"
scaler_path = "svm_scaler_treinado.pkl"
dados_novos_path = "/Users/jaimewojciechowski/Dropbox/Jaime/AA-UFPR/EspecializacaoIAA2026/Praticas Python/6 - Veiculos/Veiculos - Novos Casos - Para Python.csv"  # CSV sem variável alvo

# Carrega o modelo e o scaler
modelo = joblib.load(modelo_path)
scaler = joblib.load(scaler_path)

# Carrega os novos dados
dados_novos = pd.read_csv(dados_novos_path)

# Padroniza os dados
dados_novos_scaled = scaler.transform(dados_novos)

# Faz as predições
predicoes = modelo.predict(dados_novos_scaled)

# Salva os resultados
dados_novos['predicao'] = predicoes
dados_novos.to_csv("/Users/jaimewojciechowski/Dropbox/Jaime/AA-UFPR/EspecializacaoIAA2026/Praticas Python/6 - Veiculos/Veiculos - Novos Casos - Predicoes em Python SVM.csv", index=False)

# Exibe o resultado
print("Predições realizadas com sucesso.")
print(dados_novos[['predicao']])
print("Arquivo salvo como 'Veiculos - Novos Casos - Predicoes em Python SVM.csv'")
