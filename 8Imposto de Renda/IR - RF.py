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
import csv

# Definada as sementes para reprodutibilidade
random_seed = 196572
np.random.seed(random_seed)

scaler = MinMaxScaler()
plt.rcParams["figure.figsize"] = [22,8]
le = preprocessing.LabelEncoder()


##############################################
# Abre o arquivo e mostra o conteúdo
df = pd.read_csv('/Users/jaimewojciechowski/Dropbox/Jaime/AA-UFPR/EspecializacaoIAA2026/Praticas Python/8 - Imposto de Renda/IR - Dados1.csv',sep=',')

print(' ')
print('###################')
print('Conteúdo do arquivo')
print(df.head())


##############################################
# Transforma as variáveis discretas em 0,1,2

df['rest'] = le.fit_transform(df['rest'])
df['ecivil'] = le.fit_transform(df['ecivil'])
df['sonegador'] = le.fit_transform(df['sonegador'])

# print(' ')
# print('###########################################')
# print('Transformou as variáveis discretas em 0,1,2')
# print(df.head())



##############################################
# Mostra gráfico de correlação

# corr = df.corr(method='pearson')
# sns.heatmap(corr,cmap='seismic',annot=True, fmt=".3f")
#plt.show()


##############################################
# Separa as bases

y = df['sonegador']
###y = le.fit_transform(df['tipo'])
X = df.drop('sonegador', axis = 1)
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 13)


##############################################
# EXPERIMENTO RF
# Demorado
# param_grid = {
#    'n_estimators': list(range(5, 50, 5)),
#    'max_depth': list(range(1, 10, 1)),
#    'min_samples_split': list(range(1, 10, 1)),
#    'min_samples_leaf': list(range(1, 10, 1))
# }
param_grid = {
   'n_estimators': list(range(5, 15, 5)),
   'max_depth': list(range(1, 3, 1)),
   'min_samples_split': list(range(1, 3, 1)),
   'min_samples_leaf': list(range(1, 3, 1))
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, n_jobs= -1, cv=9)
grid.fit(X_train, y_train)


print(' ')
print('###########################')
print('EXPERIMENTO RF')
print('Acurácia')

print(grid.best_params_) 
y_pred = grid.predict(X_test)
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)

##############################################
# Plota gráfico
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.xlabel('Previsto')
# plt.ylabel('Corretos')
# plt.title('Matrix de confusão - RandomForestClassifier')
# class_names = np.unique(y_test)
# tick_marks = np.arange(len(class_names))
# plt.xticks(tick_marks, class_names, rotation=45)
# plt.yticks(tick_marks, class_names)
#plt.show()

print(conf_matrix)


##############################################
# Predição de Novos Casos

# Salva o modelo e o scaler
joblib.dump(grid.best_estimator_, "modelo_treinado.pkl")
joblib.dump(scaler, "scaler_treinado.pkl")

# Caminhos para os arquivos
modelo_path = "modelo_treinado.pkl"
scaler_path = "scaler_treinado.pkl"
dados_novos_path = "/Users/jaimewojciechowski/Dropbox/Jaime/AA-UFPR/EspecializacaoIAA2026/Praticas Python/8 - Imposto de Renda/IR - Novos Casos - Para Python.csv"  # CSV SEM a variável alvo

# Carrega o modelo e o scaler
modelo = joblib.load(modelo_path)
scaler = joblib.load(scaler_path)

# Lê o novo arquivo CSV sem a variável alvo
dados_novos = pd.read_csv(dados_novos_path)

# Transforma as variáveis discretas em 0,1,2
dados_novos['rest'] = le.fit_transform(dados_novos['rest'])
dados_novos['ecivil'] = le.fit_transform(dados_novos['ecivil'])

# print(' ')
# print('###########################################')
# print('Transformou as variáveis discretas em 0,1,2 em dados_novos')
# print(dados_novos.head())

# Aplica a mesma padronização dos dados
#dados_novos_scaled = scaler.transform(dados_novos)

# Faz a predição
#predicoes = modelo.predict(dados_novos_scaled)
predicoes = modelo.predict(dados_novos)

# Mostra os resultados
print("Predições:")
print(predicoes)

# Salva as predições no mesmo DataFrame
dados_novos['predicao'] = predicoes

# Exporta para novo CSV
dados_novos.to_csv("/Users/jaimewojciechowski/Dropbox/Jaime/AA-UFPR/EspecializacaoIAA2026/Praticas Python/8 - Imposto de Renda/IR - Novos Casos - Predicoes em Python RF.csv", index=False)
print("\nPredições salvas em 'IR - Novos Casos - Predicoes em Python RF.csv'")


