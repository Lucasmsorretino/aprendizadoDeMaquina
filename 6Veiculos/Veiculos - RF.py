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
random_seed = 202526
np.random.seed(random_seed)

scaler = MinMaxScaler()
plt.rcParams["figure.figsize"] = [22,8]
le = preprocessing.LabelEncoder()

##############################################
# Abre o arquivo e mostra o conteúdo

df = pd.read_csv('VeiculosDados.csv',sep=',')
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = random_seed)
#X.head()

rfSemCV = RandomForestClassifier(n_estimators=110, max_depth=10, random_state=random_seed)
rfSemCV.fit(X_train, y_train)

y_predSemCV = rfSemCV.predict(X_test)

accuracySemCV = accuracy_score(y_test, y_predSemCV)
class_reportSemCV = classification_report(y_test, y_predSemCV)
jaccardSemCV = jaccard_score(y_test, y_predSemCV, average=None)
cohen_kappaSemCV = cohen_kappa_score(y_test, y_predSemCV)
hammingSemCV = hamming_loss(y_test, y_predSemCV)
##############################################
# EXPERIMENTO RF
# Este Grid demora 17 minutos rodando no Mac
# param_grid = {
#     'n_estimators': list(range(10, 200, 10)),
#     'max_depth': list(range(1, 50, 10)),
#     'min_samples_split': list(range(1, 10, 1)),
#     'min_samples_leaf': list(range(1, 10, 1))
# }
param_grid = {
    'n_estimators': list(range(100, 120, 10)),
    'max_depth': list(range(1, 20, 10)),
    'min_samples_split': list(range(1, 3, 1)),
    'min_samples_leaf': list(range(1, 3, 1))
}

grid = GridSearchCV(RandomForestClassifier(random_state=random_seed), param_grid, n_jobs= -1, cv=9)
grid.fit(X_train, y_train)



# Imprimindo as métricas sem Cross Validations
print(' ')
print('Métricas sem Cross Validation')
print("Accuracy:", accuracySemCV)
print("Jaccard Index:", jaccardSemCV)
print("Cohen's Kappa:", cohen_kappaSemCV)
print("Hamming Loss:", hammingSemCV)
print("Classification Report:\n", class_reportSemCV)
conf_matrixSemCV = confusion_matrix(y_test, y_predSemCV)
print(' ')

print(' ')
print('###########################')

print('Melhores parâmetros GridSearchCV:')
print(' ')

print(grid.best_params_) 
y_pred = grid.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
jaccard = jaccard_score(y_test, y_pred, average=None)
cohen_kappa = cohen_kappa_score(y_test, y_pred)
hamming = hamming_loss(y_test, y_pred)

# Imprimindo as métricas com Cross Validations
print('Métricas com Cross Validation')
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
print('RF - Veiculos - Matriz de Confusão sem Cross Validation')
print(conf_matrixSemCV)

print(' ')
print('RF - Veiculos - Matriz de Confusão com Cross Validation')
print(conf_matrix)

##############################################
# Predição de Novos Casos
#
# Salvando modelo e scaler
joblib.dump(grid.best_estimator_, "rf_modelo_treinado.pkl")
joblib.dump(scaler, "rf_scaler_treinado.pkl")

# Arquivos
modelo_path = "rf_modelo_treinado.pkl"
scaler_path = "rf_scaler_treinado.pkl"
dados_novos_path = "VeiculosNovos CasosPara Python.csv"  # sem a variável alvo

# Carregar modelo e scaler
modelo = joblib.load(modelo_path)
scaler = joblib.load(scaler_path)

# Carregar dados novos
dados_novos = pd.read_csv(dados_novos_path)

# Aplicar escala
dados_novos_scaled = scaler.transform(dados_novos)

# Fazer predições
predicoes = modelo.predict(dados_novos_scaled)

# Anexar resultados
dados_novos['predicao'] = predicoes
dados_novos.to_csv("Veiculos - Novos Casos - Predicoes em Python RF.csv", index=False)

# Exibir resultados
print("Predições realizadas com sucesso.")
print(dados_novos[['predicao']])
print("Arquivo salvo como 'Veiculos - Novos Casos - Predicoes em Python RF.csv'")
