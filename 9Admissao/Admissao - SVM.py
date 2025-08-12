import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

#from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error

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
df = pd.read_csv('/Users/jaimewojciechowski/Dropbox/Jaime/AA-UFPR/EspecializacaoIAA2026/Praticas Python/9 - Admissao/Admissao - Dados.csv',sep=',')
df = df.drop('num', axis = 1)

results = []

#df.info()

print(' ')
print('###################')
print('Conteúdo do arquivo - Admissão')
print(df.head())

##############################################
# Define as métricas
def get_regression_metrics(y_test, y_pred, modelo,params):
    metrics = {
        "MODELO":modelo,
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
        "MAPE": np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
        "MedAE": median_absolute_error(y_test, y_pred),
        "params":params
    }
    return metrics


##############################################
# Mostra gráfico de correlação
# corr = df.corr(method='pearson')
# sns.heatmap(corr,cmap='seismic',annot=True, fmt=".3f")
# plt.show()

##############################################
# EXPERIMENTO - Separa as bases
y = df['ChanceOfAdmit ']
#y = le.fit_transform(df['ChanceOfAdmit'])
X = df.drop('ChanceOfAdmit ', axis = 1)

columns = list(X.columns)
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=columns)
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 9)

#X.head()



##############################################
# EXPERIMENTO - ???????
# x = sm.add_constant(X)
# model = sm.OLS(y, x.astype(float)).fit()
# print(model.summary())
# print('R2: ', model.rsquared)



##############################################
# EXPERIMENTO SVM
# Demorado
# param_grid = {
#     'C': range(1, 100, 10),
#     'kernel': ['linear', 'rbf'],
#     'gamma': ['scale', 'auto']
# }
param_grid = {
    'C': range(1, 10),
    'kernel': ['rbf'],
    'gamma': ['auto']
}
grid = GridSearchCV(SVR(), param_grid, refit=True, error_score='raise')
grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)
metrics_model = get_regression_metrics(y_test, y_pred,"SVR",grid.best_estimator_)
results.append(metrics_model)



print(' ')
print('###########################')
print('EXPERIMENTO SVM - Admissão')
print(' ')
print('Melhores parâmetros:')
print(' ')

best_params = grid.best_estimator_
print(f"Melhores parametros: {best_params}")

mae = mean_absolute_error(y_test, y_pred)
print("Erro Médio Absoluto (MAE)", mae)

mse = mean_squared_error(y_test, y_pred)
print("Erro Quadrático Médio (MSE)", mse)

rmse = np.sqrt(mse)
print("Raiz do Erro Quadrático Médio (RMSE)", rmse)

r2 = r2_score(y_test, y_pred)
print("R-quadrado (Coeficiente de Determinação):", r2)

mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print("Erro Médio Absoluto Percentual (MAPE):", mape)

medae = median_absolute_error(y_test, y_pred)
print("Erro Absoluto Mediano:", medae)


##############################################
# Predição de Novos Casos

# Salva o modelo e o scaler
joblib.dump(grid.best_estimator_, "modelo_treinado.pkl")
joblib.dump(scaler, "scaler_treinado.pkl")


# Caminhos para os arquivos
modelo_path = "modelo_treinado.pkl"
scaler_path = "scaler_treinado.pkl"
dados_novos_path = "/Users/jaimewojciechowski/Dropbox/Jaime/AA-UFPR/EspecializacaoIAA2026/Praticas Python/9 - Admissao/Admissao - Novos Casos - Para Python.csv"  # CSV SEM a variável alvo

# Carrega o modelo e o scaler
modelo = joblib.load(modelo_path)
scaler = joblib.load(scaler_path)

# Lê o novo arquivo CSV sem a variável alvo
dados_novos = pd.read_csv(dados_novos_path)

# Aplica a mesma padronização dos dados
dados_novos_scaled = scaler.transform(dados_novos)

# Faz a predição
predicoes = modelo.predict(dados_novos_scaled)

# Mostra os resultados
print("Predições:")
print(predicoes)

# Salva as predições no mesmo DataFrame
dados_novos['predicao'] = predicoes

# Exporta para novo CSV
dados_novos.to_csv("/Users/jaimewojciechowski/Dropbox/Jaime/AA-UFPR/EspecializacaoIAA2026/Praticas Python/9 - Admissao/Admissao - Novos Casos - Predicoes em Python SVM.csv", index=False)
print("\nPredições salvas em 'Admissao - Novos Casos - Predicoes em Python SVM.csv'")

