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

from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error, mean_squared_error

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
df = pd.read_csv('/Users/jaimewojciechowski/Dropbox/Jaime/AA-UFPR/EspecializacaoIAA2026/Praticas Python/3 - Alunos/Alunos - Dados.csv',sep=',')
#df = df.drop('num', axis = 1)
results = []
#df.info()

print(' ')
print('###################')
print('Conteúdo do arquivo - Alunos')
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
# Discretiza as variáveis com domínio
df['school'] = le.fit_transform(df['school'])
df['address'] = le.fit_transform(df['address'])
df['famsize'] = le.fit_transform(df['famsize'])
df['Pstatus'] = le.fit_transform(df['Pstatus'])
df['Mjob'] = le.fit_transform(df['Mjob'])
df['Fjob'] = le.fit_transform(df['Fjob'])
df['reason'] = le.fit_transform(df['reason'])
df['guardian'] = le.fit_transform(df['guardian'])
df['schoolsup'] = le.fit_transform(df['schoolsup'])
df['famsup'] = le.fit_transform(df['famsup'])
df['paid'] = le.fit_transform(df['paid'])
df['activities'] = le.fit_transform(df['activities'])
df['nursery'] = le.fit_transform(df['nursery'])
df['higher'] = le.fit_transform(df['higher'])
df['internet'] = le.fit_transform(df['internet'])
df['romantic'] = le.fit_transform(df['romantic'])



##############################################
# Mostra gráfico de correlação
# corr = df.corr(method='pearson')
# sns.heatmap(corr,cmap='seismic',annot=True, fmt=".2f")
# plt.show()



##############################################
# EXPERIMENTO - Separa as bases
y = df['G3']
#y = le.fit_transform(df['tipo'])
X = df.drop('G3', axis = 1)

columns = list(X.columns)
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=columns)
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 9)

#X.head()


##############################################
# EXPERIMENTO RNA
# Este grid demora quase 1 hora pra executar no mac
# param_grid = {
#     'hidden_layer_sizes': [(100,),(50,50,), (25,25,25,25,), (20,20,20,20,20,)],#
#     'max_iter': range(100,1000,100),#
#     'activation': ['tanh', 'relu'],
#     'solver': ['lbfgs', 'sgd', 'adam'],
#     'alpha': [0.0001, 0.0002, 0.0003],
#     'learning_rate': ['constant','adaptive', 'invscaling'],
#     'random_state': [3,5,7,9]#
# }
# param_grid = {
#     'hidden_layer_sizes': [(100,),(50,50,)],#
#     'max_iter': [500],#
#     'activation': ['tanh', 'relu'],
#     'solver': ['sgd', 'adam'],
#     'alpha': [0.0001, 0.0002],
#     'learning_rate': ['constant','adaptive'],
#     'random_state': [3,5]#
# }

param_grid = {
    'hidden_layer_sizes': [(100,)],#
    'max_iter': [500],#
    'activation': ['tanh'],
    'solver': ['sgd'],
    'alpha': [0.0001],
    'learning_rate': ['constant'],
    'random_state': [3]#
}
grid = GridSearchCV(MLPRegressor(), param_grid, refit=True, n_jobs=-1)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
metrics_model = get_regression_metrics(y_test, y_pred,"MLPRegressor", grid.best_estimator_)
results.append(metrics_model)


print(' ')
print('###########################')
print('EXPERIMENTO RNA - Alunos')
print(' ')
print('Melhores parâmetros:')
print(' ')

best_params = grid.best_estimator_
print(f"Melhores parametros: {best_params}")

mae = mean_absolute_error(y_test, y_pred)
print("Erro Médio Absoluto:", mae)

mse = mean_squared_error(y_test, y_pred)
print("Erro Quadrático Médio:", mse)

rmse = np.sqrt(mse)
print("Raiz do Erro Quadrático Médio:", rmse)

r2 = r2_score(y_test, y_pred)
print("R-quadrado:", r2)

mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print("Erro Médio Absoluto Percentual:", mape)

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
dados_novos_path = "/Users/jaimewojciechowski/Dropbox/Jaime/AA-UFPR/EspecializacaoIAA2026/Praticas Python/3 - Alunos/Alunos - Novos Casos - Para Python.csv"  # CSV SEM a variável alvo

# Carrega o modelo e o scaler
modelo = joblib.load(modelo_path)
scaler = joblib.load(scaler_path)

# Lê o novo arquivo CSV sem a variável alvo
dados_novos = pd.read_csv(dados_novos_path)

##############################################
# Discretiza as variáveis com domínio
dados_novos['school'] = le.fit_transform(dados_novos['school'])
dados_novos['address'] = le.fit_transform(dados_novos['address'])
dados_novos['famsize'] = le.fit_transform(dados_novos['famsize'])
dados_novos['Pstatus'] = le.fit_transform(dados_novos['Pstatus'])
dados_novos['Mjob'] = le.fit_transform(dados_novos['Mjob'])
dados_novos['Fjob'] = le.fit_transform(dados_novos['Fjob'])
dados_novos['reason'] = le.fit_transform(dados_novos['reason'])
dados_novos['guardian'] = le.fit_transform(dados_novos['guardian'])
dados_novos['schoolsup'] = le.fit_transform(dados_novos['schoolsup'])
dados_novos['famsup'] = le.fit_transform(dados_novos['famsup'])
dados_novos['paid'] = le.fit_transform(dados_novos['paid'])
dados_novos['activities'] = le.fit_transform(dados_novos['activities'])
dados_novos['nursery'] = le.fit_transform(dados_novos['nursery'])
dados_novos['higher'] = le.fit_transform(dados_novos['higher'])
dados_novos['internet'] = le.fit_transform(dados_novos['internet'])
dados_novos['romantic'] = le.fit_transform(dados_novos['romantic'])



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
dados_novos.to_csv("/Users/jaimewojciechowski/Dropbox/Jaime/AA-UFPR/EspecializacaoIAA2026/Praticas Python/3 - Alunos/Alunos - Novos Casos - Predicoes em Python RNA.csv", index=False)
print("\nAlunos - Novos Casos - Predicoes em Python RNA.csv'")


