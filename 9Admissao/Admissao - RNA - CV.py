import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn import preprocessing
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import joblib
#%%
# Definada as sementes para reprodutibilidade
random_seed = 202526
np.random.seed(random_seed)

scaler = MinMaxScaler()
plt.rcParams["figure.figsize"] = [22,8]
le = preprocessing.LabelEncoder()

##############################################
# Abre o arquivo e mostra o conteúdo
df = pd.read_csv(r'C:\Users\lcast\OneDrive\Documents\Especialização UFPR\IAA08 - Aprendizado de Máquina\aprendizadoDeMaquina\9Admissao\dados\admissao.csv', sep=',')
df = df.drop('num', axis = 1)

results = []

#df.info()

print(' ')
print('###################')
print('Conteúdo do arquivo - Admissão')
print(df.head())
#%%
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
        "PearsonR": np.corrcoef(y_test, y_pred)[0, 1],
        "syx": np.sqrt(mean_squared_error(y_test,y_pred)/ (len(y_test) - len(X_train.columns) - 1)),
        "params":params
    }
    return metrics
#%%
# EXPERIMENTO - Separa as bases
y = df['ChanceOfAdmit ']
#y = le.fit_transform(df['ChanceOfAdmit'])
X = df.drop('ChanceOfAdmit ', axis = 1)

columns = list(X.columns)
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=columns)
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 9)

#X.head()
#%%
##############################################
# EXPERIMENTO RNA
param_grid = {
    'hidden_layer_sizes': [(100,),(50,50,)],#
    'max_iter': [500],#
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.0002],
    'learning_rate': ['constant','adaptive'],
    'random_state': [3,5]#
}

grid = GridSearchCV(MLPRegressor(), param_grid, refit=True, n_jobs=-1)

grid.fit(X_train, y_train)
#%%
y_pred = grid.predict(X_test)
metrics_model = get_regression_metrics(y_test, y_pred,"MLPRegressor",grid.best_estimator_)
results.append(metrics_model)

#%%
print(' ')
print('###########################')
print('EXPERIMENTO RNA - Admissão')
print(' ')
print('Melhores parâmetros:')
print(' ')

best_params = grid.best_estimator_
print(f"Melhores parametros: {best_params}")
print(results)
"""
Melhores parametros: MLPRegressor(alpha=0.0002, max_iter=500, random_state=3)
[{'MODELO': 'MLPRegressor', 'MAE': 0.043012985024392644, 'MSE': 0.003173733036811219, 'RMSE': np.float64(0.0563358947458121), 'R2': 0.8116396436263283, 'MAPE': np.float64(6.352780562613799), 'MedAE': 0.0312006599960461, 'PearsonR': np.float64(0.9048692787018731), 'syx': np.float64(0.004727603267364991), 'params': MLPRegressor(alpha=0.0002, max_iter=500, random_state=3)}]
"""
#%%
##############################################
# Predição de Novos Casos

# Salva o modelo e o scaler
joblib.dump(grid.best_estimator_, "modelo_treinado_rna_cv.pkl")
joblib.dump(scaler, "scaler_treinado_rna_cv.pkl")


# Caminhos para os arquivos
modelo_path = "modelo_treinado_rna_cv.pkl"
scaler_path = "scaler_treinado_rna_cv.pkl"
dados_novos_path = r'C:\Users\lcast\OneDrive\Documents\Especialização UFPR\IAA08 - Aprendizado de Máquina\aprendizadoDeMaquina\9Admissao\dados\admissao_novos_casos.csv'
#%%
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
dados_novos.to_csv("Admissao - Novos Casos - Predicoes em Python RNA CV.csv", index=False)
print("\nPredições salvas em 'Admissao - Novos Casos - Predicoes em Python RNA CV.csv'")
"""
Predições:
[0.45806044 0.42210164 0.43925738]
"""