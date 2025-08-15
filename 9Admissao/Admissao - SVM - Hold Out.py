import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error

from sklearn.preprocessing import MinMaxScaler

import joblib
#%%
# Defina as sementes para reprodutibilidade
random_seed = 202526
np.random.seed(random_seed)

scaler = MinMaxScaler()
##############################################
# Abre o arquivo e mostra o conteúdo
df = pd.read_csv(r'C:\Users\lcast\OneDrive\Documents\Especialização UFPR\IAA08 - Aprendizado de Máquina\aprendizadoDeMaquina\9Admissao\dados\admissao.csv', sep=',')
df = df.drop('num', axis = 1)

results = []

print(' ')
print('###################')
print('Conteúdo do arquivo - Admissao')
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
X = df.drop('ChanceOfAdmit ', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 13)

columns = X_train.columns
# Assim evita data leakage do scaler para os dados de teste
X_train = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns=columns)

X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test, columns=columns)
#%%
##############################################
# EXPERIMENTO SVM
svr = SVR(C=100, kernel='linear', max_iter=5000)
svr.fit(X_train, y_train)
#%%
y_pred = svr.predict(X_test)
params = {"C": 100, "kernel": 'linear', "max_iter": 5000 }
metrics_model = get_regression_metrics(y_test, y_pred,"SVM", params)
results.append(metrics_model)

#%%
print(' ')
print('###########################')
print('EXPERIMENTO SVM - Admissao')
print(' ')
print('parâmetros:')
print(' ')
print(params)
print('resultados:')
print(results)
"""
{'C': 100, 'kernel': 'linear', 'max_iter': 5000}
resultados:
[{'MODELO': 'SVM', 'MAE': 0.05292808501405733, 'MSE': 0.004691423511214409, 'RMSE': np.float64(0.0684939669694668), 'R2': 0.7501446153244905, 'MAPE': np.float64(8.26425060027035), 'MedAE': 0.044639025271354715, 'PearsonR': np.float64(0.9032502322667647), 'syx': np.float64(0.00574788602365657), 'params': {'C': 100, 'kernel': 'linear', 'max_iter': 5000}}]

"""
#%%
##############################################
# Predição de Novos Casos

# Salva o modelo e o scaler
joblib.dump(svr, "modelo_treinado_svm_hold_out.pkl")
joblib.dump(scaler, "scaler_treinado_svm_hold_out.pkl")


# Caminhos para os arquivos
modelo_path = "modelo_treinado_svm_hold_out.pkl"
scaler_path = "scaler_treinado_svm_hold_out.pkl"
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
dados_novos.to_csv("Admissao - Novos Casos - Predicoes em Python SVM HOLD OUT.csv", index=False)
print("\nPredições salvas em 'Admissao - Novos Casos - Predicoes em Python SVM HOLD OUT.csv'")
"""
Predições:
[ 0.61709911 -0.07444339  0.533709  ]
"""