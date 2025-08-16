import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error

from sklearn.preprocessing import StandardScaler

import joblib
#%%
# Defina as sementes para reprodutibilidade
random_seed = 202526
np.random.seed(random_seed)

scaler = StandardScaler()
##############################################
# Abre o arquivo e mostra o conteúdo
df = pd.read_csv(r'C:\Users\lcast\OneDrive\Documents\Especialização UFPR\IAA08 - Aprendizado de Máquina\aprendizadoDeMaquina\5Biomassa\dados\biomassa.csv', sep=',')

results = []

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
y = df['biomassa']
X = df.drop('biomassa', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 13)

columns = X_train.columns
# Assim evita data leakage do scaler para os dados de teste
X_train = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns=columns)

X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test, columns=columns)
#%%
##############################################
# EXPERIMENTO RNA
# EXPERIMENTO - Treinamento com HOLD-OUT
print(' ')
print('###########################')
print('EXPERIMENTO RNA - Biomassa - HOLD OUT')
print(' ')

mlp = MLPRegressor(hidden_layer_sizes=(35,5), max_iter=2000, random_state=42)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

params = {"hidden_layer_sizes": (35,5), "max_iter": 2000, "random_state": 42}
metrics_model = get_regression_metrics(y_test, y_pred,"MLPRegressor", params)
results.append(metrics_model)
#%%
print(' ')
print('###########################')
print('EXPERIMENTO RNA - Biomassa')
print(' ')
print('parâmetros:')
print(' ')
print(params)
print('resultados:')
print(results)
"""
parâmetros:
 
{'hidden_layer_sizes': (35, 5), 'max_iter': 2000, 'random_state': 42}
resultados:
[{'MODELO': 'MLPRegressor', 'MAE': 241.83435190394235, 'MSE': 306962.7940354382, 'RMSE': np.float64(554.04223127433), 'R2': 0.8566376632617969, 'MAPE': np.float64(93.05585371775469), 'MedAE': 63.55174189634387, 'PearsonR': np.float64(0.9258854866269596), 'syx': np.float64(59.743910270131806), 'params': {'hidden_layer_sizes': (35, 5), 'max_iter': 2000, 'random_state': 42}}]

"""
#%%
##############################################
# Predição de Novos Casos

# Salva o modelo e o scaler
joblib.dump(mlp, "modelo_treinado_rna_cv.pkl")
joblib.dump(scaler, "scaler_treinado_rna_cv.pkl")


# Caminhos para os arquivos
modelo_path = "modelo_treinado_rna_cv.pkl"
scaler_path = "scaler_treinado_rna_cv.pkl"
dados_novos_path = r'C:\Users\lcast\OneDrive\Documents\Especialização UFPR\IAA08 - Aprendizado de Máquina\aprendizadoDeMaquina\5Biomassa\dados\biomassa_novos_casos.csv'
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
dados_novos.to_csv("Biomassa - Novos Casos - Predicoes em Python RNA CV.csv", index=False)
print("\nPredições salvas em 'Biomassa - Novos Casos - Predicoes em Python RNA CV.csv'")
"""
Predições:
[0.7732581 0.7732581 0.7732581 0.7732581]
"""