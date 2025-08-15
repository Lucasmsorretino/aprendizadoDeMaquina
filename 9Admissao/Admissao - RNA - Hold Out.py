import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import matplotlib.pyplot as plt
import joblib
#%%
# Defina as sementes para reprodutibilidade
random_seed = 202526
np.random.seed(random_seed)

plt.rcParams["figure.figsize"] = [22,8]
le = preprocessing.LabelEncoder()

##############################################
# Abre o arquivo e mostra o conteúdo
df = pd.read_csv(r'C:\Users\lcast\OneDrive\Documents\Especialização UFPR\IAA08 - Aprendizado de Máquina\aprendizadoDeMaquina\9Admissao\dados\admissao.csv', sep=',')
df = df.drop('num', axis = 1)

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
y = df['ChanceOfAdmit ']
X = df.drop('ChanceOfAdmit ', axis = 1)

# O escalonamento manual foi removido daqui para evitar data leakage.
# O Pipeline cuidará disso da maneira correta.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 9)

#%%
##############################################
# EXPERIMENTO RNA
# EXPERIMENTO - Treinamento com HOLD-OUT
print(' ')
print('###########################')
print('EXPERIMENTO RNA - Alunos - HOLD OUT')
print(' ')

# O Pipeline aplica o MinMaxScaler (apenas nos dados de treino) e depois treina o MLP.
pipeline = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(100,), activation='logistic', max_iter=1000, random_state=42)
)
mlp = pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

params = {"hidden_layer_sizes": 100, "activation":'logistic', "max_iter": 1000, "random_state": 42}
metrics_model = get_regression_metrics(y_test, y_pred,"MLPRegressor", params)
results.append(metrics_model)
#%%
print(' ')
print('###########################')
print('EXPERIMENTO RNA - Admissão')
print(' ')
print('parâmetros:')
print(' ')
print(params)
print('resultados:')
print(results)
"""
parâmetros:
 
{'hidden_layer_sizes': 100, activation: 'logistic', 'max_iter': 1000, 'random_state': 42}
resultados:
[{'MODELO': 'MLPRegressor', 'MAE': 0.0751404127636066, 'MSE': 0.00825629965349675, 'RMSE': np.float64(0.09086418245654747), 'R2': 0.5099904349160373, 'MAPE': np.float64(11.468825921254592), 'MedAE': 0.0665369954050547, 'PearsonR': np.float64(0.8970037458998672), 'syx': np.float64(0.007625152805440369), 'params': {'hidden_layer_sizes': 100, activation: 'logistic', 'max_iter': 1000, 'random_state': 42}}]
"""
#%%
##############################################
# Predição de Novos Casos

# Salva o pipeline inteiro, que contém tanto o scaler quanto o modelo já treinados.
pipeline_path = "pipeline_admissao_rna_hold_out.pkl"
joblib.dump(mlp, pipeline_path)
print(f"\nPipeline completo salvo em: {pipeline_path}")

# Caminho para os novos dados
dados_novos_path = r'C:\Users\lcast\OneDrive\Documents\Especialização UFPR\IAA08 - Aprendizado de Máquina\aprendizadoDeMaquina\9Admissao\dados\admissao_novos_casos.csv'
#%%
# Carrega o pipeline completo
pipeline_carregado = joblib.load(pipeline_path)

# Lê o novo arquivo CSV sem a variável alvo
dados_novos = pd.read_csv(dados_novos_path)

# Faz a predição usando o pipeline. Ele aplicará o scaler e o modelo automaticamente.
predicoes = pipeline_carregado.predict(dados_novos)


# Mostra os resultados
print("\nPredições para novos casos:")
print(predicoes)

# Salva as predições no mesmo DataFrame
dados_novos['predicao'] = predicoes

# Exporta para novo CSV
output_path = "predicoes_admissao_rna_hold_out.csv"
dados_novos.to_csv(output_path, index=False)
print(f"\nPredições salvas em '{output_path}'")
"""
Predições para novos casos:
[0.46255357 0.37504691 0.38921742]
"""