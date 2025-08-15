import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error

from sklearn.preprocessing import MinMaxScaler

import joblib
#%%
# Defina as sementes para reprodutibilidade
random_seed = 202526
np.random.seed(random_seed)
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

# O escalonamento manual foi removido daqui para evitar data leakage.
# O Pipeline cuidará disso da maneira correta.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 9)

#%%
##############################################
# EXPERIMENTO SVM
# EXPERIMENTO - Treinamento com HOLD-OUT
print(' ')
print('###########################')
print('EXPERIMENTO SVM - Biomassa - HOLD OUT')
print(' ')

# O Pipeline aplica o MinMaxScaler (apenas nos dados de treino) e depois treina o MLP.
pipeline = make_pipeline(
    MinMaxScaler(),
    SVR(C=10, kernel='poly', max_iter=1000)
)
mlp = pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

params = {"C": 10, "kernel": 'poly', "max_iter": 1000 }
metrics_model = get_regression_metrics(y_test, y_pred,"SVC", params)
results.append(metrics_model)
#%%
print(' ')
print('###########################')
print('EXPERIMENTO SVM - Biomassa')
print(' ')
print('parâmetros:')
print(' ')
print(params)
print('resultados:')
print(results)
"""
{'C': 10, 'kernel': 'poly', 'max_iter': 1000}
resultados:
[{'MODELO': 'SVC', 'MAE': 70.16429652274589, 'MSE': 33568.70707471243, 'RMSE': np.float64(183.21764946290637), 'R2': 0.9134708949155026, 'MAPE': np.float64(56.092975598257574), 'MedAE': 20.54633697411831, 'PearsonR': np.float64(0.9737370172027915), 'syx': np.float64(19.756867241400663), 'params': {'C': 10, 'kernel': 'poly', 'max_iter': 1000}}]

"""
#%%
##############################################
# Predição de Novos Casos

# Salva o pipeline inteiro, que contém tanto o scaler quanto o modelo já treinados.
pipeline_path = "pipeline_biomassa_svm_hold_out.pkl"
joblib.dump(mlp, pipeline_path)
print(f"\nPipeline completo salvo em: {pipeline_path}")

# Caminho para os novos dados
dados_novos_path = r'C:\Users\lcast\OneDrive\Documents\Especialização UFPR\IAA08 - Aprendizado de Máquina\aprendizadoDeMaquina\5Biomassa\dados\biomassa_novos_casos.csv'
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
output_path = "dados/predicoes_biomassa_svm_hold_out.csv"
dados_novos.to_csv(output_path, index=False)
print(f"\nPredições salvas em '{output_path}'")
"""
Predições para novos casos:
[-14.97358762 -30.88368122  20.68622957  68.27722741]
"""