import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, jaccard_score, cohen_kappa_score, \
    hamming_loss

from sklearn.preprocessing import MinMaxScaler

import joblib
#%%
# Defina a semente para reprodutibilidade
random_seed = 202526
np.random.seed(random_seed)

scaler = MinMaxScaler()
##############################################
# Abre o arquivo e mostra o conteúdo

df = pd.read_csv(r'C:\Users\lcast\OneDrive\Documents\Especialização UFPR\IAA08 - Aprendizado de Máquina\aprendizadoDeMaquina\6Veiculos\dados\veiculos.csv', sep=',')
df = df.drop('a', axis = 1)

print(' ')
print('###################')
print('Conteúdo do arquivo - Veículos')
print(df.head())
#%%
##############################################
# EXPERIMENTO - Separa as bases
y = df['tipo']
X = df.drop('tipo', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 13)

columns = X_train.columns
# Assim evita data leakage do scaler para os dados de teste
X_train = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns=columns)

X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test, columns=columns)
#%%
##############################################
# EXPERIMENTO - Trinamento com HOLD-OUT
mlp = MLPClassifier(
    hidden_layer_sizes=(100,),  # Uma camada oculta com 100 neurônios
    max_iter=1000,
    alpha=0.0001,
    solver='adam',
    random_state=42
)
mlp.fit(X_train, y_train)
#%%
# 6. Fazer previsões nos dados de teste
y_pred = mlp.predict(X_test)
#%%
# 7. Avaliar o desempenho HOLD-OUT
print('###########################')
print('EXPERIMENTO RNA - Veiculos - HOLD-OUT')

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
"""
Accuracy: 0.7755905511811023
Jaccard Index: [0.92857143 0.35802469 0.46601942 0.96491228]
Cohen's Kappa: 0.6994145247685088
Hamming Loss: 0.22440944881889763
Classification Report:
               precision    recall  f1-score   support
         bus       0.97      0.96      0.96        68
        opel       0.59      0.48      0.53        61
        saab       0.59      0.69      0.64        70
         van       0.96      1.00      0.98        55
    accuracy                           0.78       254
   macro avg       0.78      0.78      0.78       254
weighted avg       0.77      0.78      0.77       254
"""
#%%
conf_matrix = confusion_matrix(y_test, y_pred)
print(' ')
print('RNA - Veiculos - Matriz de Confusão - HOLD OUT')
print(conf_matrix)
"""
RNA - Veiculos - Matriz de Confusão - HOLD OUT
[[65  0  1  2]
 [ 0 30 31  0]
 [ 2 20 48  0]
 [ 0  0  0 55]]
"""
#%%
##############################################
# Predição de Novos Casos
# Salva o modelo e o scaler
joblib.dump(mlp, "modelo_treinado_rna_hold_out.pkl")
joblib.dump(scaler, "scaler_treinado_rna_hold_out.pkl")

# Caminhos para os arquivos
modelo_path = "modelo_treinado_rna_hold_out.pkl"
scaler_path = "scaler_treinado_rna_hold_out.pkl"
dados_novos_path = r"C:\Users\lcast\OneDrive\Documents\Especialização UFPR\IAA08 - Aprendizado de Máquina\aprendizadoDeMaquina\6Veiculos\dados\veiculos_novos_casos.csv"  # CSV SEM a variável alvo
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
#%%
# Mostra os resultados
print("Predições:")
print(predicoes)

# Salva as predições no mesmo DataFrame
dados_novos['predicao'] = predicoes

# Exporta para novo CSV
dados_novos.to_csv("Veiculos - Novos Casos - Predicoes em Python RNA HOLD OUT.csv", index=False)
print("\nPredições salvas em 'Veiculos - Novos Casos - Predicoes em Python RNA HOLD OUT.csv'")
"""
Predições:
['van' 'van' 'saab']
"""



