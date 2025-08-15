import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, jaccard_score, cohen_kappa_score, \
    hamming_loss

from sklearn.preprocessing import MinMaxScaler

import joblib
#%%
# Definada as sementes para reprodutibilidade
random_seed = 202526
np.random.seed(random_seed)

scaler = MinMaxScaler()
le = preprocessing.LabelEncoder()
##############################################
# Abre o arquivo e mostra o conteúdo

df = pd.read_csv(r'C:\Users\lcast\OneDrive\Documents\Especialização UFPR\IAA08 - Aprendizado de Máquina\aprendizadoDeMaquina\6Veiculos\dados\veiculos.csv', sep=',')
df = df.drop('a', axis = 1)

df.head()

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
# EXPERIMENTO SVM

svm = SVC(
    C=1.0,
    kernel='rbf',
    gamma='scale',
    random_state=42
)
svm.fit(X_train, y_train)
#%%
# 6. Fazer previsões nos dados de teste
y_pred = svm.predict(X_test)
#%%
# 7. Avaliar o desempenho HOLD-OUT
print('###########################')
print('EXPERIMENTO SVM - Veiculos - HOLD-OUT')

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
Accuracy: 0.7125984251968503
Jaccard Index: [0.87837838 0.2625     0.38095238 0.80882353]
Cohen's Kappa: 0.6157735504993991
Hamming Loss: 0.2874015748031496
Classification Report:
               precision    recall  f1-score   support
         bus       0.92      0.96      0.94        68
        opel       0.53      0.34      0.42        61
        saab       0.53      0.57      0.55        70
         van       0.81      1.00      0.89        55
    accuracy                           0.71       254
   macro avg       0.70      0.72      0.70       254
weighted avg       0.69      0.71      0.70       254
"""
#%%
conf_matrix = confusion_matrix(y_test, y_pred)
print(' ')
print('SVM - Veiculos - Matriz de Confusão')
print(conf_matrix)
"""
SVM - Veiculos - Matriz de Confusão
[[65  0  1  2]
 [ 2 21 34  4]
 [ 4 19 40  7]
 [ 0  0  0 55]]
"""
#%%
##############################################
# Predição de Novos Casos
#

# Salvando o modelo e scaler
joblib.dump(svm, "modelo_treinado_svm_hold_out.pkl")
joblib.dump(scaler, "scaler_treinado_svm_hold_out.pkl")

# Caminhos dos arquivos
modelo_path = "modelo_treinado_svm_hold_out.pkl"
scaler_path = "scaler_treinado_svm_hold_out.pkl"
dados_novos_path = r"C:\Users\lcast\OneDrive\Documents\Especialização UFPR\IAA08 - Aprendizado de Máquina\aprendizadoDeMaquina\6Veiculos\dados\veiculos_novos_casos.csv"  # CSV SEM a variável alvo
#%%
# Carrega o modelo e o scaler
modelo = joblib.load(modelo_path)
scaler = joblib.load(scaler_path)

# Carrega os novos dados
dados_novos = pd.read_csv(dados_novos_path)

# Padroniza os dados
dados_novos_scaled = scaler.transform(dados_novos)

# Faz as predições
predicoes = modelo.predict(dados_novos_scaled)
#%%
# Mostra os resultados
print("Predições:")
print(predicoes)

# Salva os resultados
dados_novos['predicao'] = predicoes
dados_novos.to_csv("Veiculos - Novos Casos - Predicoes em Python SVM HOLD OUT.csv", index=False)
print("\nArquivo salvo como 'Veiculos - Novos Casos - Predicoes em Python SVM HOLD OUT.csv'")
"""
Predições:
['van' 'van' 'saab']
"""