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

df = pd.read_csv(r'C:\Users\lcast\OneDrive\Documents\Especialização UFPR\IAA08 - Aprendizado de Máquina\aprendizadoDeMaquina\10Diabetes\dados\diabetes.csv', sep=',')
df = df.drop('num', axis = 1)

df.head()

print(' ')
print('###################')
print('Conteúdo do arquivo - Diabetes')
print(df.head())
#%%
##############################################
# EXPERIMENTO - Separa as bases
y = df['diabetes']
X = df.drop('diabetes', axis = 1)
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
print('EXPERIMENTO SVM - Diabetes - HOLD-OUT')

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
Accuracy: 0.7489177489177489
Jaccard Index: [0.68648649 0.44230769]
Cohen's Kappa: 0.4343971631205674
Hamming Loss: 0.2510822510822511
Classification Report:
               precision    recall  f1-score   support
         neg       0.76      0.88      0.81       144
         pos       0.73      0.53      0.61        87
    accuracy                           0.75       231
   macro avg       0.74      0.71      0.71       231
weighted avg       0.75      0.75      0.74       231
"""
#%%
conf_matrix = confusion_matrix(y_test, y_pred)
print(' ')
print('SVM - Diabetes - Matriz de Confusão')
print(conf_matrix)
"""
SVM - Diabetes - Matriz de Confusão
[[127  17]
 [ 41  46]]
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
dados_novos_path = r"C:\Users\lcast\OneDrive\Documents\Especialização UFPR\IAA08 - Aprendizado de Máquina\aprendizadoDeMaquina\10Diabetes\dados\diabetes_novos_casos.csv"  # CSV SEM a variável alvo
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
dados_novos.to_csv("Diabetes - Novos Casos - Predicoes em Python SVM HOLD OUT.csv", index=False)
print("\nArquivo salvo como 'Diabetes - Novos Casos - Predicoes em Python SVM HOLD OUT.csv'")
"""
Predições:
['neg' 'neg' 'pos']
"""