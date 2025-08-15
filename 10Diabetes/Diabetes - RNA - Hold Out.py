import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, jaccard_score, cohen_kappa_score, \
    hamming_loss

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import joblib
#%%
# Defina a semente para reprodutibilidade
random_seed = 202526
np.random.seed(random_seed)

scaler = MinMaxScaler()
plt.rcParams["figure.figsize"] = [22,8]
le = preprocessing.LabelEncoder()

##############################################
# Abre o arquivo e mostra o conteúdo

df = pd.read_csv(r'C:\Users\lcast\OneDrive\Documents\Especialização UFPR\IAA08 - Aprendizado de Máquina\aprendizadoDeMaquina\10Diabetes\dados\diabetes.csv', sep=',')
df = df.drop('num', axis = 1)

df.head()

print(' ')
print('###################')
print('Conteúdo do arquivo - Veículos')
print(df.head())
#%%
##############################################
# EXPERIMENTO - Separa as bases
y = df['diabetes']
X = df.drop('diabetes', axis = 1)

columns = list(X.columns)
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 13)
#X.head()
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
print('EXPERIMENTO RNA - Diabetes - HOLD-OUT')

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
EXPERIMENTO RNA - Diabetes - HOLD-OUT
Accuracy: 0.7402597402597403
Jaccard Index: [0.67032967 0.44954128]
Cohen's Kappa: 0.4259443339960238
Hamming Loss: 0.2597402597402597
Classification Report:
               precision    recall  f1-score   support
         neg       0.76      0.85      0.80       144
         pos       0.69      0.56      0.62        87
    accuracy                           0.74       231
   macro avg       0.73      0.71      0.71       231
weighted avg       0.74      0.74      0.73       231
"""
#%%
conf_matrix = confusion_matrix(y_test, y_pred)
print(' ')
print('RNA - Diabetes - Matriz de Confusão - HOLD OUT')
print(conf_matrix)
"""
RNA - Diabetes - Matriz de Confusão - HOLD OUT
[[122  22]
 [ 38  49]]
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
dados_novos_path = r"C:\Users\lcast\OneDrive\Documents\Especialização UFPR\IAA08 - Aprendizado de Máquina\aprendizadoDeMaquina\10Diabetes\dados\diabetes_novos_casos.csv"  # CSV SEM a variável alvo
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
dados_novos.to_csv("Diabetes - Novos Casos - Predicoes em Python RNA HOLD OUT.csv", index=False)
print("\nPredições salvas em 'Diabetes - Novos Casos - Predicoes em Python RNA HOLD OUT.csv'")
"""
Predições:
['neg' 'neg' 'pos']
"""



