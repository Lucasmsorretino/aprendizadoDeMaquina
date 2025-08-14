import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

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

print(' ')
print('###################')
print('Conteúdo do arquivo - Diabetes')
print(df.head())
#%%
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
grid = GridSearchCV(MLPClassifier(), param_grid, n_jobs= -1)
grid.fit(X_train, y_train)
#%%
print(' ')
print('###########################')
print('EXPERIMENTO RNA - Diabetes')
print(' ')
print('Melhores parâmetros:')
print(' ')

print(grid.best_params_)
"""
Melhores parâmetros:
 
{'activation': 'relu', 'alpha': 0.0002, 'hidden_layer_sizes': (50, 50), 'learning_rate': 'constant', 'max_iter': 500, 'random_state': 5, 'solver': 'adam'}
"""
#%%
y_pred = grid.predict(X_test)

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
Accuracy: 0.7619047619047619
Jaccard Index: [0.68390805 0.50892857]
Cohen's Kappa: 0.48710185297323483
Hamming Loss: 0.23809523809523808
Classification Report:
               precision    recall  f1-score   support
         neg       0.80      0.83      0.81       144
         pos       0.70      0.66      0.67        87
    accuracy                           0.76       231
   macro avg       0.75      0.74      0.74       231
weighted avg       0.76      0.76      0.76       231
"""
#%%

conf_matrix = confusion_matrix(y_test, y_pred)
print(' ')
print('RNA - Diabetes - Matriz de Confusão')
print(conf_matrix)
"""
RNA - Diabetes - Matriz de Confusão
[[119  25]
 [ 30  57]]
"""
#%%
##############################################
# Predição de Novos Casos
#

# Salva o modelo e o scaler
joblib.dump(grid.best_estimator_, "modelo_treinado_rna_cv.pkl")
joblib.dump(scaler, "scaler_treinado_rna_cv.pkl")

# Caminhos para os arquivos
modelo_path = "modelo_treinado_rna_cv.pkl"
scaler_path = "scaler_treinado_rna_cv.pkl"
dados_novos_path = r'C:\Users\lcast\OneDrive\Documents\Especialização UFPR\IAA08 - Aprendizado de Máquina\aprendizadoDeMaquina\10Diabetes\dados\diabetes_novos_casos.csv'
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
dados_novos.to_csv("Diabetes - Novos Casos - Predicoes em Python RNA CV.csv", index=False)
print("\nPredições salvas em 'Diabetes - Novos Casos - Predicoes em Python RNA CV.csv'")
"""
Predições:
['neg' 'neg' 'pos']
"""