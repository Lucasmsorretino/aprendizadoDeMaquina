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
grid = GridSearchCV(MLPClassifier(), param_grid, n_jobs= -1, cv=9)
grid.fit(X_train, y_train)
#%%
print(' ')
print('###########################')
print('EXPERIMENTO RNA - Veículos')
print(' ')
print('Melhores parâmetros:')
print(' ')
print(grid.best_params_)
# Melhores parâmetros:
#
# {'activation': 'relu', 'alpha': 0.0002, 'hidden_layer_sizes': (50, 50), 'learning_rate': 'constant', 'max_iter': 500, 'random_state': 3, 'solver': 'adam'}
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
Accuracy: 0.7913385826771654
Jaccard Index: [0.95652174 0.36708861 0.49514563 0.98214286]
Cohen's Kappa: 0.7203166226912929
Hamming Loss: 0.20866141732283464
Classification Report:
               precision    recall  f1-score   support
         bus       0.99      0.97      0.98        68
        opel       0.62      0.48      0.54        61
        saab       0.61      0.73      0.66        70
         van       0.98      1.00      0.99        55
    accuracy                           0.79       254
   macro avg       0.80      0.79      0.79       254
weighted avg       0.79      0.79      0.79       254

"""

#%%
conf_matrix = confusion_matrix(y_test, y_pred)
print(' ')
print('RNA - Veiculos - Matriz de Confusão')
print(conf_matrix)
"""
RNA - Veiculos - Matriz de Confusão
[[66  0  1  1]
 [ 0 29 32  0]
 [ 1 18 51  0]
 [ 0  0  0 55]]
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
dados_novos.to_csv("Veiculos - Novos Casos - Predicoes em Python RNA CV.csv", index=False)
print("\nPredições salvas em 'Veiculos - Novos Casos - Predicoes em Python RNA CV.csv'")
"""
Predições:
['van' 'van' 'saab']
"""