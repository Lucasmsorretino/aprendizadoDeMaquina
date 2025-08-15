import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

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
print('Conteúdo do arquivo - Veículos')
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
# Demorado
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}
grid = GridSearchCV(SVC(), param_grid, n_jobs= -1)
grid.fit(X_train, y_train)
#%%
print(' ')
print('###########################')
print('EXPERIMENTO SVM - Diabetes')
print(' ')
print('Melhores parâmetros:')
print(' ')

print(grid.best_params_)
"""
Melhores parâmetros:
 
{'C': 10, 'gamma': 'scale', 'kernel': 'linear'}
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
Accuracy: 0.7402597402597403
Jaccard Index: [0.67391304 0.43925234]
Cohen's Kappa: 0.4204716507777221
Hamming Loss: 0.2597402597402597
Classification Report:
               precision    recall  f1-score   support
         neg       0.76      0.86      0.81       144
         pos       0.70      0.54      0.61        87
    accuracy                           0.74       231
   macro avg       0.73      0.70      0.71       231
weighted avg       0.74      0.74      0.73       231
"""
#%%
conf_matrix = confusion_matrix(y_test, y_pred)
print(' ')
print('SVM - Diabetes - Matriz de Confusão')
print(conf_matrix)
"""
SVM - Diabetes - Matriz de Confusão
[[124  20]
 [ 40  47]]
"""
#%%
##############################################
# Predição de Novos Casos
#

# Salvando o modelo e scaler
joblib.dump(grid.best_estimator_, "modelo_treinado_svm_cv.pkl")
joblib.dump(scaler, "scaler_treinado_svm_cv.pkl")

# Caminhos dos arquivos
modelo_path = "modelo_treinado_svm_cv.pkl"
scaler_path = "scaler_treinado_svm_cv.pkl"
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
dados_novos.to_csv("Diabetes - Novos Casos - Predicoes em Python SVM CV.csv", index=False)
print("\nArquivo salvo como 'Diabetes - Novos Casos - Predicoes em Python SVM CV.csv'")
"""
Predições:
['neg' 'neg' 'pos']
"""