# Práticas em Python - Aprendizado de Máquina

Este repositório contém implementações práticas de algoritmos de aprendizado de máquina em Python, organizadas por diferentes casos de estudo.

## Estrutura do Projeto

O projeto está organizado em pastas numeradas, cada uma contendo um caso de estudo específico:

- `1Cancer de Mama/` - Classificação de câncer de mama
- `2Volume/` - Análise de volume
- `3Alunos/` - Classificação de dados de alunos
- `4Banco/` - Análise de dados bancários
- `5Biomassa/` - Análise de biomassa
- `6Veiculos/` - Classificação de veículos
- `7Previsao do Tempo/` - Previsão meteorológica
- `8Imposto de Renda/` - Análise de imposto de renda
- `9Admissao/` - Processos de admissão
- `10Diabetes/` - Classificação de diabetes
- `11AgrupamentoPraticas2Moveis/` - Agrupamento de móveis
- `12Regras de AssociacaoPraticas1Lista de Compras/` - Regras de associação

## Algoritmos Implementados

Para cada caso de estudo, são implementados os seguintes algoritmos:

- **KNN** (K-Nearest Neighbors)
- **Random Forest** (RF)
- **Support Vector Machine** (SVM)
- **Redes Neurais Artificiais** (RNA/MLP)
- **Agrupamento** (para casos específicos)
- **Regras de Associação** (para casos específicos)

## Configuração do Ambiente

### Pré-requisitos

- Python 3.12+
- pip

### Instalação

1. Clone o repositório
2. Navegue até a pasta do projeto
3. **Ative o ambiente virtual:**
   
   **Opção 1 - Script automático (recomendado):**
   ```bash
   # No Windows (CMD)
   ativar_ambiente.bat
   
   # No Windows (PowerShell)
   .\ativar_ambiente.ps1
   ```
   
   **Opção 2 - Manual:**
   ```bash
   # PowerShell
   venv\Scripts\Activate.ps1
   
   # CMD
   venv\Scripts\activate.bat
   ```

4. As dependências já estão instaladas no ambiente virtual

### Dependências

- numpy - Computação numérica
- pandas - Manipulação de dados
- scikit-learn - Algoritmos de ML
- matplotlib - Visualização básica
- seaborn - Visualização estatística
- joblib - Persistência de modelos
- statsmodels - Modelos estatísticos

## Como Usar

1. Navegue até a pasta do caso de estudo desejado
2. Execute o script Python correspondente ao algoritmo:
   ```bash
   python VeiculosKNN.py
   ```
3. Os resultados serão exibidos no console e arquivos de predição serão salvos

## Estrutura dos Arquivos

Cada pasta de caso de estudo contém:

- `*Dados.csv` - Dataset original
- `*KNN.py` - Implementação K-Nearest Neighbors
- `*RF.py` - Implementação Random Forest
- `*SVM.py` - Implementação Support Vector Machine
- `*RNA.py` - Implementação Redes Neurais
- `*Novos Casos - Para Python.csv` - Dados para predição
- `*Novos Casos - Predicoes em Python *.csv` - Resultados das predições

## Características dos Algoritmos

### Métricas Avaliadas

- Accuracy (Acurácia)
- Precision (Precisão)
- Recall (Revocação)
- F1-Score
- Jaccard Index
- Cohen's Kappa
- Hamming Loss
- Confusion Matrix (Matriz de Confusão)

### Otimização de Hiperparâmetros

Todos os algoritmos utilizam GridSearchCV para otimização automática de hiperparâmetros.

## Contribuição

Este projeto é parte de um curso de Inteligência Artificial Aplicada e serve como material de estudo e prática.

## Licença

Projeto educacional - uso acadêmico.
