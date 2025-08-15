import os.path
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

from itertools import combinations

import statsmodels.api as sm

import matplotlib.pyplot as plt 
import seaborn as sns
import joblib


# --- PARÂMETROS DE ANÁLISE (AJUSTE SE NECESSÁRIO) ---
# Suporte mínimo para um itemset ser considerado frequente.
MIN_SUPPORT = 0.15
# Confiança mínima para uma regra ser considerada relevante. 
MIN_CONFIDENCE = 0.7
# ----------------------------------------------------

def formatar_itemsets(itemset):
    """Função para converter frozensets em strings legíveis."""
    return ', '.join(list(itemset))

# --- LEITURA E PREPARAÇÃO DOS DADOS (MÉTODO CORRIGIDO) ---

    # Este método força a leitura de cada linha em uma única coluna chamada 'Exercicios',
    # resolvendo o problema de número variável de itens por linha.
df = pd.read_csv('2 - Musculacao - Dados.csv', header=None, names=['Exercicios'])
print("Arquivo '2 - Musculacao - Dados.csv' carregado com sucesso.")

    # Agora, usamos o método str.get_dummies para separar os exercícios corretamente.
df_onehot = df['Exercicios'].str.get_dummies(sep=';')
print(f"Dados da coluna 'Exercicios' processados com sucesso.\n")


# ----------------------------------------------------------------

# 1. GERAR ITEMSETS FREQUENTES
print(f"Buscando conjuntos com suporte mínimo de {MIN_SUPPORT*100:.1f}%...")
frequent_itemsets = apriori(df_onehot, min_support=MIN_SUPPORT, use_colnames=True)

if frequent_itemsets.empty:
    print("Nenhum conjunto frequente encontrado. Tente um MIN_SUPPORT e/ou MIN_CONFIDENCE menores.")
else:
    # 2. GERAR REGRAS DE ASSOCIAÇÃO
    print("Gerando regras de associação...")
    regras = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)

    if regras.empty:
        print("Nenhuma regra de associação pôde ser gerada.")
    else:
        # 3. FILTRAR E RANKear AS REGRAS
        regras['total_items'] = regras['antecedents'].apply(len) + regras['consequents'].apply(len)
        regras_filtradas = regras[regras['total_items'] <= 5].copy()
        regras_rankeadas = regras_filtradas.sort_values(by=['confidence', 'lift'], ascending=[False, False])
        
        print(f"Total de {len(regras_rankeadas)} regras encontradas.")

        # 4. FORMATAÇÃO DO RESULTADO FINAL
        regras_rankeadas['Rank'] = range(1, len(regras_rankeadas) + 1)
        regras_rankeadas['SE (Antecedente)'] = regras_rankeadas['antecedents'].apply(formatar_itemsets)
        regras_rankeadas['ENTÃO (Consequente)'] = regras_rankeadas['consequents'].apply(formatar_itemsets)
        regras_rankeadas['Confiança (%)'] = (regras_rankeadas['confidence'] * 100).map('{:.2f}%'.format)
        regras_rankeadas['Suporte (%)'] = (regras_rankeadas['support'] * 100).map('{:.2f}%'.format)
        regras_rankeadas['Lift'] = regras_rankeadas['lift'].map('{:.2f}'.format)
        
        resultado_final = regras_rankeadas[['Rank', 'SE (Antecedente)', 'ENTÃO (Consequente)', 'Confiança (%)', 'Suporte (%)', 'Lift']]

        # 5. SALVAR E EXIBIR OS RESULTADOS
        resultado_final.to_csv("ranking_completo_regras.csv", index=False)
        print("\n-> RANKING COMPLETO salvo com sucesso no arquivo 'ranking_completo_regras.csv'\n")

        print("--- TOP 25 REGRAS DO RANKING (lista completa no arquivo CSV) ---")
        pd.options.display.max_colwidth = 100
        print(resultado_final.head(25).to_string(index=False))

