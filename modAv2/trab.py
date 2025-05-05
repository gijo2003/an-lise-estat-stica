import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt

# Carrega o dataset
df = pd.read_csv('dataset_13.csv')  # ajuste o caminho se necessário

# Identifica colunas categóricas
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print("Variáveis categóricas:", cat_cols)

# Cria dummies, removendo a categoria-base (drop_first=True)
df_model = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Lista de variáveis explicativas (todas menos 'tempo_resposta')
X = df_model.drop(columns=['tempo_resposta'])
y = df_model['tempo_resposta']

# Adiciona intercepto
X_const = sm.add_constant(X)

# Converte todas as colunas de X_const para float (inclui bool->0/1)
X_const = X_const.astype(float)
# Garante que y seja float
y = y.astype(float)

# Remove linhas com NaN (caso haja)
df_clean = pd.concat([X_const, y.rename('tempo_resposta')], axis=1).dropna()
X_clean = df_clean[X_const.columns]
y_clean = df_clean['tempo_resposta']

# Ajusta o modelo usando os dados limpos
modelo_full = sm.OLS(y_clean, X_clean).fit()

# Exibe sumário com intercepto, coeficientes, R², R²-ajustado, p-valores e teste F
print(modelo_full.summary())

# Calcula VIF para cada variável (incluindo a constante)
vif_data = pd.DataFrame({
    'feature': X_clean.columns,
    'VIF': [variance_inflation_factor(X_clean.values, i)
            for i in range(X_clean.shape[1])]
})
print(vif_data)

# 5.1 Gráfico resíduos vs. valores ajustados
resid = modelo_full.resid
fitted = modelo_full.fittedvalues

plt.figure(figsize=(6,4))
plt.scatter(fitted, resid)
plt.axhline(0, linestyle='--')
plt.xlabel('Valores ajustados')
plt.ylabel('Resíduos')
plt.title('Resíduos vs. Ajustados')
plt.show()

# 5.2 Teste de Breusch–Pagan
bp_test = sms.het_breuschpagan(resid, modelo_full.model.exog)
labels = ['LM-Stat', 'p-value', 'F-Stat', 'F p-value']
print("Breusch–Pagan:", dict(zip(labels, bp_test)))

# Comparação de modelos
vars_to_drop = ['armazenamento_tb']  # justifique no relatório
X2 = X_clean.drop(columns=vars_to_drop)

modelo_reduzido = sm.OLS(y_clean, X2).fit()

print("Full model R²-ajust:", modelo_full.rsquared_adj)
print("Reduced model R²-ajust:", modelo_reduzido.rsquared_adj)

from statsmodels.stats.anova import anova_lm
anova_res = anova_lm(modelo_reduzido, modelo_full)
print("\nComparação ANOVA entre modelos:\n", anova_res)
