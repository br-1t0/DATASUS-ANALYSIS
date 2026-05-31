import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px

df = pd.read_parquet("df_refined_neuro.parquet")

# 1. Agrupa os cids com maior frequencia
top_15_cids = df["cid_principal"].value_counts().nlargest(15).index
df["cid_agrupado"] = df["cid_principal"].apply(
    lambda x: x if x in top_15_cids else "Outros_CIDs"
)

# 2. Variaveis do "Dia 0"
colunas_preditoras = [
    "idade",
    "sexo_paciente",
    "nivel_complexidade",
    "carater_internacao",
    "cid_agrupado",
]

X = df[colunas_preditoras]
y = df["dias_totais_internacao"]

# 3. Transformação Categórica para Numérica (OneHotEncoding)
X_encoded = pd.get_dummies(X, drop_first=True)

# 4. Divisão 80-20 (Usamos os dias normais aqui)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# ==============================================================
# A MÁGICA: Transformamos apenas os dados de treino em Logaritmo
# ==============================================================
y_train_log = np.log1p(y_train)

print("--- Regressão Linear ---")
modelo_lr = LinearRegression()
# O modelo treina no "Mundo Logarítmico"
modelo_lr.fit(X_train, y_train_log)

# As previsões saem em formato logarítmico
lr_previsoes_log = modelo_lr.predict(X_test)

# Revertemos as previsões para "Dias Reais" antes de avaliar!
lr_previsoes_dias = np.expm1(lr_previsoes_log)

lr_mae = mean_absolute_error(y_test, lr_previsoes_dias)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_previsoes_dias))
lr_r2 = r2_score(y_test, lr_previsoes_dias)

print(f"MAE:  {lr_mae:.2f} dias")
print(f"RMSE: {lr_rmse:.2f} dias")
print(f"R²:   {lr_r2:.4f}\n")


print("--- Random Forest ---")
modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
# O modelo treina no "Mundo Logarítmico"
modelo_rf.fit(X_train, y_train_log)

# As previsões saem e são logo revertidas
rf_previsoes_log = modelo_rf.predict(X_test)
rf_previsoes_dias = np.expm1(rf_previsoes_log)

rf_mae = mean_absolute_error(y_test, rf_previsoes_dias)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_previsoes_dias))
rf_r2 = r2_score(y_test, rf_previsoes_dias)

print(f"MAE:  {rf_mae:.2f} dias")
print(f"RMSE: {rf_rmse:.2f} dias")
print(f"R²:   {rf_r2:.4f}\n")

# O gráfico de importâncias mantém-se inalterado
importancias = pd.DataFrame(
    {
        "Variavel": X_train.columns,
        "Importancia_Percentual": modelo_rf.feature_importances_ * 100,
    }
).sort_values(by="Importancia_Percentual", ascending=True)

fig_imp = px.bar(
    importancias.tail(15),
    x="Importancia_Percentual",
    y="Variavel",
    orientation="h",
    title="Top 15 Atributos Mais Importantes - Random Forest",
    labels={
        "Importancia_Percentual": "Impacto na Previsão (%)",
        "Variavel": "Atributo",
    },
)
# fig_imp.show()
