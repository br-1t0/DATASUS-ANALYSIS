import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px

df = pd.read_parquet("df_refined_neuro.parquet")

# agrupa os cids com maior frequencia
top_15_cids = df["cid_principal"].value_counts().nlargest(15).index
df["cid_agrupado"] = df["cid_principal"].apply(
    lambda x: x if x in top_15_cids else "Outros_CIDs"
)

# evita data leak, pegando só as variaveis que teriamos na hora do paciente entrar
colunas_preditoras = [
    "idade",
    "sexo_paciente",
    "nivel_complexidade",
    "carater_internacao",
    "cid_agrupado",
]

X = df[colunas_preditoras]
y = df["dias_totais_internacao"]


# função do p´roprio sklearn pra transformar variável categórica em numérica (OneHotEncoding)
X_encoded = pd.get_dummies(X, drop_first=True)

# 80 - 20
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)


print("Regressão linear")
modelo_lr = LinearRegression()
modelo_lr.fit(X_train, y_train)
lr_previsoes = modelo_lr.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_previsoes)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_previsoes))
lr_r2 = r2_score(y_test, lr_previsoes)

print(f"MAE:  {lr_mae:.2f} dias")
print(f"RMSE: {lr_rmse:.2f} dias")
print(f"R²:   {lr_r2:.4f}\n")


print("Random forest")
modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
modelo_rf.fit(X_train, y_train)
rf_previsoes = modelo_rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_previsoes)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_previsoes))
rf_r2 = r2_score(y_test, rf_previsoes)

print(f"MAE:  {rf_mae:.2f} dias")
print(f"RMSE: {rf_rmse:.2f} dias")
print(f"R²:   {rf_r2:.4f}\n")

# grafico com as variaveis mais importantes
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
fig_imp.show()
