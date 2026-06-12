import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Iniciando o Pipeline de Mineração de Dados com Agrupamento de CID...")

# =====================================================================
# 1. CARREGAMENTO E ENGENHARIA DE FEATURES (Agrupamento do CID)
# =====================================================================
df = pd.read_parquet("df_refined_neuro.parquet")

# Pega os 15 CIDs mais frequentes
top_15_cids = df["cid_principal"].value_counts().nlargest(15).index

# Cria uma nova coluna: se o CID estiver no top 15, mantém; se não, vira 'Outros'
df["cid_agrupado"] = df["cid_principal"].where(
    df["cid_principal"].isin(top_15_cids), "Outros_CIDs"
)

# Selecionamos o alvo e as informações disponíveis na ADMISSÃO + CID Agrupado
colunas_modelo = [
    "dias_totais_internacao",
    "idade",
    "sexo_paciente",
    "carater_internacao",
    "nivel_complexidade",
    "especialidade_leito",
    "qtd_comorbidades",
    "origem_paciente",
    "cid_agrupado",  # Nossa nova variável
]

# Filtra o DataFrame e remove linhas com valores nulos
df_modelo = df[colunas_modelo].dropna().copy()

# =====================================================================
# 2. TRATAMENTO DE OUTLIERS (Corte nos 95%)
# =====================================================================
tamanho_original = df_modelo.shape[0]
limite_dias = df_modelo["dias_totais_internacao"].quantile(0.95)

# Mantém apenas os pacientes dentro do quantil 95%
df_modelo = df_modelo[df_modelo["dias_totais_internacao"] <= limite_dias]

print(f"\nRemoção de Outliers (Corte no Quantil 95% -> {limite_dias:.0f} dias):")
print(f"Registros antes: {tamanho_original} | Registros agora: {df_modelo.shape[0]}")

# =====================================================================
# 3. PRÉ-PROCESSAMENTO (One-Hot Encoding)
# =====================================================================
y = df_modelo["dias_totais_internacao"]
X_categoricas = df_modelo.drop(columns=["dias_totais_internacao"])

# Converte strings para colunas 0 e 1 (Dummies)
X = pd.get_dummies(X_categoricas, drop_first=True, dtype=float)

# Separa 80% para treino e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nMatriz de entrada (X) criada com", X_train.shape[1], "variáveis preditoras.")

# =====================================================================
# 4. REGRESSÃO LINEAR MÚLTIPLA COM STEPWISE (Backward Elimination)
# =====================================================================
print("\nTreinando Modelo 1: Regressão Linear Múltipla (Stepwise)...")


def backward_elimination(X_data, y_data, significance_level=0.05):
    features = X_data.columns.tolist()
    while len(features) > 0:
        X_with_constant = sm.add_constant(X_data[features])
        model = sm.OLS(y_data, X_with_constant).fit()
        p_values = model.pvalues[1:]
        max_p_value = p_values.max()

        if max_p_value > significance_level:
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break

    return sm.OLS(y_data, sm.add_constant(X_data[features])).fit(), features


# Treina o Stepwise
modelo_ols, features_selecionadas = backward_elimination(X_train, y_train)

# Testa o OLS
X_test_ols = sm.add_constant(X_test[features_selecionadas])
y_pred_ols = modelo_ols.predict(X_test_ols)

# =====================================================================
# 5. MODELOS PREDITIVOS AVANÇADOS (Sklearn)
# =====================================================================
print("Treinando Modelo 2: Random Forest Regressor...")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Treinando Modelo 3: Gradient Boosting Regressor...")
gb = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)


# =====================================================================
# 6. COMPARAÇÃO E RELATÓRIO DE MÉTRICAS
# =====================================================================
def calcular_metricas(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


metricas_ols = calcular_metricas(y_test, y_pred_ols)
metricas_rf = calcular_metricas(y_test, y_pred_rf)
metricas_gb = calcular_metricas(y_test, y_pred_gb)

print("\n" + "=" * 50)
print("🏆 RESULTADOS DA COMPARAÇÃO DOS MODELOS 🏆")
print("=" * 50)
tabela_resultados = pd.DataFrame(
    {
        "Modelo": ["Regressão Linear (Stepwise)", "Random Forest", "Gradient Boosting"],
        "MAE (Dias)": [metricas_ols[0], metricas_rf[0], metricas_gb[0]],
        "RMSE (Dias)": [metricas_ols[1], metricas_rf[1], metricas_gb[1]],
        "R²": [metricas_ols[2], metricas_rf[2], metricas_gb[2]],
    }
)
# Usando to_string para evitar problemas de dependência com tabulate
print(tabela_resultados.to_string(index=False))

print("\n" + "=" * 50)
print("📊 EQUAÇÃO DO MODELO ESTATÍSTICO (Exigência do Orientador) 📊")
print("=" * 50)
print("As variáveis que sobreviveram ao corte do P-Valor (< 0.05) foram:")
print(modelo_ols.summary().tables[1])
