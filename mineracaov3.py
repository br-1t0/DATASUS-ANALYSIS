import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Iniciando o Pipeline de Mineração com Ajuste Fino (Hyperparameter Tuning)...")

# =====================================================================
# 1. CARREGAMENTO E ENGENHARIA DE FEATURES (Agrupamento do CID)
# =====================================================================
df = pd.read_parquet("df_refined_neuro.parquet")

top_15_cids = df["cid_principal"].value_counts().nlargest(15).index
df["cid_agrupado"] = df["cid_principal"].where(
    df["cid_principal"].isin(top_15_cids), "Outros_CIDs"
)

colunas_modelo = [
    "dias_totais_internacao",
    "idade",
    "sexo_paciente",
    "carater_internacao",
    "nivel_complexidade",
    "especialidade_leito",
    "qtd_comorbidades",
    "origem_paciente",
    "cid_agrupado",
]

df_modelo = df[colunas_modelo].dropna().copy()

# =====================================================================
# 2. TRATAMENTO DE OUTLIERS (Corte nos 95%)
# =====================================================================
limite_dias = df_modelo["dias_totais_internacao"].quantile(0.95)
df_modelo = df_modelo[df_modelo["dias_totais_internacao"] <= limite_dias]

# =====================================================================
# 3. PRÉ-PROCESSAMENTO (One-Hot Encoding)
# =====================================================================
y = df_modelo["dias_totais_internacao"]
X = pd.get_dummies(
    df_modelo.drop(columns=["dias_totais_internacao"]), drop_first=True, dtype=float
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Matriz X criada com {X_train.shape[1]} colunas.")

# =====================================================================
# 4. REGRESSÃO LINEAR MÚLTIPLA COM STEPWISE
# =====================================================================
print("\n[1/3] Treinando Regressão Linear (Stepwise)...")


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


modelo_ols, features_selecionadas = backward_elimination(X_train, y_train)
X_test_ols = sm.add_constant(X_test[features_selecionadas])
y_pred_ols = modelo_ols.predict(X_test_ols)

# =====================================================================
# 5. RANDOM FOREST E GRADIENT BOOSTING COM GRID SEARCH
# =====================================================================
print("[2/3] Treinando Random Forest Base...")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(
    "[3/3] Iniciando GridSearch no Gradient Boosting (Isso pode levar alguns minutos)..."
)
# Grade de parâmetros para o computador testar qual é o melhor
param_grid = {
    "n_estimators": [100, 200],  # Testa com 100 e 200 árvores
    "max_depth": [3, 5, 7],  # Testa diferentes profundidades
    "learning_rate": [0.05, 0.1, 0.2],  # Testa a agressividade do aprendizado
}

# Configura o GridSearchCV
gb_base = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=gb_base, param_grid=param_grid, cv=3, n_jobs=-1, scoring="r2", verbose=1
)

# O modelo agora vai testar todas as 18 combinações de hiperparâmetros acima!
grid_search.fit(X_train, y_train)

# Captura o melhor modelo encontrado
gb_otimizado = grid_search.best_estimator_
y_pred_gb_otimizado = gb_otimizado.predict(X_test)


# =====================================================================
# 6. RELATÓRIO FINAL E FEATURE IMPORTANCE
# =====================================================================
def calcular_metricas(y_true, y_pred):
    return (
        mean_absolute_error(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred)),
        r2_score(y_true, y_pred),
    )


m_ols = calcular_metricas(y_test, y_pred_ols)
m_rf = calcular_metricas(y_test, y_pred_rf)
m_gb = calcular_metricas(y_test, y_pred_gb_otimizado)

print("\n" + "=" * 60)
print("🏆 RESULTADOS APÓS AJUSTE FINO (Hyperparameter Tuning) 🏆")
print("=" * 60)
tabela_resultados = pd.DataFrame(
    {
        "Modelo": [
            "Regressão Linear",
            "Random Forest Base",
            "Gradient Boosting Otimizado",
        ],
        "MAE (Dias)": [m_ols[0], m_rf[0], m_gb[0]],
        "R² (Explicação)": [m_ols[2], m_rf[2], m_gb[2]],
    }
)
print(tabela_resultados.to_string(index=False))

print(
    f"\nOs melhores parâmetros encontrados para o Gradient Boosting foram:\n{grid_search.best_params_}"
)

# Extraindo o Feature Importance do GB Otimizado para criar gráficos no Streamlit depois
importancias = pd.DataFrame(
    {"Atributo": X.columns, "Importância (%)": gb_otimizado.feature_importances_ * 100}
).sort_values(by="Importância (%)", ascending=False)

print("\n" + "=" * 60)
print("🌳 FEATURE IMPORTANCE (O que o Gradient Boosting achou mais importante) 🌳")
print("=" * 60)
print(importancias.head(10).to_string(index=False))
