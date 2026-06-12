import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

print("Iniciando o Pipeline Purista de Classificação...")

# =====================================================================
# 1. CARREGAMENTO E SPLIT INICIAL (A Regra de Ouro)
# =====================================================================
df = pd.read_parquet("df_refined_neuro.parquet")

colunas_modelo = [
    "dias_totais_internacao",
    "idade",
    "sexo_paciente",
    "carater_internacao",
    "nivel_complexidade",
    "especialidade_leito",
    "qtd_comorbidades",
    "origem_paciente",
    "cid_principal",
]

df_modelo = df[colunas_modelo].dropna().copy()

# O Split acontece ANTES de calcular qualquer estatística (Evita Data Leakage)
df_train, df_test = train_test_split(df_modelo, test_size=0.2, random_state=42)

# =====================================================================
# 2. APRENDIZADO DE REGRAS APENAS NO TREINO (Sem espiar o Teste)
# =====================================================================
limite_dias = df_train["dias_totais_internacao"].quantile(0.95)
mediana_dias = df_train["dias_totais_internacao"].median()
top_15_cids = df_train["cid_principal"].value_counts().nlargest(15).index

print(f"\n📊 Regras extraídas apenas dos 80% de treino:")
print(f"-> Corte de Outliers: {limite_dias:.0f} dias")
print(f"-> Mediana (Alvo): {mediana_dias:.0f} dias")


# =====================================================================
# 3. APLICAÇÃO DAS REGRAS NO TREINO E NO TESTE
# =====================================================================
def preparar_dados(df_part, limite, mediana, cids_conhecidos):
    # 1. Remove outliers
    df_clean = df_part[df_part["dias_totais_internacao"] <= limite].copy()

    # 2. Agrupa CIDs
    df_clean["cid_agrupado"] = df_clean["cid_principal"].where(
        df_clean["cid_principal"].isin(cids_conhecidos), "Outros_CIDs"
    )

    # 3. Cria o alvo
    y = (df_clean["dias_totais_internacao"] > mediana).astype(int)

    # 4. Remove colunas que não vão pro modelo
    X_raw = df_clean.drop(columns=["dias_totais_internacao", "cid_principal"])
    return X_raw, y


X_train_raw, y_train = preparar_dados(df_train, limite_dias, mediana_dias, top_15_cids)
X_test_raw, y_test = preparar_dados(df_test, limite_dias, mediana_dias, top_15_cids)

# Dummies: Aplica no treino e força o teste a ter exatamente as mesmas colunas
X_train_final = pd.get_dummies(X_train_raw, drop_first=True, dtype=float)
X_test_final = pd.get_dummies(X_test_raw, drop_first=True, dtype=float).reindex(
    columns=X_train_final.columns, fill_value=0
)

# =====================================================================
# 4. TREINAMENTO DOS MODELOS
# =====================================================================
print("\n[1/3] Treinando Regressão Logística (Stepwise)...")


def backward_elimination_logit(X_data, y_data, significance_level=0.05):
    features = X_data.columns.tolist()
    while len(features) > 0:
        X_with_constant = sm.add_constant(X_data[features])
        try:
            model = sm.Logit(y_data, X_with_constant).fit(disp=0)
        except np.linalg.LinAlgError:
            feature_problematica = X_data[features].var().idxmin()
            features.remove(feature_problematica)
            continue
        p_values = model.pvalues[1:]
        max_p_value = p_values.max()
        if max_p_value > significance_level:
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break
    return sm.Logit(y_data, sm.add_constant(X_data[features])).fit(disp=0), features


modelo_logit, features_selecionadas = backward_elimination_logit(X_train_final, y_train)
X_test_logit = sm.add_constant(X_test_final[features_selecionadas])
y_pred_prob_logit = modelo_logit.predict(X_test_logit)
y_pred_class_logit = (y_pred_prob_logit >= 0.5).astype(int)

print("[2/3] Treinando Random Forest Classifier...")
rfc = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rfc.fit(X_train_final, y_train)
y_pred_class_rf = rfc.predict(X_test_final)
y_pred_prob_rf = rfc.predict_proba(X_test_final)[:, 1]

print("[3/3] Treinando Gradient Boosting Classifier...")
gbc = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
)
gbc.fit(X_train_final, y_train)
y_pred_class_gb = gbc.predict(X_test_final)
y_pred_prob_gb = gbc.predict_proba(X_test_final)[:, 1]


# =====================================================================
# 5. RELATÓRIO FINAL
# =====================================================================
def calcular_metricas_classificacao(y_true, y_pred_class, y_pred_prob):
    return (
        accuracy_score(y_true, y_pred_class),
        roc_auc_score(y_true, y_pred_prob),
        recall_score(y_true, y_pred_class),
    )


m_log = calcular_metricas_classificacao(y_test, y_pred_class_logit, y_pred_prob_logit)
m_rf = calcular_metricas_classificacao(y_test, y_pred_class_rf, y_pred_prob_rf)
m_gb = calcular_metricas_classificacao(y_test, y_pred_class_gb, y_pred_prob_gb)

print("\n" + "=" * 80)
print("🏆 RESULTADOS METODOLOGICAMENTE BLINDADOS 🏆")
print("=" * 80)
tabela_resultados = pd.DataFrame(
    {
        "Modelo": ["Regressão Logística", "Random Forest", "Gradient Boosting"],
        "Acurácia": [
            f"{m_log[0]*100:.1f}%",
            f"{m_rf[0]*100:.1f}%",
            f"{m_gb[0]*100:.1f}%",
        ],
        "ROC-AUC": [
            f"{m_log[1]*100:.1f}%",
            f"{m_rf[1]*100:.1f}%",
            f"{m_gb[1]*100:.1f}%",
        ],
        "Sensibilidade": [
            f"{m_log[2]*100:.1f}%",
            f"{m_rf[2]*100:.1f}%",
            f"{m_gb[2]*100:.1f}%",
        ],
    }
)
print(tabela_resultados.to_string(index=False))
