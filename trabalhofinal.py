import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
import warnings

# Oculta avisos no terminal
warnings.filterwarnings("ignore")

# Carrega base de dados
df = pd.read_parquet("df_refined_neuro.parquet")

# Define colunas de interesse
colunas_modelo = [
    "dias_totais_internacao",
    "idade",
    "sexo_paciente",
    "carater_internacao",
    "nivel_complexidade",
    "qtd_comorbidades",
    "origem_paciente",
    "cid_principal",
    "procedimento_solicitado",
]

# remove valores nulos da base
df_modelo = df[colunas_modelo].dropna().copy()

# separa 80% para treino e 20% para teste com semente fixa
df_train, df_test = train_test_split(df_modelo, test_size=0.2, random_state=2026)

# Define alvo usando mediana do treino
mediana_alvo = df_train["dias_totais_internacao"].median()

# identifica categorias mais frequentes para reduzir dimensionalidade
top_15_cids = df_train["cid_principal"].value_counts().nlargest(15).index
top_15_procs = df_train["procedimento_solicitado"].value_counts().nlargest(15).index


# funcao de tratamento e definicao do alvo binario
def preparar_dados(df_part, mediana, cids_conhecidos, procs_conhecidos):
    df_clean = df_part.copy()

    # agrupa categorias minoritarias em 'Outros'
    df_clean["cid_agrupado"] = df_clean["cid_principal"].where(
        df_clean["cid_principal"].isin(cids_conhecidos), "Outros_CIDs"
    )
    df_clean["proc_agrupado"] = df_clean["procedimento_solicitado"].where(
        df_clean["procedimento_solicitado"].isin(procs_conhecidos), "Outros_Procs"
    )

    # converte alvo em 0 ou 1
    y = (df_clean["dias_totais_internacao"] > mediana).astype(int)

    # remove colunas originais nao utilizaveis
    X_raw = df_clean.drop(
        columns=["dias_totais_internacao", "cid_principal", "procedimento_solicitado"]
    )

    return X_raw, y


# aplica preparacao em treino e teste
X_train_raw, y_train = preparar_dados(df_train, mediana_alvo, top_15_cids, top_15_procs)
X_test_raw, y_test = preparar_dados(df_test, mediana_alvo, top_15_cids, top_15_procs)

# aplica One-Hot Encoding evitando multicolinearidade
X_train_final = pd.get_dummies(X_train_raw, drop_first=True, dtype=float)

# iguala as colunas do teste as do treino
X_test_final = pd.get_dummies(X_test_raw, drop_first=True, dtype=float).reindex(
    columns=X_train_final.columns, fill_value=0
)

# inicializa e treina Random Forest
rfc = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rfc.fit(X_train_final, y_train)

# gera previsoes do Random Forest
y_pred_class_rf = rfc.predict(X_test_final)
y_pred_prob_rf = rfc.predict_proba(X_test_final)[:, 1]

# inicializa e treina Gradient Boosting
gbc = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=5, random_state=2026
)
gbc.fit(X_train_final, y_train)

# gera previsoes do Gradient Boosting
y_pred_class_gb = gbc.predict(X_test_final)
y_pred_prob_gb = gbc.predict_proba(X_test_final)[:, 1]


# funcao para consolidar metricas
def calcular_metricas(y_true, y_pred_class, y_pred_prob):
    return (
        accuracy_score(y_true, y_pred_class),
        roc_auc_score(y_true, y_pred_prob),
        recall_score(y_true, y_pred_class),
    )


# calcula resultados finais
m_rf = calcular_metricas(y_test, y_pred_class_rf, y_pred_prob_rf)
m_gb = calcular_metricas(y_test, y_pred_class_gb, y_pred_prob_gb)

# imprime consolidado de metricas
print("Resultados de Desempenho")
tabela_resultados = pd.DataFrame(
    {
        "Modelo": ["Random Forest", "Gradient Boosting"],
        "Acuracia": [f"{m_rf[0]*100:.1f}%", f"{m_gb[0]*100:.1f}%"],
        "ROC-AUC": [f"{m_rf[1]*100:.1f}%", f"{m_gb[1]*100:.1f}%"],
        "Sensibilidade": [f"{m_rf[2]*100:.1f}%", f"{m_gb[2]*100:.1f}%"],
    }
)
print(tabela_resultados.to_string(index=False))

# imprime pesos das variaveis do modelo vencedor
print("\nFeature Importance Gradient Boosting")
importancias = gbc.feature_importances_
df_importancia = (
    pd.DataFrame(
        {
            "Caracteristica": X_train_final.columns,
            "Peso no Modelo (%)": (importancias * 100).round(2),
        }
    )
    .sort_values(by="Peso no Modelo (%)", ascending=False)
    .head(15)
)

print(df_importancia.to_string(index=False))
