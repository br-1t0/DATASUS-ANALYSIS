import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Dashboard neurológico", layout="wide", initial_sidebar_state="expanded"
)


@st.cache_data
def load_data():
    return pd.read_parquet("df_refined_neuro.parquet")


try:
    df = load_data()
except Exception as e:
    st.error(
        f"Erro ao carregar o arquivo: {e}. Verifique se o 'df_refined_neuro.parquet' está na mesma pasta."
    )
    st.stop()

with st.sidebar:
    st.title("Tipos de análises")

    tipo_analise = st.radio(
        "Selecione o tipo de análise:",
        options=[
            "1. Análise univariada",
            "2. Análise multivariada",
        ],
    )

    st.divider()
    st.caption("Trabalho de TBD (big data)")
    st.divider()

if tipo_analise == "1. Análise univariada":

    st.title("Análise univariada")
    st.markdown("Compreendendo o perfil individual de cada variável neurológica.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de internações (AIHs)", f"{df.shape[0]:,}".replace(",", "."))
    col2.metric(
        "Média de internação (dias)", f"{df['dias_totais_internacao'].mean():.1f}"
    )
    taxa_mortalidade = (df[df["obito"] == "com_obito"].shape[0] / df.shape[0]) * 100
    col3.metric("Taxa de mortalidade", f"{taxa_mortalidade:.2f}%")

    st.divider()

    aba_num, aba_cat, aba_desfecho = st.tabs(
        [
            "Variáveis contínuas",
            "Variáveis categóricas",
            "Motivos de saída e CIDs",
        ]
    )

    with aba_num:
        st.subheader("Resumo estatístico")
        st.markdown(
            "Medidas de tendência central e dispersão das características numéricas dos pacientes."
        )

        colunas_numericas = [
            "dias_totais_internacao",
            "idade",
            "valor_total",
            "dias_uti_mes",
        ]

        traducao_colunas = {
            "dias_totais_internacao": "Dias de internação",
            "idade": "Idade (anos)",
            "valor_total": "Valor total (R$)",
            "dias_uti_mes": "Dias de UTI no mês",
        }

        traducao_linhas = {
            "count": "Total de registros",
            "mean": "Média",
            "std": "Desvio padrão",
            "min": "Mínimo",
            "25%": "25% (1º quartil)",
            "50%": "Mediana (2º quartil)",
            "75%": "75% (3º quartil)",
            "max": "Máximo",
        }

        tabela_stats = (
            df[colunas_numericas]
            .describe()
            .rename(columns=traducao_colunas, index=traducao_linhas)
            .round(2)
        )
        st.dataframe(tabela_stats, width="stretch")

        st.divider()

        st.subheader("Tempo de internação (dias)")
        col_t1, col_t2 = st.columns([7, 3])

        with col_t1:
            fig_tempo = px.histogram(
                df,
                x="dias_totais_internacao",
                nbins=50,
                color_discrete_sequence=["#4B0082"],
                labels={
                    "dias_totais_internacao": "Dias internado",
                    "count": "Quantidade de casos",
                },
            )
            st.plotly_chart(fig_tempo, width="stretch")

        with col_t2:
            fig_box_tempo = px.box(
                df,
                y="dias_totais_internacao",
                color_discrete_sequence=["#4B0082"],
                labels={"dias_totais_internacao": "Dias internado"},
            )
            st.plotly_chart(fig_box_tempo, width="stretch")

        st.divider()

        st.subheader("Distribuição etária")
        faixas = [-1, 2, 12, 18, 59, 120]
        nomes_faixas = [
            "Bebês (0-2)",
            "Crianças (3-12)",
            "Adolescentes (13-18)",
            "Adultos (19-59)",
            "Idosos (60+)",
        ]
        df["faixa_etaria"] = pd.cut(df["idade"], bins=faixas, labels=nomes_faixas)

        col_i1, col_i2 = st.columns([6, 4])
        with col_i1:
            fig_idade = px.histogram(
                df,
                x="idade",
                color_discrete_sequence=["#008080"],
                labels={"idade": "Idade (anos)", "count": "Quantidade de casos"},
            )
            fig_idade.update_traces(xbins=dict(start=0, end=120, size=5))
            st.plotly_chart(fig_idade, width="stretch")

        with col_i2:
            df_faixa = df["faixa_etaria"].value_counts(normalize=True).reset_index()
            df_faixa.columns = ["Faixa etária", "Proporção"]
            df_faixa["Porcentagem"] = (df_faixa["Proporção"] * 100).round(1).astype(
                str
            ) + "%"

            fig_faixa = px.bar(
                df_faixa,
                x="Proporção",
                y="Faixa etária",
                orientation="h",
                text="Porcentagem",
                color_discrete_sequence=["#20B2AA"],
                title="Distribuição por faixa etária",
            )
            fig_faixa.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_faixa, width="stretch")

        st.divider()

        st.subheader("Custo da internação (R$)")
        col_v1, col_v2 = st.columns([7, 3])
        with col_v1:
            fig_valor = px.histogram(
                df,
                x="valor_total",
                nbins=60,
                color_discrete_sequence=["#2E8B57"],
                labels={
                    "valor_total": "Valor pago pelo SUS (R$)",
                    "count": "Quantidade de casos",
                },
            )
            st.plotly_chart(fig_valor, width="stretch")

        with col_v2:
            fig_box_valor = px.box(
                df,
                y="valor_total",
                color_discrete_sequence=["#2E8B57"],
                labels={"valor_total": "Valor pago pelo SUS (R$)"},
            )
            st.plotly_chart(fig_box_valor, width="stretch")

    with aba_cat:
        st.subheader("Frequências: complexidade, sexo e caráter de internação")
        st.markdown(
            "Proporções das variáveis qualitativas (atributos) da base de pacientes."
        )

        col_esq, col_dir = st.columns(2)

        with col_esq:
            fig_comp = px.pie(
                df, names="nivel_complexidade", title="Nível de complexidade", hole=0.4
            )
            fig_comp.update_traces(
                textposition="inside",
                texttemplate="%{label}<br>%{value} casos<br>(%{percent:.2%})",
            )
            st.plotly_chart(fig_comp, width="stretch")

            freq_carater = df["carater_internacao"].value_counts().reset_index()
            fig_carater = px.bar(
                freq_carater,
                x="count",
                y="carater_internacao",
                orientation="h",
                title="Caráter da internação",
                labels={
                    "count": "Quantidade de casos",
                    "carater_internacao": "Caráter",
                },
            )
            fig_carater.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_carater, width="stretch")

        with col_dir:
            fig_sexo = px.pie(
                df, names="sexo_paciente", title="Proporção por sexo", hole=0.4
            )
            fig_sexo.update_traces(
                textposition="inside",
                texttemplate="%{label}<br>%{value} casos<br>(%{percent:.2%})",
            )
            st.plotly_chart(fig_sexo, width="stretch")

            fig_obito = px.bar(
                df["obito"].value_counts().reset_index(),
                x="count",
                y="obito",
                orientation="h",
                title="Status de óbito",
                labels={"count": "Quantidade de casos", "obito": "Status"},
            )
            st.plotly_chart(fig_obito, width="stretch")

    with aba_desfecho:
        st.subheader("Motivos de saída e CIDs de maior frequência")
        col_a, col_b = st.columns(2)

        with col_a:
            freq_saida = (
                df["motivo_saida_permanencia"].value_counts().reset_index().head(10)
            )
            fig_saida = px.bar(
                freq_saida,
                x="count",
                y="motivo_saida_permanencia",
                orientation="h",
                title="Motivos de saída/permanência",
                labels={
                    "count": "Quantidade de casos",
                    "motivo_saida_permanencia": "Motivo de saída",
                },
            )
            fig_saida.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_saida, width="stretch")

        with col_b:
            freq_cid = (
                df.groupby(["cid_principal", "diagnostico_principal"])
                .size()
                .reset_index(name="count")
                .sort_values(by="count", ascending=False)
                .head(10)
            )

            fig_cid = px.bar(
                freq_cid,
                x="count",
                y="cid_principal",
                orientation="h",
                title="Ranking dos 10 principais CIDs neurológicos",
                color_discrete_sequence=["#FF7F50"],
                hover_data={
                    "cid_principal": True,
                    "count": True,
                    "diagnostico_principal": True,
                },
                labels={
                    "count": "Quantidade de casos",
                    "cid_principal": "Código do CID",
                    "diagnostico_principal": "Diagnóstico clínico",
                },
            )
            fig_cid.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_cid, width="stretch")

elif tipo_analise == "2. Análise multivariada":

    st.title("Análise multivariada")
    st.markdown(
        "Investigando como as variáveis preditoras de entrada (Dia 0) "
        "impactam a nossa variável alvo (dias de internação)."
    )

    aba_demografica, aba_clinica, aba_corr, aba_work = st.tabs(
        [
            "Dimensão demográfica",
            "Dimensão clínica",
            "Correlação numérica",
            "Material do trabalho",
        ]
    )

    with aba_demografica:
        st.subheader("Impacto da idade e sexo no tempo de internação")
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            fig_scatter_idade = px.scatter(
                df,
                x="idade",
                y="dias_totais_internacao",
                opacity=0.4,
                color_discrete_sequence=["#4682B4"],
                labels={
                    "idade": "Idade (anos)",
                    "dias_totais_internacao": "Dias internado",
                },
                title="Idade vs Dias de internação",
            )
            st.plotly_chart(fig_scatter_idade, width="stretch")

        with col_d2:
            fig_box_sexo = px.box(
                df,
                x="sexo_paciente",
                y="dias_totais_internacao",
                color="sexo_paciente",
                labels={
                    "sexo_paciente": "Sexo",
                    "dias_totais_internacao": "Dias internado",
                },
                title="Distribuição de tempo por sexo",
            )
            st.plotly_chart(fig_box_sexo, width="stretch")

    with aba_clinica:
        st.subheader("Impacto da complexidade e caráter da internação")
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            fig_box_comp = px.box(
                df,
                x="nivel_complexidade",
                y="dias_totais_internacao",
                color="nivel_complexidade",
                labels={
                    "nivel_complexidade": "Complexidade",
                    "dias_totais_internacao": "Dias internado",
                },
                title="Tempo por nível de complexidade",
            )
            st.plotly_chart(fig_box_comp, width="stretch")

        with col_c2:
            fig_box_carater = px.box(
                df,
                x="carater_internacao",
                y="dias_totais_internacao",
                color="carater_internacao",
                labels={
                    "carater_internacao": "Caráter",
                    "dias_totais_internacao": "Dias internado",
                },
                title="Tempo por caráter de internação",
            )
            st.plotly_chart(fig_box_carater, width="stretch")

    with aba_corr:
        st.subheader("Matriz de correlação de Pearson")
        st.markdown(
            "Análise restrita às variáveis estritamente numéricas para identificar colinearidade."
        )

        cols_corr = ["dias_totais_internacao", "idade"]
        matriz_corr = df[cols_corr].corr().round(2)
        nomes_eixos = ["Dias internado", "Idade (anos)"]

        fig_corr = px.imshow(
            matriz_corr,
            x=nomes_eixos,
            y=nomes_eixos,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Correlação linear",
        )
        st.plotly_chart(fig_corr, width="stretch")
    with aba_work:
        st.subheader("Links pra apresentação")
        url_relatorio, url_drive, url_collab, url_github = (
            "https://docs.google.com/document/d/1pGE8pd-mW799gN04NQI7TvgqMr47MtyxiXau7XD_oDk/edit?usp=sharing",
            "https://drive.google.com/drive/folders/1clVSfoL3IQmiYx0XGxRrZs-i7rtgx36b?usp=sharing",
            "https://colab.research.google.com/drive/1UkgyP2sbi-4nmPZv-IHLqqmfVmFNsVWf?usp=sharing",
            "https://github.com/br-1t0/DATASUS-ANALYSIS",
        )
        st.markdown("[Relatório](%s)" % url_relatorio)
        st.markdown("[Drive](%s)" % url_drive)
        st.markdown("[Collab](%s)" % url_collab)
        st.markdown("[Github](%s)" % url_github)
