"""
╔══════════════════════════════════════════════════════════════╗
║     PREVISÃO DE PREÇOS — CONCESSIONÁRIA DE CARROS           
║      Dashboard Streamlit | Portfolio Fevereiro 2026           
╚══════════════════════════════════════════════════════════════╝

Execução local:
    pip install streamlit pandas numpy scikit-learn matplotlib seaborn
    streamlit run app_streamlit.py

Deploy gratuito:
    1. Faça push do repositório no GitHub
    2. Acesse https://streamlit.io/sharing
    3. Cole a URL do repositório → Deploy
    4. Arquivo Gerado com auxilio de I.A
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─── ESTILO DA PÁGINA ────────────────────────────────────────
st.set_page_config(
    page_title="Previsão de Preços — Concessionária",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* fundo global */
    .stApp { background-color: #0f1117; color: #c8d0de; font-family: 'Segoe UI', sans-serif; }

    /* sidebar */
    [data-testid="stSidebar"] { background-color: #161922; border-right: 1px solid #2a3040; }
    [data-testid="stSidebar"] * { color: #c8d0de !important; }

    /* cabeçalho principal */
    h1 { color: #ffffff !important; font-size: 2rem !important; margin-bottom: 4px !important; }
    h2 { color: #00e5a0 !important; font-size: 1.3rem !important; border-bottom: 1px solid #2a3040; padding-bottom: 6px; }
    h3 { color: #00c9ff !important; font-size: 1.05rem !important; }

    /* cards via container */
    .stMetric { background: #161922; border-radius: 12px; padding: 16px; border: 1px solid #2a3040; }
    .stMetric label { color: #7a8490 !important; font-size: 0.82rem !important; }
    .stMetric .css-1ld2y9r { color: #ffffff !important; font-size: 1.8rem !important; }

    /* inputs */
    .stSelectbox select, .stNumberInput input { background: #1e2530 !important; color: #fff !important; border: 1px solid #2a3040 !important; border-radius: 8px; }

    /* botão predict */
    .stButton button { background: linear-gradient(135deg, #00e5a0, #00c9ff) !important; color: #0f1117 !important;
        font-weight: 700 !important; border-radius: 10px !important; padding: 10px 28px !important; border: none !important; font-size: 1rem !important; }
    .stButton button:hover { opacity: 0.88 !important; }

    /* resultado box */
    .resultado-box { background: linear-gradient(135deg, #1a2636, #161922); border: 1px solid #00e5a0;
        border-radius: 16px; padding: 28px; text-align: center; margin-top: 18px; }
    .resultado-box .preco { font-size: 2.6rem; font-weight: 800; color: #00e5a0; }
    .resultado-box .label { color: #7a8490; font-size: 0.85rem; margin-top: 4px; }

    /* tables */
    table { background: #161922 !important; color: #c8d0de !important; border-radius: 10px; }
    th { background: #1e2530 !important; color: #00e5a0 !important; }

    /* plots bg */
    .stPlotlyChart, .stPyplotChart { background: #161922; border-radius: 12px; border: 1px solid #2a3040; }

    /* tabs */
    .stTabs [role="tab"] { color: #7a8490 !important; border-bottom: 2px solid transparent !important; }
    .stTabs [role="tab"][aria-selected="true"] { color: #00e5a0 !important; border-bottom-color: #00e5a0 !important; }

    /* divider */
    hr { border-color: #2a3040 !important; }

    /* toast / info */
    .stInfo { background: #1a2636 !important; border: 1px solid #00c9ff !important; border-radius: 10px; color: #c8d0de !important; }
    .stSuccess { background: #14281f !important; border: 1px solid #00e5a0 !important; border-radius: 10px; color: #c8d0de !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# CARREGAR DADOS & TREINAR MODELO (cached)
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def carregar_dados():
    return pd.read_csv("dados_concessionaria.csv")

@st.cache_resource
def treinar_modelo(df):
    df_m = df.drop(columns=["ID","Modelo","Cor","Cidade","Estado"]).copy()

    cat_cols = ["Marca","Combustível","Condição"]
    les = {}
    for c in cat_cols:
        le = LabelEncoder()
        df_m[c] = le.fit_transform(df_m[c])
        les[c] = le

    X = df_m.drop(columns=["Preço (R$)"])
    y = df_m["Preço (R$)"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    model = LinearRegression().fit(X_tr_sc, y_tr)
    y_pred = model.predict(X_te_sc)

    metrics = {
        "R2":   r2_score(y_te, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_te, y_pred)),
        "MAE":  mean_absolute_error(y_te, y_pred)
    }
    return model, scaler, les, X.columns.tolist(), metrics, y_te, y_pred, X_te

df = carregar_dados()
model, scaler, les, feature_cols, metrics, y_te, y_pred, X_te = treinar_modelo(df)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("###Menu")
    st.markdown("---")
    pagina = st.radio("Seção", ["Início & Previsão", "Análise Exploradora", "Resultados do Modelo", "Dados Brutos"],
                      label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Projeto**")
    st.caption("Regressão Linear — Previsão de Preços de Veículos")
    st.caption(f"Dataset: {len(df):,} registros · 14 colunas")
    st.markdown("---")
    st.markdown("**Links**")
    st.markdown("• [GitHub](#)  · [LinkedIn](#)  · [Portfólio](#)")


# ═══════════════════════════════════════════════════════════════
# PÁGINA 1 — INÍCIO & PREVISÃO
# ═══════════════════════════════════════════════════════════════
if pagina == "Início & Previsão":

    st.markdown("#Previsão de Preço de Veículos")
    st.caption("Insira as características do veículo abaixo e receba a previsão de preço instantânea.")

    # ── KPIs topo ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de Registros", f"{len(df):,}")
    c2.metric("Preço Médio", f"R$ {df['Preço (R$)'].mean():,.0f}")
    c3.metric("R² do Modelo", f"{metrics['R2']:.4f}")
    c4.metric("RMSE", f"R$ {metrics['RMSE']:,.0f}")

    st.divider()

    # ── FORMULÁRIO DE ENTRADA ──
    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("###Dados do Veículo")

        marca = st.selectbox("Marca", sorted(df["Marca"].unique()))
        ano = st.slider("Ano de Fabricação", min_value=2012, max_value=2025, value=2020, step=1)
        combustivel = st.selectbox("Combustível", df["Combustível"].unique())
        potencia = st.slider("Potência (cv)", min_value=50, max_value=350, value=130, step=5)
        portas = st.selectbox("Número de Portas", [2, 4, 5])
        km = st.number_input("Quilômetros Rodados", min_value=0, max_value=500000, value=45000, step=1000)
        condicao = st.selectbox("Condição", ["Novo","Usado - Como Novo","Usado - Bom Estado","Usado - Regular"])
        garantia = st.selectbox("Garantia (meses)", [0, 6, 12, 24, 36, 48])

        predict_btn = st.button("Prever Preço", use_container_width=True)

    with col_result:
        st.markdown("###Resultado da Previsão")

        if predict_btn:
            # montar array
            entrada = {
                "Marca": les["Marca"].transform([marca])[0],
                "Ano": ano,
                "Combustível": les["Combustível"].transform([combustivel])[0],
                "Potência (cv)": potencia,
                "Número de Portas": portas,
                "Quilômetros": km,
                "Condição": les["Condição"].transform([condicao])[0],
                "Garantia (meses)": garantia
            }
            X_new = pd.DataFrame([entrada])[feature_cols]
            X_new_sc = scaler.transform(X_new)
            preco_pred = model.predict(X_new_sc)[0]
            preco_pred = max(preco_pred, 3500)

            st.markdown(f"""
            <div class="resultado-box">
                <div style="color:#7a8490; font-size:0.9rem; margin-bottom:4px;">Preço Estimado</div>
                <div class="preco">R$ {preco_pred:,.2f}</div>
                <div class="label">baseado no modelo de regressão linear múltipla</div>
            </div>
            """, unsafe_allow_html=True)

            # faixa estimativa (±RMSE)
            st.info(f"Faixa estimativa (±1 RMSE): **R$ {preco_pred - metrics['RMSE']:,.0f}** a **R$ {preco_pred + metrics['RMSE']:,.0f}**")

            # veículos similares
            st.markdown("#### Veículos Similares no Dataset")
            sim = df[(df["Marca"] == marca) & (df["Condição"] == condicao)].sort_values(
                by="Preço (R$)", key=lambda x: (x - preco_pred).abs()
            ).head(5)[["Marca","Modelo","Ano","Quilômetros","Condição","Preço (R$)"]]
            st.dataframe(sim, use_container_width=True, hide_index=True)
        else:
            st.info("Preenche os campos à esquerda e clica em **Prever Preço**.")


# ═══════════════════════════════════════════════════════════════
# PÁGINA 2 — EDA
# ═══════════════════════════════════════════════════════════════
elif pagina == "Análise Exploradora":

    st.markdown("# Análise Exploradora dos Dados (EDA)")

    # ── filtro lateral via sidebar dinâmico ──
    marcas_sel = st.multiselect("Filtrar Marcas", sorted(df["Marca"].unique()), default=sorted(df["Marca"].unique()))
    df_filt = df[df["Marca"].isin(marcas_sel)] if marcas_sel else df

    tab1, tab2, tab3, tab4 = st.tabs(["Distribuição", "Por Marca", "Correlações", "Extras"])

    # ── tab distribuição ──
    with tab1:
        st.markdown("### Distribuição do Preço")
        fig, ax = plt.subplots(figsize=(12,4.5))
        fig.patch.set_facecolor("#161922"); ax.set_facecolor("#161922")
        ax.hist(df_filt["Preço (R$)"], bins=70, color="#00e5a0", edgecolor="#0f1117", alpha=0.85)
        ax.axvline(df_filt["Preço (R$)"].mean(), color="#ff6b6b", lw=2, ls="--", label=f'Média: R$ {df_filt["Preço (R$)"].mean():,.0f}')
        ax.axvline(df_filt["Preço (R$)"].median(), color="#f0a500", lw=2, ls="--", label=f'Mediana: R$ {df_filt["Preço (R$)"].median():,.0f}')
        ax.set_xlabel("Preço (R$)", color="#c8d0de"); ax.set_ylabel("Frequência", color="#c8d0de")
        ax.tick_params(colors="#7a8490"); ax.legend(facecolor="#1e2530", edgecolor="#2a3040", labelcolor="#c8d0de")
        ax.spines["bottom"].set_color("#2a3040"); ax.spines["left"].set_color("#2a3040")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(color="#1e2530")
        st.pyplot(fig); plt.close()

    # ── tab por marca ──
    with tab2:
        st.markdown("### Preço Médio por Marca")
        media = df_filt.groupby("Marca")["Preço (R$)"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(11, max(4, len(media)*0.4)))
        fig.patch.set_facecolor("#161922"); ax.set_facecolor("#161922")
        ax.barh(media.index, media.values, color="#00c9ff", edgecolor="#0f1117", height=0.6)
        ax.set_xlabel("Preço Médio (R$)", color="#c8d0de")
        ax.tick_params(colors="#c8d0de"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#2a3040"); ax.spines["left"].set_color("#2a3040")
        ax.grid(color="#1e2530", axis="x")
        for i, v in enumerate(media.values):
            ax.text(v+300, i, f"R$ {v:,.0f}", va="center", fontsize=9, color="#c8d0de")
        st.pyplot(fig); plt.close()

    # ── tab correlações ──
    with tab3:
        st.markdown("### Heatmap de Correlações")
        num_cols = ["Ano","Potência (cv)","Número de Portas","Quilômetros","Garantia (meses)","Preço (R$)"]
        fig, ax = plt.subplots(figsize=(9,7))
        fig.patch.set_facecolor("#161922")
        sns.heatmap(df_filt[num_cols].corr(), annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                    vmin=-1, vmax=1, linewidths=0.8, linecolor="#0f1117", ax=ax,
                    cbar_kws={"shrink":0.85}, annot_kws={"size":11,"color":"#0f1117"})
        ax.set_facecolor("#161922"); ax.tick_params(colors="#c8d0de")
        st.pyplot(fig); plt.close()

    # ── tab extras ──
    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Preço por Condição")
            fig, ax = plt.subplots(figsize=(7,4.5))
            fig.patch.set_facecolor("#161922"); ax.set_facecolor("#161922")
            order = ["Novo","Usado - Como Novo","Usado - Bom Estado","Usado - Regular"]
            cols_b = ["#00e5a0","#00c9ff","#f0a500","#ff6b6b"]
            bp = ax.boxplot([df_filt[df_filt["Condição"]==c]["Preço (R$)"].values for c in order],
                            labels=[c.replace(" - ","\n") for c in order], patch_artist=True, widths=0.5,
                            medianprops=dict(color="#fff",linewidth=2))
            for p, col in zip(bp["boxes"], cols_b): p.set_facecolor(col); p.set_alpha(0.6)
            ax.tick_params(colors="#c8d0de"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#2a3040"); ax.spines["left"].set_color("#2a3040")
            ax.set_ylabel("Preço (R$)", color="#c8d0de"); ax.grid(color="#1e2530")
            st.pyplot(fig); plt.close()

        with c2:
            st.markdown("### Preço por Combustível")
            fig, ax = plt.subplots(figsize=(7,4.5))
            fig.patch.set_facecolor("#161922"); ax.set_facecolor("#161922")
            mc = df_filt.groupby("Combustível")["Preço (R$)"].mean().sort_values(ascending=False)
            ax.bar(mc.index, mc.values, color=["#00e5a0","#00c9ff","#f0a500","#ff6b6b","#a78bfa"], edgecolor="#0f1117")
            ax.set_ylabel("Preço Médio (R$)", color="#c8d0de")
            ax.tick_params(colors="#c8d0de"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#2a3040"); ax.spines["left"].set_color("#2a3040")
            ax.grid(color="#1e2530", axis="y")
            for i, v in enumerate(mc.values):
                ax.text(i, v+200, f"R$ {v:,.0f}", ha="center", fontsize=9, color="#c8d0de")
            st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════════
# PÁGINA 3 — RESULTADOS DO MODELO
# ═══════════════════════════════════════════════════════════════
elif pagina == "Resultados do Modelo":

    st.markdown("#Resultados do Modelo")

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("R² (Múltipla)", f"{metrics['R2']:.4f}", delta="+0.49 vs simples")
    c2.metric("RMSE", f"R$ {metrics['RMSE']:,.0f}")
    c3.metric("MAE", f"R$ {metrics['MAE']:,.0f}")

    tab1, tab2, tab3 = st.tabs(["Real vs Predito", "Resíduos", "Importância"])

    with tab1:
        st.markdown("### Preço Real vs Preço Predito")
        fig, ax = plt.subplots(figsize=(8,7))
        fig.patch.set_facecolor("#161922"); ax.set_facecolor("#161922")
        ax.scatter(y_te, y_pred, c="#00c9ff", alpha=0.4, s=18, edgecolors="none")
        lim = [min(y_te.min(), y_pred.min())-1000, max(y_te.max(), y_pred.max())+1000]
        ax.plot(lim, lim, color="#00e5a0", lw=2, ls="--", label="Ideal")
        ax.set_xlabel("Preço Real (R$)", color="#c8d0de"); ax.set_ylabel("Preço Predito (R$)", color="#c8d0de")
        ax.tick_params(colors="#7a8490"); ax.legend(facecolor="#1e2530", edgecolor="#2a3040", labelcolor="#c8d0de")
        ax.spines["bottom"].set_color("#2a3040"); ax.spines["left"].set_color("#2a3040")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(color="#1e2530")
        st.pyplot(fig); plt.close()

    with tab2:
        st.markdown("### Análise de Resíduos")
        residuos = y_te.values - y_pred
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6,5))
            fig.patch.set_facecolor("#161922"); ax.set_facecolor("#161922")
            ax.scatter(y_pred, residuos, c="#f0a500", alpha=0.35, s=14, edgecolors="none")
            ax.axhline(0, color="#ff6b6b", lw=1.8, ls="--")
            ax.set_xlabel("Predito", color="#c8d0de"); ax.set_ylabel("Resíduo", color="#c8d0de")
            ax.tick_params(colors="#7a8490"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#2a3040"); ax.spines["left"].set_color("#2a3040"); ax.grid(color="#1e2530")
            st.pyplot(fig); plt.close()
        with c2:
            fig, ax = plt.subplots(figsize=(6,5))
            fig.patch.set_facecolor("#161922"); ax.set_facecolor("#161922")
            ax.hist(residuos, bins=55, color="#00e5a0", edgecolor="#0f1117", alpha=0.8)
            ax.axvline(0, color="#ff6b6b", lw=2, ls="--")
            ax.set_xlabel("Resíduo", color="#c8d0de"); ax.set_ylabel("Frequência", color="#c8d0de")
            ax.tick_params(colors="#7a8490"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#2a3040"); ax.spines["left"].set_color("#2a3040"); ax.grid(color="#1e2530")
            st.pyplot(fig); plt.close()

    with tab3:
        st.markdown("### Importância das Variáveis (Coeficientes)")
        coef_df = pd.DataFrame({"Feature": feature_cols, "Coeficiente": model.coef_}).sort_values("Coeficiente", key=abs, ascending=False)
        fig, ax = plt.subplots(figsize=(9,5))
        fig.patch.set_facecolor("#161922"); ax.set_facecolor("#161922")
        colors_c = ["#00e5a0" if v > 0 else "#ff6b6b" for v in coef_df["Coeficiente"]]
        ax.barh(coef_df["Feature"], coef_df["Coeficiente"], color=colors_c, edgecolor="#0f1117", height=0.5)
        ax.axvline(0, color="#7a8490", lw=1)
        ax.set_xlabel("Coeficiente Padronizado", color="#c8d0de")
        ax.tick_params(colors="#c8d0de"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#2a3040"); ax.spines["left"].set_color("#2a3040"); ax.grid(color="#1e2530", axis="x")
        st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════════
# PÁGINA 4 — DADOS BRUTOS
# ═══════════════════════════════════════════════════════════════
elif pagina == "Dados Brutos":

    st.markdown("#Dados Brutos")

    # filtros
    c1, c2, c3 = st.columns(3)
    marca_filt = c1.multiselect("Marca", sorted(df["Marca"].unique()))
    cond_filt  = c2.multiselect("Condição", df["Condição"].unique())
    comb_filt  = c3.multiselect("Combustível", df["Combustível"].unique())

    df_show = df.copy()
    if marca_filt: df_show = df_show[df_show["Marca"].isin(marca_filt)]
    if cond_filt:  df_show = df_show[df_show["Condição"].isin(cond_filt)]
    if comb_filt:  df_show = df_show[df_show["Combustível"].isin(comb_filt)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Registros Filtrados", f"{len(df_show):,}")
    c2.metric("Preço Médio (filtro)", f"R$ {df_show['Preço (R$)'].mean():,.0f}")
    c3.metric("Preço Máximo (filtro)", f"R$ {df_show['Preço (R$)'].max():,.0f}")

    st.dataframe(df_show.sort_values("Preço (R$)", ascending=False).head(500),
                 use_container_width=True, hide_index=True)

    # download
    st.download_button("Baixar CSV Filtrado", df_show.to_csv(index=False).encode("utf-8"),
                       file_name="dados_filtrados.csv", mime="text/csv")
