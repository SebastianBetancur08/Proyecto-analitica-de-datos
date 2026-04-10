"""
app.py — Dashboard EDA: Clasificación (IBM HR Attrition) + Regresión (CGPA)
Ejecutar: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st
from pathlib import Path
from itertools import combinations, product
from scipy import stats
from scipy.stats import chi2_contingency, spearmanr, kruskal
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

# ──────────────────────────────────────────────────────────────────────────────
# Configuración de página
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDA Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})
sns.set_style("whitegrid")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent

def _find_csv(name: str) -> Path:
    for candidate in [
        ROOT / name,
        ROOT / "data" / "raw" / name,
        ROOT.parent / "data" / "raw" / name,
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No se encontró '{name}'. Colócalo junto a app.py o en data/raw/")


@st.cache_data
def load_clasificacion() -> pd.DataFrame:
    df = pd.read_csv(_find_csv("dataset_clasificacion.csv"))
    str_cols = df.select_dtypes("object").columns
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip().str.lower()
    return df


@st.cache_data
def load_regresion() -> pd.DataFrame:
    df = pd.read_csv(_find_csv("dataset_regresion.csv"))
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
    columnas = list(df.columns)
    correcciones = {
        columnas[25]: (df[columnas[25]].unique()[2],  "No"),
        columnas[26]: (df[columnas[26]].unique()[135], 1.42),
        columnas[28]: (df[columnas[28]].unique()[161], 3.1),
    }
    for col, (mal, bien) in correcciones.items():
        df[col] = df[col].replace(mal, bien)
    cols_num = df.select_dtypes("number").columns
    df[cols_num] = df[cols_num].fillna(df[cols_num].mean())
    return df


def show_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.title("EDA Dashboard")
    dataset = st.radio("Dataset", ["📋 Clasificación — IBM HR", "📈 Regresión — CGPA"])
    st.divider()
    st.caption("Proyecto Analítica de Datos")

# ══════════════════════════════════════════════════════════════════════════════
#  CLASIFICACIÓN
# ══════════════════════════════════════════════════════════════════════════════
if dataset.startswith("📋"):
    st.title("📋 EDA — Clasificación: Rotación de Empleados (IBM HR)")
    st.markdown("> **Variable objetivo:** `Attrition`  |  **Fuente:** [Kaggle IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)")

    try:
        df = load_clasificacion()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    num_cols = [
        "Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyIncome",
        "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "TotalWorkingYears",
        "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole",
        "YearsSinceLastPromotion", "YearsWithCurrManager",
    ]
    ord_cols = [
        "Education", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel",
        "JobSatisfaction", "PerformanceRating", "RelationshipSatisfaction",
        "StockOptionLevel", "WorkLifeBalance",
    ]
    cat_cols = ["Attrition", "BusinessTravel", "Department", "EducationField",
                "Gender", "JobRole", "MaritalStatus", "OverTime"]
    palette = {"yes": "#e74c3c", "no": "#3498db"}

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1. Descripción", "2. Distribuciones", "3. Outliers",
        "4. Correlaciones", "5. Relaciones × Attrition", "6. Pruebas estadísticas"
    ])

    # ── Tab 1: Descripción ────────────────────────────────────────────────────
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Observaciones", df.shape[0])
        c2.metric("Variables", df.shape[1])
        c3.metric("Attrition Yes", f"{(df['Attrition']=='yes').sum()} ({(df['Attrition']=='yes').mean()*100:.1f}%)")
        st.divider()
        st.subheader("Vista previa del dataset")
        st.dataframe(df.head(20), use_container_width=True)
        st.divider()
        st.subheader("Estadísticas descriptivas")
        desc = df[num_cols + ord_cols].agg(["mean", "median", "std"]).T.round(2)
        desc.columns = ["Media", "Mediana", "Desv. Estándar"]
        st.dataframe(desc, use_container_width=True)

    # ── Tab 2: Distribuciones ─────────────────────────────────────────────────
    with tab2:
        st.subheader("Variables Numéricas Continuas")
        fig, axes = plt.subplots(4, 4, figsize=(18, 14))
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            ax = axes[i]
            ax.hist(df[col].dropna(), bins=28, color="steelblue", edgecolor="white", alpha=0.85)
            ax.axvline(df[col].mean(),   color="red",    linestyle="--", lw=1.3, label="Media")
            ax.axvline(df[col].median(), color="orange", linestyle="-",  lw=1.3, label="Mediana")
            ax.set_title(col, fontweight="bold", fontsize=9)
            if i == 0: ax.legend(fontsize=7)
        for j in range(len(num_cols), len(axes)): axes[j].set_visible(False)
        fig.suptitle("Distribuciones — Variables Numéricas", fontsize=12, fontweight="bold")
        plt.tight_layout()
        show_fig(fig)

        st.subheader("Variables Ordinales")
        ord_labels = {
            "Education":               {1:"Below College", 2:"College", 3:"Bachelor", 4:"Master", 5:"Doctor"},
            "EnvironmentSatisfaction": {1:"Low", 2:"Medium", 3:"High", 4:"Very High"},
            "JobInvolvement":          {1:"Low", 2:"Medium", 3:"High", 4:"Very High"},
            "JobLevel":                {1:"Entry", 2:"Junior", 3:"Mid", 4:"Senior", 5:"C-Level"},
            "JobSatisfaction":         {1:"Low", 2:"Medium", 3:"High", 4:"Very High"},
            "PerformanceRating":       {1:"Low", 2:"Good", 3:"Excellent", 4:"Outstanding"},
            "RelationshipSatisfaction":{1:"Low", 2:"Medium", 3:"High", 4:"Very High"},
            "StockOptionLevel":        {0:"None", 1:"Low", 2:"Medium", 3:"High"},
            "WorkLifeBalance":         {1:"Bad", 2:"Good", 3:"Better", 4:"Best"},
        }
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        axes = axes.flatten()
        for i, col in enumerate(ord_cols):
            ax = axes[i]
            vc = df[col].value_counts().sort_index()
            labels = [ord_labels[col].get(k, str(k)) for k in vc.index]
            colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(vc)))
            bars = ax.bar(labels, vc.values, color=colors, edgecolor="white")
            for bar, val in zip(bars, vc.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+3, str(val),
                        ha="center", fontsize=8)
            ax.set_title(col, fontweight="bold", fontsize=9)
            ax.tick_params(axis="x", rotation=20)
        fig.suptitle("Distribuciones — Variables Ordinales", fontsize=12, fontweight="bold")
        plt.tight_layout()
        show_fig(fig)

        st.subheader("Variables Categóricas Nominales")
        fig, axes = plt.subplots(2, 4, figsize=(20, 9))
        axes = axes.flatten()
        for i, col in enumerate(cat_cols):
            ax = axes[i]
            vc = df[col].value_counts()
            colors = plt.cm.Set2(np.linspace(0, 1, len(vc)))
            bars = ax.barh(vc.index, vc.values, color=colors, edgecolor="white")
            for bar, val in zip(bars, vc.values):
                ax.text(bar.get_width()+2, bar.get_y()+bar.get_height()/2, str(val),
                        va="center", fontsize=8)
            ax.set_title(col, fontweight="bold", fontsize=9)
        fig.suptitle("Distribuciones — Variables Categóricas", fontsize=12, fontweight="bold")
        plt.tight_layout()
        show_fig(fig)

    # ── Tab 3: Outliers ───────────────────────────────────────────────────────
    with tab3:
        st.subheader("Boxplots — Detección de Valores Atípicos")
        fig, axes = plt.subplots(4, 4, figsize=(18, 14))
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            ax = axes[i]
            data = df[col].dropna()
            ax.boxplot(data, vert=True, patch_artist=True, widths=0.5,
                       boxprops=dict(facecolor="steelblue", alpha=0.6),
                       medianprops=dict(color="red", linewidth=2),
                       flierprops=dict(marker="o", color="tomato", alpha=0.5, markersize=3))
            Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
            IQR = Q3 - Q1
            n_out = ((data < Q1-1.5*IQR) | (data > Q3+1.5*IQR)).sum()
            ax.set_title(f"{col}\n({n_out} atípicos)", fontweight="bold", fontsize=8)
            ax.set_xticks([])
        for j in range(len(num_cols), len(axes)): axes[j].set_visible(False)
        fig.suptitle("Valores Atípicos — Variables Numéricas", fontsize=12, fontweight="bold")
        plt.tight_layout()
        show_fig(fig)

        st.subheader("Tabla resumen de outliers")
        rows = []
        for col in num_cols:
            data = df[col].dropna()
            Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
            IQR = Q3 - Q1
            lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
            n_out = ((data < lo) | (data > hi)).sum()
            rows.append({"Variable": col, "Q1": round(Q1,2), "Q3": round(Q3,2),
                         "IQR": round(IQR,2), "Límite inf": round(lo,2),
                         "Límite sup": round(hi,2), "# Outliers": n_out})
        st.dataframe(pd.DataFrame(rows).set_index("Variable"), use_container_width=True)

    # ── Tab 4: Correlaciones ──────────────────────────────────────────────────
    with tab4:
        st.subheader("Mapa de calor — Correlación de Pearson")
        all_num = num_cols + ord_cols
        corr = df[all_num].corr()
        fig, ax = plt.subplots(figsize=(16, 13))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, linewidths=0.4, linecolor="white",
                    annot_kws={"size": 7}, ax=ax, vmin=-1, vmax=1,
                    cbar_kws={"label": "Pearson r"})
        ax.set_title("Matriz de Correlación", fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        show_fig(fig)

        st.subheader("Scatter: Tasas de ingreso vs MonthlyIncome")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        attrition_colors = df["Attrition"].map({"yes": "#e74c3c", "no": "#3498db"})
        for ax, xcol in zip(axes, ["DailyRate", "HourlyRate"]):
            m, b, r, p_val, _ = stats.linregress(df[xcol], df["MonthlyIncome"])
            ax.scatter(df[xcol], df["MonthlyIncome"], c=attrition_colors,
                       alpha=0.4, s=20, edgecolors="none")
            xs = np.linspace(df[xcol].min(), df[xcol].max(), 200)
            ax.plot(xs, m*xs+b, color="black", lw=1.5, linestyle="--",
                    label=f"r={r:.3f}")
            ax.set_title(f"{xcol} vs MonthlyIncome", fontweight="bold")
            ax.set_xlabel(xcol); ax.set_ylabel("MonthlyIncome")
            ax.legend(fontsize=9)
        plt.tight_layout()
        show_fig(fig)

    # ── Tab 5: Relaciones × Attrition ─────────────────────────────────────────
    with tab5:
        st.subheader("Ingresos por Attrition")
        income_vars = ["MonthlyIncome", "DailyRate", "HourlyRate", "MonthlyRate"]
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        for i, col in enumerate(income_vars):
            sns.boxplot(data=df, x="Attrition", y=col, palette=palette,
                        order=["yes","no"], ax=axes[i], linewidth=1,
                        flierprops=dict(marker="o", markersize=2, alpha=0.4))
            axes[i].set_title(col, fontweight="bold", fontsize=9)
            axes[i].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        fig.suptitle("Variables de Ingreso según Attrition", fontsize=12, fontweight="bold")
        plt.tight_layout()
        show_fig(fig)

        st.subheader("Antigüedad y Edad por Attrition")
        years_vars = ["Age","TotalWorkingYears","YearsAtCompany",
                      "YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager"]
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        for i, col in enumerate(years_vars):
            sns.boxplot(data=df, x="Attrition", y=col, palette=palette,
                        order=["yes","no"], ax=axes[i], linewidth=1,
                        flierprops=dict(marker="o", markersize=2, alpha=0.4))
            axes[i].set_title(col, fontweight="bold", fontsize=9)
        fig.suptitle("Edad y Años de Experiencia según Attrition", fontsize=12, fontweight="bold")
        plt.tight_layout()
        show_fig(fig)

        st.subheader("Satisfacción laboral por Attrition")
        sat_vars = {
            "EnvironmentSatisfaction": "1=Low…4=VH",
            "JobSatisfaction": "1=Low…4=VH",
            "RelationshipSatisfaction": "1=Low…4=VH",
            "WorkLifeBalance": "1=Bad…4=Best",
            "JobInvolvement": "1=Low…4=VH",
        }
        fig, axes = plt.subplots(1, 5, figsize=(22, 6))
        for i, (col, scale) in enumerate(sat_vars.items()):
            sns.boxplot(data=df, x="Attrition", y=col, palette=palette,
                        order=["yes","no"], ax=axes[i], linewidth=1,
                        flierprops=dict(marker="o", markersize=2, alpha=0.4))
            axes[i].set_title(f"{col}\n({scale})", fontweight="bold", fontsize=8)
        fig.suptitle("Satisfacción y Bienestar según Attrition", fontsize=12, fontweight="bold")
        plt.tight_layout()
        show_fig(fig)

        st.subheader("MonthlyIncome por Departamento y JobRole")
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        order_dept = df.groupby("Department")["MonthlyIncome"].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x="Department", y="MonthlyIncome", order=order_dept,
                    palette="Set2", ax=axes[0], linewidth=1)
        axes[0].set_title("Por Departamento", fontweight="bold")
        axes[0].tick_params(axis="x", rotation=15)
        order_role = df.groupby("JobRole")["MonthlyIncome"].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x="JobRole", y="MonthlyIncome", order=order_role,
                    palette="tab10", ax=axes[1], linewidth=1)
        axes[1].set_title("Por JobRole", fontweight="bold")
        axes[1].tick_params(axis="x", rotation=40)
        for ax in axes:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        fig.suptitle("Distribución de MonthlyIncome", fontsize=12, fontweight="bold")
        plt.tight_layout()
        show_fig(fig)

    # ── Tab 6: Pruebas estadísticas ───────────────────────────────────────────
    with tab6:
        st.subheader("Chi-cuadrado — Variables Categóricas vs Attrition")
        cat_test = ["BusinessTravel","Department","EducationField","Gender",
                    "JobRole","MaritalStatus","OverTime"]
        rows_chi = []
        for var in cat_test:
            tabla = pd.crosstab(df[var], df["Attrition"])
            chi2, p, dof, _ = chi2_contingency(tabla)
            rows_chi.append({"Variable": var, "Chi²": round(chi2,4),
                             "p-valor": round(p,4), "gl": dof,
                             "Significativa (α=0.05)": "✔ Sí" if p < 0.05 else "✘ No"})
        df_chi = pd.DataFrame(rows_chi).set_index("Variable").sort_values("p-valor")
        st.dataframe(df_chi.style
            .applymap(lambda v: "color:#c0392b;font-weight:bold" if v=="✔ Sí"
                      else ("color:#27ae60;font-weight:bold" if v=="✘ No" else ""),
                      subset=["Significativa (α=0.05)"])
            .format({"Chi²":"{:.4f}","p-valor":"{:.4f}"}),
            use_container_width=True)

        st.divider()
        st.subheader("Correlación de Spearman — Variables Numéricas vs Attrition")
        df_temp = df.copy()
        df_temp["Attrition_bin"] = (df_temp["Attrition"] == "yes").astype(int)
        rows_sp = []
        for col in num_cols + ord_cols:
            t = df_temp[[col,"Attrition_bin"]].dropna()
            coef, p = spearmanr(t[col], t["Attrition_bin"])
            rows_sp.append({"Variable": col, "ρ": round(coef,5),
                            "p-valor": round(p,4),
                            "Significativa": "✔ Sí" if p < 0.05 else "✘ No"})
        df_sp = pd.DataFrame(rows_sp).set_index("Variable").sort_values("p-valor")
        st.dataframe(df_sp.style
            .background_gradient(subset=["ρ"], cmap="RdBu_r")
            .applymap(lambda v: "color:#c0392b;font-weight:bold" if v=="✔ Sí"
                      else ("color:#27ae60;font-weight:bold" if v=="✘ No" else ""),
                      subset=["Significativa"])
            .format({"ρ":"{:.5f}","p-valor":"{:.4f}"}),
            use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  REGRESIÓN
# ══════════════════════════════════════════════════════════════════════════════
else:
    TARGET = "What is your current CGPA?"
    st.title("📈 EDA — Regresión: Rendimiento Académico (CGPA)")
    st.markdown("> **Variable objetivo:** `CGPA`  |  **Fuente:** [IUBAT Students Performance — Mendeley](https://data.mendeley.com/datasets/ns87rtkv58/2)")

    try:
        df = load_regresion()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    def identificar_tipos(df, umbral=8):
        cat, num = [], []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() <= umbral and df[col].nunique() < len(df)*0.05:
                    cat.append(col)
                else:
                    num.append(col)
            else:
                cat.append(col)
        return cat, num

    EXCLUIR = ["Program","What are the skills do you have ?","What is you interested area?"]
    categoricas_raw, numericas = identificar_tipos(df)
    categoricas = [v for v in categoricas_raw if v not in EXCLUIR]

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1. Descripción", "2. Categóricas", "3. Numéricas",
        "4. Histogramas", "5. Correlaciones", "6. Pruebas estadísticas"
    ])

    # ── Tab 1 ─────────────────────────────────────────────────────────────────
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Observaciones", df.shape[0])
        c2.metric("Variables", df.shape[1])
        c3.metric("CGPA promedio", f"{df[TARGET].mean():.3f}")
        st.divider()
        st.subheader("Vista previa")
        st.dataframe(df.head(20), use_container_width=True)

        st.divider()
        st.subheader("Valores faltantes")
        miss = df.isnull().sum()
        miss = miss[miss > 0]
        if miss.empty:
            st.success("No hay valores faltantes.")
        else:
            miss_df = miss.rename("Nulos").to_frame()
            miss_df["%"] = (miss_df["Nulos"]/len(df)*100).round(2)
            st.dataframe(miss_df, use_container_width=True)

    # ── Tab 2: Categóricas ────────────────────────────────────────────────────
    with tab2:
        st.subheader("Distribución de frecuencias")
        n_cols2 = 3
        n_filas2 = int(np.ceil(len(categoricas) / n_cols2))
        fig, axes = plt.subplots(n_filas2, n_cols2, figsize=(18, 4*n_filas2))
        axes = axes.flatten()
        for i, var in enumerate(categoricas):
            order = df[var].value_counts().index
            sns.countplot(data=df.dropna(subset=[var]), x=var, order=order,
                          ax=axes[i], palette="Blues_r", edgecolor="black")
            axes[i].set_title(var, fontsize=8, fontweight="bold")
            axes[i].set_xlabel("")
            axes[i].tick_params(axis="x", rotation=35, labelsize=7)
        for j in range(i+1, len(axes)): axes[j].set_visible(False)
        fig.suptitle("Variables Categóricas — Frecuencias", fontsize=13, fontweight="bold")
        plt.tight_layout()
        show_fig(fig)

        st.subheader("Boxplots CGPA por categoría")
        n_cols2 = 2
        n_filas2 = int(np.ceil(len(categoricas) / n_cols2))
        fig, axes = plt.subplots(n_filas2, n_cols2, figsize=(16, 5*n_filas2))
        axes = axes.flatten()
        for i, var in enumerate(categoricas):
            temp = df[[var, TARGET]].dropna()
            if temp.empty:
                axes[i].set_visible(False)
                continue
            order = temp.groupby(var)[TARGET].median().sort_values(ascending=False).index
            sns.boxplot(data=temp, x=var, y=TARGET, order=order,
                        palette="Blues", flierprops=dict(marker="o", alpha=0.4), ax=axes[i])
            axes[i].set_title(f"CGPA por {var}", fontsize=9, fontweight="bold")
            axes[i].set_xlabel("")
            axes[i].tick_params(axis="x", rotation=35, labelsize=7)
        for j in range(i+1, len(axes)): axes[j].set_visible(False)
        fig.suptitle("CGPA por Variables Categóricas", fontsize=13, fontweight="bold")
        plt.tight_layout()
        show_fig(fig)

    # ── Tab 3: Numéricas ──────────────────────────────────────────────────────
    with tab3:
        st.subheader("Estadísticas descriptivas")
        rows_desc = []
        for var in numericas:
            s = df[var].dropna()
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            rows_desc.append({"Variable": var, "n": len(s),
                "Media": round(s.mean(),4), "Mediana": round(s.median(),4),
                "Std": round(s.std(),4), "Q1": round(Q1,4), "Q3": round(Q3,4),
                "IQR": round(Q3-Q1,4), "Min": round(s.min(),4), "Max": round(s.max(),4)})
        st.dataframe(pd.DataFrame(rows_desc).set_index("Variable"), use_container_width=True)

        st.subheader("Outliers (IQR)")
        rows_out = []
        for var in numericas:
            s = df[var].dropna()
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR = Q3 - Q1
            lb, ub = Q1-1.5*IQR, Q3+1.5*IQR
            n_out = ((s < lb) | (s > ub)).sum()
            rows_out.append({"Variable": var, "N Outliers": n_out,
                             "Límite inf": round(lb,4), "Límite sup": round(ub,4),
                             "% Outliers": round(n_out/len(s)*100,2)})
        st.dataframe(pd.DataFrame(rows_out).set_index("Variable"), use_container_width=True)

    # ── Tab 4: Histogramas ────────────────────────────────────────────────────
    with tab4:
        st.subheader("Histogramas — Variables Numéricas")

        def _bins(data):
            if data.nunique() <= 20: return data.nunique()
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            if iqr == 0: return 20
            bw = 2*iqr/len(data)**(1/3)
            return int(np.clip(np.ceil((data.max()-data.min())/bw), 10, 50))

        n_cols3 = 3
        n_filas3 = int(np.ceil(len(numericas) / n_cols3))
        fig, axes = plt.subplots(n_filas3, n_cols3, figsize=(18, 5*n_filas3))
        axes = axes.flatten()
        for i, col in enumerate(numericas):
            ax = axes[i]
            data = df[col].dropna()
            Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
            IQR = Q3 - Q1
            lb, ub = Q1-1.5*IQR, Q3+1.5*IQR
            data_plot = data[(data >= lb) & (data <= ub)]
            n_out = len(data) - len(data_plot)
            ax.hist(data_plot, bins=_bins(data_plot), color="steelblue",
                    edgecolor="black", alpha=0.85)
            ax.axvline(data.mean(),   color="green", linestyle="--", lw=1.3,
                       label=f"Media:{data.mean():.2f}")
            ax.axvline(data.median(), color="red",   linestyle="--", lw=1.3,
                       label=f"Med:{data.median():.2f}")
            sufijo = f"\n(sin {n_out} out.)" if n_out else ""
            ax.set_title(f"{col}{sufijo}", fontsize=8, fontweight="bold")
            ax.legend(fontsize=6)
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(True, alpha=0.2)
        for j in range(i+1, len(axes)): axes[j].set_visible(False)
        fig.suptitle("Histogramas sin Outliers", fontsize=13, fontweight="bold")
        plt.tight_layout()
        show_fig(fig)

        st.subheader("Scatter plots vs CGPA")
        vars_sc = [v for v in numericas if v != TARGET]
        n_cols3 = 3
        n_filas3 = int(np.ceil(len(vars_sc) / n_cols3))
        fig, axes = plt.subplots(n_filas3, n_cols3, figsize=(18, 4*n_filas3))
        axes = axes.flatten()
        for i, var in enumerate(vars_sc):
            temp = df[[var, TARGET]].dropna()
            axes[i].scatter(temp[var], temp[TARGET], alpha=0.4, s=15,
                            color="steelblue", edgecolors="k", linewidths=0.2)
            m, b = np.polyfit(temp[var], temp[TARGET], 1)
            xs = np.linspace(temp[var].min(), temp[var].max(), 100)
            axes[i].plot(xs, m*xs+b, color="red", lw=1.5, linestyle="--")
            axes[i].set_title(var, fontsize=8, fontweight="bold")
            axes[i].set_xlabel(var, fontsize=7)
            axes[i].set_ylabel("CGPA", fontsize=7)
            axes[i].grid(True, alpha=0.2)
        for j in range(i+1, len(axes)): axes[j].set_visible(False)
        fig.suptitle("Scatter plots vs CGPA", fontsize=13, fontweight="bold")
        plt.tight_layout()
        show_fig(fig)

    # ── Tab 5: Correlaciones ──────────────────────────────────────────────────
    with tab5:
        st.subheader("Correlación con CGPA — Numéricas")
        df_num = df[numericas].dropna()
        fig, axes = plt.subplots(1, 3, figsize=(18, max(5, len(numericas)*0.45)), sharey=True)
        for ax, metodo in zip(axes, ["spearman","kendall","pearson"]):
            corr = df_num.corr(method=metodo)[TARGET].drop(TARGET).sort_values()
            colors = ["#e74c3c" if v < 0 else "#2980b9" for v in corr]
            corr.plot(kind="barh", color=colors, edgecolor="black", ax=ax)
            ax.axvline(0, color="black", lw=0.8, linestyle="--")
            ax.set_title(metodo.capitalize(), fontsize=11, fontweight="bold")
        fig.suptitle(f"Correlaciones con CGPA", fontsize=13, fontweight="bold")
        plt.tight_layout()
        show_fig(fig)

        st.subheader("Correlación con CGPA — Categóricas (codificadas)")
        df_enc = df[categoricas + [TARGET]].dropna().copy()
        for col in categoricas:
            df_enc[col] = LabelEncoder().fit_transform(df_enc[col])
        corr_cat = df_enc.corr(method="spearman")[TARGET].drop(TARGET).sort_values()
        colors = ["#e74c3c" if v < 0 else "#2980b9" for v in corr_cat]
        fig, ax = plt.subplots(figsize=(10, max(4, len(corr_cat)*0.45)))
        corr_cat.plot(kind="barh", color=colors, edgecolor="black", ax=ax)
        ax.axvline(0, color="black", lw=0.8, linestyle="--")
        ax.set_title("Spearman — Categóricas vs CGPA", fontsize=12, fontweight="bold")
        plt.tight_layout()
        show_fig(fig)

    # ── Tab 6: Pruebas estadísticas ───────────────────────────────────────────
    with tab6:
        alpha = 0.05

        st.subheader("Chi-cuadrado — Variables Categóricas")
        with st.spinner("Calculando Chi-cuadrado..."):
            rows_chi = []
            for v1, v2 in combinations(categoricas, 2):
                tabla = pd.crosstab(df[v1], df[v2])
                chi2_val, p, dof, _ = chi2_contingency(tabla)
                rows_chi.append({"Variable 1": v1, "Variable 2": v2,
                                  "Chi²": round(chi2_val,4), "p-valor": round(p,4),
                                  "gl": dof, "Dependencia": "✔ Sí" if p < alpha else "✘ No"})
            df_chi = pd.DataFrame(rows_chi).sort_values("p-valor").reset_index(drop=True)
        dep = (df_chi["Dependencia"]=="✔ Sí").sum()
        st.info(f"**{dep} pares dependientes** de {len(df_chi)} totales (α={alpha})")
        st.dataframe(df_chi.style
            .applymap(lambda v: "color:#c0392b;font-weight:bold" if v=="✔ Sí"
                      else ("color:#27ae60;font-weight:bold" if v=="✘ No" else ""),
                      subset=["Dependencia"])
            .format({"Chi²":"{:.4f}","p-valor":"{:.4f}"}),
            use_container_width=True, height=400)

        st.divider()
        st.subheader("Spearman — Variables Numéricas")
        with st.spinner("Calculando Spearman..."):
            def _fuerza(c):
                a = abs(c)
                if a < 0.3: return "Débil"
                if a < 0.7: return "Moderada"
                return "Fuerte"
            rows_sp = []
            for v1, v2 in combinations(numericas, 2):
                temp = df[[v1,v2]].dropna()
                coef, p = spearmanr(temp[v1], temp[v2])
                rows_sp.append({"Variable 1": v1, "Variable 2": v2,
                                 "ρ": round(coef,5), "|ρ|": round(abs(coef),5),
                                 "Fuerza": _fuerza(coef),
                                 "Dirección": "Positiva" if coef>=0 else "Negativa",
                                 "p-valor": round(p,4),
                                 "Asociación": "✔ Sí" if p < alpha else "✘ No"})
            df_sp = pd.DataFrame(rows_sp).sort_values("p-valor").reset_index(drop=True)
        asoc = (df_sp["Asociación"]=="✔ Sí").sum()
        st.info(f"**{asoc} pares con asociación** de {len(df_sp)} totales (α={alpha})")
        st.dataframe(df_sp.style
            .background_gradient(subset=["|ρ|"], cmap="Blues")
            .applymap(lambda v: "color:#c0392b;font-weight:bold" if v=="✔ Sí"
                      else ("color:#27ae60;font-weight:bold" if v=="✘ No" else ""),
                      subset=["Asociación"])
            .format({"ρ":"{:.5f}","|ρ|":"{:.5f}","p-valor":"{:.4f}"}),
            use_container_width=True, height=400)

        st.divider()
        st.subheader("Kruskal-Wallis — Numérica × Categórica")
        with st.spinner("Calculando Kruskal-Wallis..."):
            def _efecto(eps):
                if eps < 0.01: return "Despreciable"
                if eps < 0.08: return "Pequeño"
                if eps < 0.26: return "Moderado"
                return "Grande"
            n_obs = len(df)
            rows_kw = []
            for vn, vc in product(numericas, categoricas):
                grupos = [df[vn][df[vc]==cat].dropna() for cat in df[vc].dropna().unique()]
                grupos = [g for g in grupos if len(g) > 0]
                if len(grupos) < 2: continue
                k = len(grupos)
                h, p = kruskal(*grupos)
                eps = max(0, (h-k+1)/(n_obs-k))
                rows_kw.append({"Numérica": vn, "Categórica": vc,
                                 "H": round(h,4), "p-valor": round(p,4),
                                 "ε²": round(eps,4), "Efecto": _efecto(eps),
                                 "Diferencias": "✔ Sí" if p < alpha else "✘ No"})
            df_kw = pd.DataFrame(rows_kw).sort_values("p-valor").reset_index(drop=True)
        sig = (df_kw["Diferencias"]=="✔ Sí").sum()
        st.info(f"**{sig} pares con diferencias** de {len(df_kw)} totales (α={alpha})")
        st.dataframe(df_kw.style
            .background_gradient(subset=["ε²"], cmap="Oranges")
            .applymap(lambda v: "color:#c0392b;font-weight:bold" if v=="✔ Sí"
                      else ("color:#27ae60;font-weight:bold" if v=="✘ No" else ""),
                      subset=["Diferencias"])
            .format({"H":"{:.4f}","p-valor":"{:.4f}","ε²":"{:.4f}"}),
            use_container_width=True, height=400)
