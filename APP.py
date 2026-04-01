import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path


# ==========================================================
# CONFIG GENERAL
# ==========================================================
st.set_page_config(
    page_title="Outliers univariados fenología - IQR",
    layout="wide"
)

st.title("Detección univariada de outliers fenológicos con IQR")
st.caption(
    "Enfoque: variables de conteo | método univariado | reglas Q1 - 1.5IQR y Q3 + 1.5IQR"
)

# ==========================================================
# PARÁMETROS BASE
# ==========================================================
EXCEL_FILE = "(2025) W09.xlsx"
SHEET_DEFAULT = "CONSOLIDADO"

GROUP_COLS_DEFAULT = ["AÑO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD"]

COUNT_COLS_DEFAULT = [
    "FLORES",
    "FRUTO CUAJADO",
    "FRUTO VERDE",
    "FRUTO MADURO",
    "FRUTO ROSADO",
    "FRUTO CREMOSO",
]

OPTIONAL_NUMERIC_COLS = [
    "Ha TURNO",
    "DENSIDAD",
    "PESO BAYA MADURO (g)",
    "PESO BAYA CREMOSO (g)",
    "CALIBRE MADURO (mm)",
    "CALIBRE CREMOSO (mm)",
]

DATE_COLS_CANDIDATES = [
    "FECHA CONTEO GENERAL",
    "FECHA CONTEO PROYECCIÓN"
]


# ==========================================================
# FUNCIONES
# ==========================================================
def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\n", " ") for c in df.columns]
    return df


def convertir_tipos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in DATE_COLS_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    numeric_candidates = COUNT_COLS_DEFAULT + OPTIONAL_NUMERIC_COLS + ["SEMANA", "AÑO"]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    cat_candidates = ["CAMPAÑA", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD"]
    for col in cat_candidates:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    return df


def validar_columnas(df: pd.DataFrame, required_cols: list[str]) -> tuple[bool, list[str]]:
    faltantes = [c for c in required_cols if c not in df.columns]
    return len(faltantes) == 0, faltantes


def leer_excel_repo(file_path: str, sheet_name: str) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo '{file_path}' en la raíz del repositorio."
        )

    suffix = path.suffix.lower()

    if suffix in [".xlsx", ".xlsm", ".xltx", ".xltm"]:
        df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    elif suffix == ".xlsb":
        df = pd.read_excel(path, sheet_name=sheet_name, engine="pyxlsb")
    elif suffix == ".xls":
        df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        raise ValueError("Formato no soportado. Usa .xlsx, .xlsb o .xls")

    return df


def multiselect_con_todo(label: str, options: list, default_all: bool = True, key: str | None = None):
    opciones = sorted([x for x in options if pd.notna(x)])
    opciones_ui = ["Seleccionar todo"] + opciones
    default_val = opciones_ui if default_all else []

    seleccion = st.sidebar.multiselect(
        label,
        options=opciones_ui,
        default=default_val,
        key=key
    )

    if "Seleccionar todo" in seleccion or len(seleccion) == 0:
        return opciones

    return [x for x in seleccion if x != "Seleccionar todo"]


def aplicar_filtros_sidebar(df: pd.DataFrame):
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Nivel de análisis usado para IQR**")
    st.sidebar.code("AÑO + ETAPA + CAMPO + TURNO + VARIEDAD")

    st.sidebar.header("Filtros globales")

    df_f = df.copy()

    variables_seleccionadas = multiselect_con_todo(
        "VARIABLES DE CONTEO",
        [c for c in COUNT_COLS_DEFAULT if c in df_f.columns],
        default_all=True,
        key="variables_conteo_sidebar"
    )

    filtros_cols = ["AÑO", "CAMPAÑA", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD"]
    for col in filtros_cols:
        if col in df_f.columns:
            opciones = df_f[col].dropna().unique().tolist()
            seleccion = multiselect_con_todo(col, opciones, default_all=True, key=f"filtro_{col}")
            if len(seleccion) > 0:
                df_f = df_f[df_f[col].isin(seleccion)]

    if "SEMANA" in df_f.columns:
        semanas_validas = df_f["SEMANA"].dropna()
        if not semanas_validas.empty:
            smin = int(semanas_validas.min())
            smax = int(semanas_validas.max())
            rango = st.sidebar.slider(
                "Rango de SEMANA",
                min_value=smin,
                max_value=smax,
                value=(smin, smax)
            )
            df_f = df_f[df_f["SEMANA"].between(rango[0], rango[1])]

    return df_f, variables_seleccionadas


def calcular_metricas_concordancia(
    detalle: pd.DataFrame,
    value_col: str,
    min_group_size: int
) -> pd.DataFrame:
    detalle = detalle.copy()

    detalle["dist_fuera_limite"] = np.where(
        detalle["outlier_iqr"].eq(1),
        np.where(
            detalle["direccion_outlier"].eq("ALTO"),
            detalle[value_col] - detalle["lim_sup"],
            detalle["lim_inf"] - detalle[value_col]
        ),
        0.0
    )

    detalle["iqr_ajustado"] = detalle["iqr"].replace(0, np.nan)

    detalle["severidad_relativa_iqr"] = np.where(
        detalle["outlier_iqr"].eq(1),
        detalle["dist_fuera_limite"] / detalle["iqr_ajustado"],
        0.0
    )
    detalle["severidad_relativa_iqr"] = (
        detalle["severidad_relativa_iqr"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    detalle["score_severidad"] = np.clip(detalle["severidad_relativa_iqr"] / 2.0, 0, 1)
    detalle["score_n"] = np.clip(detalle["n"] / 20.0, 0, 1)

    detalle["metrica_concordancia"] = np.where(
        detalle["outlier_iqr"].eq(1),
        100 * (0.70 * detalle["score_severidad"] + 0.30 * detalle["score_n"]),
        0.0
    )

    detalle["metrica_concordancia"] = detalle["metrica_concordancia"].round(1)

    detalle["nivel_concordancia"] = np.select(
        [
            detalle["outlier_iqr"].eq(0),
            detalle["metrica_concordancia"] < 40,
            detalle["metrica_concordancia"].between(40, 69.9999, inclusive="left"),
            detalle["metrica_concordancia"] >= 70,
        ],
        [
            "NO APLICA",
            "BAJA",
            "MEDIA",
            "ALTA"
        ],
        default="NO APLICA"
    )

    return detalle


def calcular_iqr_por_grupo(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    min_group_size: int = 5,
    whisker: float = 1.5
) -> pd.DataFrame:
    work = df.copy()

    cols_needed = group_cols + [value_col]
    work = work[cols_needed].copy()
    work = work.dropna(subset=[value_col])

    if work.empty:
        return pd.DataFrame()

    grp = work.groupby(group_cols, dropna=False)[value_col]

    resumen = grp.agg(
        n="count",
        q1=lambda s: s.quantile(0.25),
        mediana="median",
        q3=lambda s: s.quantile(0.75),
        promedio="mean",
        minimo="min",
        maximo="max"
    ).reset_index()

    resumen["iqr"] = resumen["q3"] - resumen["q1"]
    resumen["lim_inf"] = resumen["q1"] - whisker * resumen["iqr"]
    resumen["lim_sup"] = resumen["q3"] + whisker * resumen["iqr"]

    resumen["lim_inf"] = np.where(resumen["iqr"].fillna(0) == 0, resumen["q1"], resumen["lim_inf"])
    resumen["lim_sup"] = np.where(resumen["iqr"].fillna(0) == 0, resumen["q3"], resumen["lim_sup"])

    detalle = df.copy().merge(resumen, on=group_cols, how="left")

    detalle["variable"] = value_col
    detalle["grupo_valido_iqr"] = detalle["n"].fillna(0) >= min_group_size

    detalle["outlier_iqr"] = np.where(
        (detalle["grupo_valido_iqr"]) &
        (detalle[value_col].notna()) &
        (
            (detalle[value_col] < detalle["lim_inf"]) |
            (detalle[value_col] > detalle["lim_sup"])
        ),
        1,
        0
    )

    detalle["direccion_outlier"] = np.select(
        [
            detalle["outlier_iqr"].eq(1) & (detalle[value_col] < detalle["lim_inf"]),
            detalle["outlier_iqr"].eq(1) & (detalle[value_col] > detalle["lim_sup"]),
        ],
        [
            "BAJO",
            "ALTO",
        ],
        default="NORMAL"
    )

    detalle["desviacion_sobre_limite"] = np.where(
        detalle["outlier_iqr"].eq(1),
        np.where(
            detalle["direccion_outlier"].eq("ALTO"),
            detalle[value_col] - detalle["lim_sup"],
            detalle["lim_inf"] - detalle[value_col]
        ),
        0.0
    )

    detalle["ratio_vs_lim_sup"] = np.where(
        detalle["lim_sup"].notna() & (detalle["lim_sup"] != 0),
        detalle[value_col] / detalle["lim_sup"],
        np.nan
    )

    detalle["ratio_vs_lim_inf"] = np.where(
        detalle["lim_inf"].notna() & (detalle["lim_inf"] != 0),
        detalle[value_col] / detalle["lim_inf"],
        np.nan
    )

    detalle = calcular_metricas_concordancia(
        detalle=detalle,
        value_col=value_col,
        min_group_size=min_group_size
    )

    return detalle


def consolidar_resultados_iqr(
    df: pd.DataFrame,
    group_cols: list[str],
    variables: list[str],
    min_group_size: int,
    whisker: float
) -> pd.DataFrame:
    resultados = []

    for var in variables:
        if var in df.columns:
            det = calcular_iqr_por_grupo(
                df=df,
                group_cols=group_cols,
                value_col=var,
                min_group_size=min_group_size,
                whisker=whisker
            )
            if not det.empty:
                resultados.append(det)

    if len(resultados) == 0:
        return pd.DataFrame()

    return pd.concat(resultados, ignore_index=True)


def resumen_por_variable(df_det: pd.DataFrame) -> pd.DataFrame:
    if df_det.empty:
        return pd.DataFrame()

    df_valid = df_det[df_det["valor_observado"].notna()].copy()

    if df_valid.empty:
        return pd.DataFrame()

    out = (
        df_valid.groupby("variable", dropna=False)
        .agg(
            registros=("variable", "size"),
            grupos_validos=("grupo_valido_iqr", "sum"),
            outliers=("outlier_iqr", "sum"),
            pct_outliers=("outlier_iqr", lambda s: 100 * s.mean()),
            concordancia_promedio=("metrica_concordancia", "mean"),
            concordancia_outliers_promedio=("metrica_concordancia", lambda s: s[s > 0].mean() if (s > 0).any() else 0),
            q1=("q1", "first"),
            mediana=("mediana", "first"),
            q3=("q3", "first"),
            iqr=("iqr", "first"),
            lim_inf=("lim_inf", "first"),
            lim_sup=("lim_sup", "first"),
        )
        .reset_index()
        .sort_values(["outliers", "pct_outliers"], ascending=[False, False])
    )

    out["concordancia_promedio"] = out["concordancia_promedio"].round(1)
    out["concordancia_outliers_promedio"] = out["concordancia_outliers_promedio"].round(1)
    out["pct_outliers"] = out["pct_outliers"].round(4)

    return out


def resumen_por_grupo(df_det: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df_det.empty:
        return pd.DataFrame()

    df_valid = df_det[df_det["valor_observado"].notna()].copy()

    if df_valid.empty:
        return pd.DataFrame()

    out = (
        df_valid.groupby(group_cols + ["variable"], dropna=False)
        .agg(
            n=("variable", "size"),
            outliers=("outlier_iqr", "sum"),
            pct_outliers=("outlier_iqr", lambda s: 100 * s.mean()),
            concordancia_media=("metrica_concordancia", "mean"),
            concordancia_max=("metrica_concordancia", "max"),
            q1=("q1", "first"),
            mediana=("mediana", "first"),
            q3=("q3", "first"),
            iqr=("iqr", "first"),
            lim_inf=("lim_inf", "first"),
            lim_sup=("lim_sup", "first"),
        )
        .reset_index()
        .sort_values(["outliers", "pct_outliers", "concordancia_max"], ascending=[False, False, False])
    )

    out["concordancia_media"] = out["concordancia_media"].round(1)
    out["concordancia_max"] = out["concordancia_max"].round(1)
    out["pct_outliers"] = out["pct_outliers"].round(4)

    return out


# ==========================================================
# SIDEBAR - CONFIG
# ==========================================================
st.sidebar.header("Configuración")

min_group_size = st.sidebar.number_input(
    "Mínimo N por grupo para aplicar IQR",
    min_value=3,
    max_value=30,
    value=5,
    step=1
)
whisker = st.sidebar.number_input(
    "Factor IQR",
    min_value=1.0,
    max_value=3.0,
    value=1.5,
    step=0.1
)

# ==========================================================
# LECTURA
# ==========================================================
try:
    df_raw = leer_excel_repo(EXCEL_FILE, sheet_name=SHEET_DEFAULT)
except Exception as e:
    st.error(f"No se pudo leer el archivo/hoja: {e}")
    st.stop()

df_raw = normalizar_columnas(df_raw)
df = convertir_tipos(df_raw)

# ==========================================================
# VALIDACIÓN DE COLUMNAS
# ==========================================================
required_cols = GROUP_COLS_DEFAULT + COUNT_COLS_DEFAULT + ["SEMANA"]
ok, faltantes = validar_columnas(df, required_cols)

if not ok:
    st.error(f"Faltan columnas obligatorias: {faltantes}")
    st.stop()

# ==========================================================
# FILTROS
# ==========================================================
df_f, variables_seleccionadas = aplicar_filtros_sidebar(df)

if df_f.empty:
    st.warning("Con los filtros actuales no hay datos.")
    st.stop()

if len(variables_seleccionadas) == 0:
    st.warning("Selecciona al menos una variable de conteo.")
    st.stop()

# ==========================================================
# DETECCIÓN IQR
# ==========================================================
group_cols = [c for c in GROUP_COLS_DEFAULT if c in df_f.columns]

df_det = consolidar_resultados_iqr(
    df=df_f,
    group_cols=group_cols,
    variables=variables_seleccionadas,
    min_group_size=min_group_size,
    whisker=whisker
)

if df_det.empty:
    st.warning("No se pudieron generar resultados IQR.")
    st.stop()

df_det["valor_observado"] = df_det.apply(
    lambda r: r[r["variable"]] if r["variable"] in df_det.columns else np.nan,
    axis=1
)

df_valid = df_det[df_det["valor_observado"].notna()].copy()
df_outliers = df_valid[df_valid["outlier_iqr"] == 1].copy()

# ==========================================================
# RESUMEN GENERAL
# ==========================================================
st.subheader("Resumen general")

res_var = resumen_por_variable(df_det)
res_grp = resumen_por_grupo(df_det, group_cols)

filas_originales = len(df_f)
columnas_total = df_f.shape[1]
variables_conteo_total = len(variables_seleccionadas)
registros_analizados = len(df_valid)
outliers_detectados = int(df_outliers["outlier_iqr"].sum())
pct_outliers = (100 * outliers_detectados / registros_analizados) if registros_analizados > 0 else 0
concordancia_media_outliers = df_outliers["metrica_concordancia"].mean() if not df_outliers.empty else 0

fila1 = st.columns(4)
with fila1[0]:
    st.metric("Filas originales", f"{filas_originales:,}")
with fila1[1]:
    st.metric("Columnas", f"{columnas_total:,}")
with fila1[2]:
    st.metric("Variables de conteo", f"{variables_conteo_total:,}")
with fila1[3]:
    st.metric("Registros analizados", f"{registros_analizados:,}")

fila2 = st.columns(3)
with fila2[0]:
    st.metric("Outliers detectados", f"{outliers_detectados:,}")
with fila2[1]:
    st.metric("% outliers", f"{pct_outliers:.2f}%")
with fila2[2]:
    st.metric("Concordancia media outliers", f"{concordancia_media_outliers:.1f}")

# ==========================================================
# RESUMEN POR VARIABLE
# ==========================================================
st.markdown("### Resumen por variable")

if not res_var.empty:
    res_var_display = res_var.rename(columns={"pct_outliers": "% outliers"})
    columnas_resumen_var = [
        c for c in [
            "variable",
            "registros",
            "grupos_validos",
            "outliers",
            "% outliers",
            "concordancia_promedio",
            "concordancia_outliers_promedio",
            "q1",
            "mediana",
            "q3",
            "iqr",
            "lim_inf",
            "lim_sup",
        ] if c in res_var_display.columns
    ]
    st.dataframe(res_var_display[columnas_resumen_var], use_container_width=True)
else:
    st.warning("No hay datos válidos para mostrar en Resumen por variable.")

# ==========================================================
# TOP GRUPOS CON MAYOR INCIDENCIA DE OUTLIERS
# ==========================================================
st.markdown("### Top grupos con mayor incidencia de outliers")
st.dataframe(res_grp.head(100), use_container_width=True)

# ==========================================================
# MÉTRICA DE CONCORDANCIA
# ==========================================================
st.subheader("Métrica de concordancia")

st.info(
    "La métrica de concordancia no es una probabilidad estadística real. "
    "Es un score heurístico de 0 a 100 que indica qué tan fuerte es la evidencia "
    "de que el valor detectado como outlier realmente sea atípico."
)

col_mc1, col_mc2 = st.columns([1, 2])

with col_mc1:
    tabla_concordancia = (
        df_outliers.groupby(["variable", "nivel_concordancia"], dropna=False)
        .size()
        .reset_index(name="casos")
        .sort_values(["variable", "casos"], ascending=[True, False])
    )
    st.markdown("#### Distribución por nivel")
    st.dataframe(tabla_concordancia, use_container_width=True)

with col_mc2:
    if not df_outliers.empty:
        fig_hist_conc = px.histogram(
            df_outliers,
            x="metrica_concordancia",
            color="nivel_concordancia",
            nbins=20,
            title="Distribución de la métrica de concordancia en outliers"
        )
        st.plotly_chart(fig_hist_conc, use_container_width=True)
    else:
        st.warning("No hay outliers para mostrar la distribución de concordancia.")

# ==========================================================
# VISUALIZACIONES
# ==========================================================
st.subheader("Diagnóstico visual")

variables_plot = [v for v in variables_seleccionadas if v in df_valid["variable"].dropna().unique().tolist()]

if len(variables_plot) == 0:
    st.warning("No hay variables con datos válidos para visualizar.")
else:
    var_plot = st.selectbox(
        "Variable para visualizar",
        options=variables_plot,
        index=0
    )

    plot_var = df_valid[df_valid["variable"] == var_plot].copy()

    fig_box = px.box(
        plot_var,
        y="valor_observado",
        points="all",
        color="outlier_iqr",
        hover_data=[
            c for c in [
                "AÑO", "SEMANA", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
                "metrica_concordancia"
            ] if c in plot_var.columns
        ],
        title=f"Boxplot - {var_plot}"
    )
    st.plotly_chart(fig_box, use_container_width=True)

    if "SEMANA" in plot_var.columns:
        hover_cols = [
            c for c in [
                "AÑO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
                "valor_observado", "lim_inf", "lim_sup",
                "metrica_concordancia", "nivel_concordancia"
            ] if c in plot_var.columns
        ]

        fig_scatter = px.scatter(
            plot_var.sort_values("SEMANA"),
            x="SEMANA",
            y="valor_observado",
            color="nivel_concordancia",
            hover_data=hover_cols,
            title=f"Dispersión semanal - {var_plot}"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================================
# INTERPRETACIÓN
# ==========================================================
st.subheader("Cómo interpreta esta app un outlier")

st.markdown(
    """
- Para cada variable de conteo, la app agrupa por: **AÑO + ETAPA + CAMPO + TURNO + VARIEDAD**.
- Dentro de cada grupo, toma la distribución histórica disponible en las semanas filtradas.
- Calcula:
  - **Q1**
  - **Q3**
  - **IQR = Q3 - Q1**
  - **Límite inferior = Q1 - 1.5 x IQR**
  - **Límite superior = Q3 + 1.5 x IQR**
- Si el valor observado cae fuera de esos límites, se marca como **outlier**.
- Si un grupo no tiene suficientes datos (`N` menor al mínimo configurado), la app no fuerza la decisión.
- **Registros analizados** considera solo valores válidos, excluyendo vacíos.
- **% outliers** se calcula como: **outliers / registros analizados válidos**.
"""
)

st.markdown("### Cómo se calcula la métrica de concordancia")
st.markdown(
    """
La **métrica de concordancia** va de **0 a 100** y solo aplica a registros detectados como outliers.

Se construye con dos componentes:

1. **Severidad del outlier**  
   Qué tan lejos está el valor fuera del límite IQR, relativo al tamaño del IQR del grupo.

2. **Confiabilidad del grupo**  
   Qué tan grande es el grupo (`n`). Un grupo con más observaciones da más confianza que uno muy pequeño.

Interpretación sugerida:
- **BAJA**: el valor salió de los límites, pero la evidencia no es tan fuerte.
- **MEDIA**: el valor parece atípico con evidencia razonable.
- **ALTA**: el valor está claramente fuera del comportamiento esperado y además el grupo da buena base para confiar en el hallazgo.

**Importante:** esta métrica no es una probabilidad real, sino una priorización técnica para revisión.
"""
)
