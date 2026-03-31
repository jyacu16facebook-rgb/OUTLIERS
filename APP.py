import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ==========================================================
# CONFIG GENERAL
# ==========================================================
st.set_page_config(
    page_title="Outliers univariados fenología - IQR",
    layout="wide"
)

st.title("Detección univariada de outliers fenológicos con IQR")
st.caption(
    "Enfoque: variables de conteo | método univariado | reglas Q1 - 1.5*IQR y Q3 + 1.5*IQR"
)

# ==========================================================
# PARÁMETROS BASE
# ==========================================================
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


def leer_excel_subido(uploaded_file, sheet_name: str) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix in [".xlsx", ".xlsm", ".xltx", ".xltm"]:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine="openpyxl")
    elif suffix == ".xlsb":
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine="pyxlsb")
    elif suffix == ".xls":
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    else:
        raise ValueError("Formato no soportado. Sube un archivo .xlsx, .xlsb o .xls")

    return df


def aplicar_filtros_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros globales")

    df_f = df.copy()

    filtros_cols = ["AÑO", "CAMPAÑA", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD"]
    for col in filtros_cols:
        if col in df_f.columns:
            opciones = sorted([x for x in df_f[col].dropna().unique().tolist()])
            seleccion = st.sidebar.multiselect(f"{col}", options=opciones, default=opciones)
            if len(seleccion) > 0:
                df_f = df_f[df_f[col].isin(seleccion)]

    if "SEMANA" in df_f.columns:
        semanas_validas = df_f["SEMANA"].dropna()
        if not semanas_validas.empty:
            smin = int(semanas_validas.min())
            smax = int(semanas_validas.max())
            rango = st.sidebar.slider("Rango de SEMANA", min_value=smin, max_value=smax, value=(smin, smax))
            df_f = df_f[df_f["SEMANA"].between(rango[0], rango[1])]

    return df_f


def calcular_metricas_concordancia(
    detalle: pd.DataFrame,
    value_col: str,
    min_group_size: int
) -> pd.DataFrame:
    """
    Construye una métrica heurística de concordancia (0-100) para outliers IQR.
    NO es probabilidad estadística real.
    Se basa en:
    1) severidad: qué tan lejos está del límite respecto al IQR
    2) confiabilidad del grupo: tamaño n
    """
    detalle = detalle.copy()

    # Distancia fuera del límite
    detalle["dist_fuera_limite"] = np.where(
        detalle["outlier_iqr"].eq(1),
        np.where(
            detalle["direccion_outlier"].eq("ALTO"),
            detalle[value_col] - detalle["lim_sup"],
            detalle["lim_inf"] - detalle[value_col]
        ),
        0.0
    )

    # Evitar división por cero
    detalle["iqr_ajustado"] = detalle["iqr"].replace(0, np.nan)

    # Severidad relativa al IQR
    detalle["severidad_relativa_iqr"] = np.where(
        detalle["outlier_iqr"].eq(1),
        detalle["dist_fuera_limite"] / detalle["iqr_ajustado"],
        0.0
    )
    detalle["severidad_relativa_iqr"] = detalle["severidad_relativa_iqr"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Score de severidad [0,1]
    # >= 2 IQR fuera del límite ya se considera severidad máxima del score
    detalle["score_severidad"] = np.clip(detalle["severidad_relativa_iqr"] / 2.0, 0, 1)

    # Score de tamaño de grupo [0,1]
    # Se satura en n=20; puedes cambiarlo si deseas
    detalle["score_n"] = np.clip(detalle["n"] / 20.0, 0, 1)

    # Score total 0-100
    # Mayor peso a la severidad
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

    out = (
        df_det.groupby("variable", dropna=False)
        .agg(
            registros=("variable", "size"),
            grupos_validos=("grupo_valido_iqr", "sum"),
            outliers=("outlier_iqr", "sum"),
            pct_outliers=("outlier_iqr", lambda s: 100 * s.mean()),
            concordancia_promedio=("metrica_concordancia", "mean"),
            concordancia_outliers_promedio=("metrica_concordancia", lambda s: s[s > 0].mean() if (s > 0).any() else 0),
        )
        .reset_index()
        .sort_values(["outliers", "pct_outliers"], ascending=[False, False])
    )

    out["concordancia_promedio"] = out["concordancia_promedio"].round(1)
    out["concordancia_outliers_promedio"] = out["concordancia_outliers_promedio"].round(1)
    return out


def resumen_por_grupo(df_det: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df_det.empty:
        return pd.DataFrame()

    out = (
        df_det.groupby(group_cols + ["variable"], dropna=False)
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
    return out


def preparar_exportables(df_det: pd.DataFrame, group_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_det.empty:
        return pd.DataFrame(), pd.DataFrame()

    detalle_cols = [
        c for c in [
            "AÑO", "CAMPAÑA", "SEMANA", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
            "FECHA CONTEO GENERAL", "FECHA CONTEO PROYECCIÓN",
            "variable", "valor_observado", "outlier_iqr", "direccion_outlier",
            "n", "q1", "mediana", "q3", "iqr", "lim_inf", "lim_sup",
            "desviacion_sobre_limite",
            "severidad_relativa_iqr",
            "metrica_concordancia",
            "nivel_concordancia"
        ] if c in df_det.columns
    ]

    detalle_export = df_det[detalle_cols].copy()
    outliers_export = detalle_export[detalle_export["outlier_iqr"] == 1].copy()

    return detalle_export, outliers_export


def to_excel_bytes(sheets_dict: dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in sheets_dict.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    output.seek(0)
    return output.getvalue()


# ==========================================================
# SIDEBAR - CONFIG
# ==========================================================
st.sidebar.header("Configuración")

uploaded_file = st.sidebar.file_uploader(
    "Sube tu archivo Excel",
    type=["xlsx", "xlsb", "xls"]
)

sheet_name = st.sidebar.text_input("Hoja a leer", value=SHEET_DEFAULT)
min_group_size = st.sidebar.number_input("Mínimo N por grupo para aplicar IQR", min_value=3, max_value=30, value=5, step=1)
whisker = st.sidebar.number_input("Factor IQR", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Nivel de análisis usado para IQR**")
st.sidebar.code("AÑO + ETAPA + CAMPO + TURNO + VARIEDAD")

# ==========================================================
# LECTURA
# ==========================================================
if uploaded_file is None:
    st.info(
        "Sube tu archivo Excel para comenzar. "
        "Tu archivo puede ser .xlsx, .xlsb o .xls, y se leerá la hoja CONSOLIDADO por defecto."
    )
    st.stop()

try:
    df_raw = leer_excel_subido(uploaded_file, sheet_name=sheet_name)
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
# INFO GENERAL
# ==========================================================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Filas originales", f"{len(df):,}")
with col2:
    st.metric("Columnas", f"{df.shape[1]:,}")
with col3:
    st.metric("Variables de conteo", f"{len(COUNT_COLS_DEFAULT):,}")

with st.expander("Vista previa de la data"):
    st.dataframe(df.head(20), use_container_width=True)

# ==========================================================
# FILTROS
# ==========================================================
df_f = aplicar_filtros_sidebar(df)

st.subheader("Data filtrada")
c1, c2 = st.columns(2)
with c1:
    st.metric("Filas después de filtros", f"{len(df_f):,}")
with c2:
    st.metric("Semanas únicas", f"{df_f['SEMANA'].nunique():,}" if "SEMANA" in df_f.columns else "N/A")

if df_f.empty:
    st.warning("Con los filtros actuales no hay datos.")
    st.stop()

# ==========================================================
# VARIABLES A ANALIZAR
# ==========================================================
st.subheader("Selección de variables de conteo")

variables_seleccionadas = st.multiselect(
    "Variables para analizar con IQR",
    options=[c for c in COUNT_COLS_DEFAULT if c in df_f.columns],
    default=[c for c in COUNT_COLS_DEFAULT if c in df_f.columns]
)

if len(variables_seleccionadas) == 0:
    st.warning("Selecciona al menos una variable.")
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

# ==========================================================
# RESÚMENES
# ==========================================================
st.subheader("Resumen general")

res_var = resumen_por_variable(df_det)
res_grp = resumen_por_grupo(df_det, group_cols)
df_outliers = df_det[df_det["outlier_iqr"] == 1].copy()

a, b, c, d = st.columns(4)
with a:
    st.metric("Registros analizados", f"{len(df_det):,}")
with b:
    st.metric("Outliers detectados", f"{int(df_det['outlier_iqr'].sum()):,}")
with c:
    pct = 100 * df_det["outlier_iqr"].mean() if len(df_det) > 0 else 0
    st.metric("% outliers", f"{pct:.2f}%")
with d:
    conc_media = df_outliers["metrica_concordancia"].mean() if not df_outliers.empty else 0
    st.metric("Concordancia media outliers", f"{conc_media:.1f}")

st.markdown("### Resumen por variable")
st.dataframe(res_var, use_container_width=True)

st.markdown("### Top grupos con mayor incidencia de outliers")
st.dataframe(res_grp.head(100), use_container_width=True)

st.markdown("### Detalle de outliers detectados")
detalle_outliers_cols = [
    c for c in [
        "AÑO", "CAMPAÑA", "SEMANA", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
        "variable", "valor_observado", "direccion_outlier",
        "q1", "mediana", "q3", "iqr", "lim_inf", "lim_sup",
        "desviacion_sobre_limite",
        "severidad_relativa_iqr",
        "metrica_concordancia",
        "nivel_concordancia"
    ] if c in df_outliers.columns
]
st.dataframe(
    df_outliers[detalle_outliers_cols].sort_values(
        ["metrica_concordancia", "desviacion_sobre_limite"], ascending=[False, False]
    ),
    use_container_width=True
)

# ==========================================================
# VISTA ADICIONAL: METRICA DE CONCORDANCIA
# ==========================================================
st.subheader("Métrica de concordancia")

st.info(
    "La métrica de concordancia NO es una probabilidad estadística real. "
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

st.markdown("#### Top casos con mayor concordancia")
cols_conc = [
    c for c in [
        "AÑO", "CAMPAÑA", "SEMANA", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
        "variable", "valor_observado", "direccion_outlier",
        "n", "iqr", "lim_inf", "lim_sup",
        "desviacion_sobre_limite", "severidad_relativa_iqr",
        "metrica_concordancia", "nivel_concordancia"
    ] if c in df_outliers.columns
]
st.dataframe(
    df_outliers[cols_conc].sort_values(
        ["metrica_concordancia", "desviacion_sobre_limite"], ascending=[False, False]
    ).head(100),
    use_container_width=True
)

# ==========================================================
# VISUALIZACIONES
# ==========================================================
st.subheader("Diagnóstico visual")

var_plot = st.selectbox("Variable para visualizar", options=variables_seleccionadas)

fig_box = px.box(
    df_det[df_det["variable"] == var_plot],
    y="valor_observado",
    points="all",
    color="outlier_iqr",
    hover_data=[c for c in ["AÑO", "SEMANA", "ETAPA", "CAMPO", "TURNO", "VARIEDAD", "metrica_concordancia"] if c in df_det.columns],
    title=f"Boxplot - {var_plot}"
)
st.plotly_chart(fig_box, use_container_width=True)

if "SEMANA" in df_det.columns:
    plot_df = df_det[df_det["variable"] == var_plot].copy()

    hover_cols = [
        c for c in [
            "AÑO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
            "valor_observado", "lim_inf", "lim_sup", "metrica_concordancia", "nivel_concordancia"
        ] if c in plot_df.columns
    ]

    fig_scatter = px.scatter(
        plot_df.sort_values("SEMANA"),
        x="SEMANA",
        y="valor_observado",
        color="nivel_concordancia",
        hover_data=hover_cols,
        title=f"Dispersión semanal - {var_plot}"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown(f"### Tabla diagnóstica - {var_plot}")
tabla_diag = df_det[df_det["variable"] == var_plot].copy()
tabla_diag_cols = [
    c for c in [
        "AÑO", "CAMPAÑA", "SEMANA", "FUNDO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
        "valor_observado", "outlier_iqr", "direccion_outlier",
        "n", "q1", "mediana", "q3", "iqr", "lim_inf", "lim_sup",
        "desviacion_sobre_limite", "severidad_relativa_iqr",
        "metrica_concordancia", "nivel_concordancia"
    ] if c in tabla_diag.columns
]
st.dataframe(
    tabla_diag[tabla_diag_cols].sort_values(
        ["outlier_iqr", "metrica_concordancia", "SEMANA"] if "SEMANA" in tabla_diag.columns else ["outlier_iqr", "metrica_concordancia"],
        ascending=[False, False, True] if "SEMANA" in tabla_diag.columns else [False, False]
    ),
    use_container_width=True
)

# ==========================================================
# DESCARGAS
# ==========================================================
st.subheader("Descargas")

detalle_export, outliers_export = preparar_exportables(df_det, group_cols)

excel_bytes = to_excel_bytes({
    "detalle_iqr": detalle_export,
    "outliers_iqr": outliers_export,
    "resumen_variable": res_var,
    "resumen_grupo": res_grp
})

st.download_button(
    label="Descargar resultados en Excel",
    data=excel_bytes,
    file_name="outliers_univariados_iqr_fenologia.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

csv_bytes = outliers_export.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="Descargar solo outliers en CSV",
    data=csv_bytes,
    file_name="outliers_univariados_iqr_fenologia.csv",
    mime="text/csv"
)

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

**Importante:** esta métrica **no es una probabilidad real**, sino una priorización técnica para revisión.
"""
)

st.success("App lista.")