import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path


# ==========================================================
# CONFIG GENERAL
# ==========================================================
st.set_page_config(
    page_title="Outliers fenológicos | Univariado y Bivariado",
    layout="wide"
)

st.title("Detección de outliers fenológicos")
st.caption(
    "Estructura del análisis: univariado con IQR y bivariado con Mahalanobis + IQR"
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

VARIABLE_ORDER = [
    "FLORES",
    "FRUTO CUAJADO",
    "FRUTO VERDE",
    "FRUTO CREMOSO",
    "FRUTO ROSADO",
    "FRUTO MADURO",
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

BIO_COUNT_COLS = [
    "FLORES",
    "FRUTO CUAJADO",
    "FRUTO VERDE",
    "FRUTO CREMOSO",
    "FRUTO ROSADO",
    "FRUTO MADURO",
]

BIO_RELATIONS = [
    {
        "nombre": "CUAJO_t vs FLORES_t-1",
        "target": "FRUTO CUAJADO",
        "source_lag": "FLORES_LAG1",
        "x_label": "Flores t-1",
        "y_label": "Cuajo t",
    },
    {
        "nombre": "VERDE_t vs CUAJO_t-1",
        "target": "FRUTO VERDE",
        "source_lag": "FRUTO CUAJADO_LAG1",
        "x_label": "Cuajo t-1",
        "y_label": "Verde t",
    },
    {
        "nombre": "VERDE_t vs FLORES_t-2",
        "target": "FRUTO VERDE",
        "source_lag": "FLORES_LAG2",
        "x_label": "Flores t-2",
        "y_label": "Verde t",
    },
]

BIO_RELATIONS_RIPENING = [
    {
        "nombre": "AZUL_t vs CREMOSO_t-1",
        "target": "FRUTO MADURO",
        "source_lag": "FRUTO CREMOSO_LAG1",
        "x_label": "Cremoso t-1",
        "y_label": "Azul t",
    },
    {
        "nombre": "AZUL_t vs CREMOSO_t-2",
        "target": "FRUTO MADURO",
        "source_lag": "FRUTO CREMOSO_LAG2",
        "x_label": "Cremoso t-2",
        "y_label": "Azul t",
    },
    {
        "nombre": "ROSADO_t vs CREMOSO_t",
        "target": "FRUTO ROSADO",
        "source_lag": "FRUTO CREMOSO",
        "x_label": "Cremoso t",
        "y_label": "Rosado t",
    },
    {
        "nombre": "ROSADO_t vs CREMOSO_t-1",
        "target": "FRUTO ROSADO",
        "source_lag": "FRUTO CREMOSO_LAG1",
        "x_label": "Cremoso t-1",
        "y_label": "Rosado t",
    },
    {
        "nombre": "AZUL_t vs ROSADO_t",
        "target": "FRUTO MADURO",
        "source_lag": "FRUTO ROSADO",
        "x_label": "Rosado t",
        "y_label": "Azul t",
    },
    {
        "nombre": "AZUL_t vs ROSADO_t-1",
        "target": "FRUTO MADURO",
        "source_lag": "FRUTO ROSADO_LAG1",
        "x_label": "Rosado t-1",
        "y_label": "Azul t",
    },
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
    opciones = [x for x in options if pd.notna(x)]

    if label == "VARIABLES DE CONTEO":
        opciones = [v for v in VARIABLE_ORDER if v in opciones]
    else:
        opciones = sorted(opciones)

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


def multiselect_visual_con_todo(
    label: str,
    options: list,
    default_first: bool = True,
    key: str | None = None
):
    opciones = [x for x in options if pd.notna(x)]

    if len(opciones) == 0:
        return []

    opciones_ui = ["Seleccionar todo"] + opciones
    default_val = [opciones[0]] if default_first else []

    seleccion = st.multiselect(
        label,
        options=opciones_ui,
        default=default_val,
        key=key
    )

    if "Seleccionar todo" in seleccion:
        return opciones

    return [x for x in seleccion if x != "Seleccionar todo"]


def aplicar_filtros_sidebar(df: pd.DataFrame):
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Nivel de análisis base**")
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


def preparar_lags_biologicos(
    df: pd.DataFrame,
    group_cols: list[str],
    week_col: str = "SEMANA"
) -> pd.DataFrame:
    df = df.copy()

    if week_col not in df.columns:
        return df

    needed_cols = [c for c in BIO_COUNT_COLS if c in df.columns]
    if len(needed_cols) == 0:
        return df

    sort_cols = [c for c in group_cols if c in df.columns] + [week_col]
    df = df.sort_values(sort_cols).copy()

    grp = df.groupby(group_cols, dropna=False)

    if "FLORES" in df.columns:
        df["FLORES_LAG1"] = grp["FLORES"].shift(1)
        df["FLORES_LAG2"] = grp["FLORES"].shift(2)

    if "FRUTO CUAJADO" in df.columns:
        df["FRUTO CUAJADO_LAG1"] = grp["FRUTO CUAJADO"].shift(1)

    if "FRUTO VERDE" in df.columns:
        df["FRUTO VERDE_LAG1"] = grp["FRUTO VERDE"].shift(1)

    if "FRUTO CREMOSO" in df.columns:
        df["FRUTO CREMOSO_LAG1"] = grp["FRUTO CREMOSO"].shift(1)
        df["FRUTO CREMOSO_LAG2"] = grp["FRUTO CREMOSO"].shift(2)

    if "FRUTO ROSADO" in df.columns:
        df["FRUTO ROSADO_LAG1"] = grp["FRUTO ROSADO"].shift(1)

    if "FRUTO MADURO" in df.columns:
        df["FRUTO MADURO_LAG1"] = grp["FRUTO MADURO"].shift(1)

    return df


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

    out = pd.concat(resultados, ignore_index=True)
    out["variable"] = pd.Categorical(out["variable"], categories=VARIABLE_ORDER, ordered=True)
    return out


def _mahalanobis_distances_group(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2 or X.shape[0] == 0:
        return np.array([])

    media = np.nanmean(X, axis=0)
    Xc = X - media

    if X.shape[0] == 1:
        return np.array([0.0])

    cov = np.cov(Xc, rowvar=False)

    if np.ndim(cov) == 0:
        cov = np.array([[cov]])

    try:
        inv_cov = np.linalg.pinv(cov)
    except Exception:
        return np.full(X.shape[0], np.nan)

    d2 = np.einsum("ij,jk,ik->i", Xc, inv_cov, Xc)
    d2 = np.clip(d2, 0, None)
    return d2


def calcular_bivariado_mahalanobis_iqr(
    df: pd.DataFrame,
    group_cols: list[str],
    relations: list[dict],
    min_group_size: int = 5,
    whisker: float = 1.5
) -> pd.DataFrame:
    resultados = []

    for rel in relations:
        target = rel["target"]
        source_lag = rel["source_lag"]
        nombre = rel["nombre"]
        x_label = rel["x_label"]
        y_label = rel["y_label"]

        if target not in df.columns or source_lag not in df.columns:
            continue

        cols_base = list(dict.fromkeys(group_cols + ["SEMANA", source_lag, target]))
        temp = df[cols_base].copy()
        temp = temp.dropna(subset=[source_lag, target])

        if temp.empty:
            continue

        partes = []

        for _, sub in temp.groupby(group_cols, dropna=False):
            sub = sub.copy()

            n = len(sub)
            sub["n_biv_maha"] = n
            sub["grupo_valido_maha_biv"] = n >= max(min_group_size, 4)

            if not sub["grupo_valido_maha_biv"].iloc[0]:
                sub["distancia_mahalanobis2_biv"] = np.nan
                sub["distancia_mahalanobis_biv"] = np.nan
                partes.append(sub)
                continue

            X = sub[[source_lag, target]].to_numpy(dtype=float)
            d2 = _mahalanobis_distances_group(X)
            sub["distancia_mahalanobis2_biv"] = d2
            sub["distancia_mahalanobis_biv"] = np.sqrt(d2)
            partes.append(sub)

        if len(partes) == 0:
            continue

        temp_dist = pd.concat(partes, ignore_index=True)

        det = calcular_iqr_por_grupo(
            df=temp_dist,
            group_cols=group_cols,
            value_col="distancia_mahalanobis_biv",
            min_group_size=min_group_size,
            whisker=whisker
        )

        if det.empty:
            continue

        det["relacion"] = nombre
        det["target"] = target
        det["source_lag"] = source_lag
        det["x_label"] = x_label
        det["y_label"] = y_label
        det["valor_target"] = det[target] if target in det.columns else np.nan
        det["valor_source_lag"] = det[source_lag] if source_lag in det.columns else np.nan
        det["anomalia_bivariante"] = det["outlier_iqr"]
        det["flag_outlier_biv"] = np.where(det["anomalia_bivariante"].eq(1), "OUTLIER", "NORMAL")
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
        df_valid.groupby("variable", dropna=False, observed=False)
        .agg(
            registros=("variable", "size"),
            grupos_validos=("grupo_valido_iqr", "sum"),
            outliers=("outlier_iqr", "sum"),
            pct_outliers=("outlier_iqr", lambda s: 100 * s.mean()),
            q1=("q1", "first"),
            mediana=("mediana", "first"),
            q3=("q3", "first"),
            iqr=("iqr", "first"),
            lim_inf=("lim_inf", "first"),
            lim_sup=("lim_sup", "first"),
        )
        .reset_index()
    )

    out["variable"] = pd.Categorical(out["variable"], categories=VARIABLE_ORDER, ordered=True)
    out = out.sort_values("variable")
    out["pct_outliers"] = out["pct_outliers"].round(4)

    return out


def resumen_por_grupo(df_det: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df_det.empty:
        return pd.DataFrame()

    df_valid = df_det[df_det["valor_observado"].notna()].copy()

    if df_valid.empty:
        return pd.DataFrame()

    out = (
        df_valid.groupby(group_cols + ["variable"], dropna=False, observed=False)
        .agg(
            n=("variable", "size"),
            outliers=("outlier_iqr", "sum"),
            pct_outliers=("outlier_iqr", lambda s: 100 * s.mean()),
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

    out["variable"] = pd.Categorical(out["variable"], categories=VARIABLE_ORDER, ordered=True)
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
# PREPARACIÓN BASE CON LAGS BIOLÓGICOS
# ==========================================================
group_cols = [c for c in GROUP_COLS_DEFAULT if c in df_f.columns]
df_f = preparar_lags_biologicos(df_f, group_cols=group_cols, week_col="SEMANA")

# ==========================================================
# 1) ANÁLISIS UNIVARIADO | IQR SOBRE CADA VARIABLE
# ==========================================================
st.subheader("1) Análisis univariado")

df_det = consolidar_resultados_iqr(
    df=df_f,
    group_cols=group_cols,
    variables=variables_seleccionadas,
    min_group_size=min_group_size,
    whisker=whisker
)

if df_det.empty:
    st.warning("No se pudieron generar resultados univariados.")
    st.stop()

df_det["valor_observado"] = df_det.apply(
    lambda r: r[r["variable"]] if r["variable"] in df_det.columns else np.nan,
    axis=1
)

df_valid = df_det[df_det["valor_observado"].notna()].copy()
df_valid["flag_outlier_iqr"] = np.where(df_valid["outlier_iqr"].eq(1), "OUTLIER", "NORMAL")
df_outliers = df_valid[df_valid["outlier_iqr"] == 1].copy()

res_var = resumen_por_variable(df_det)
res_grp = resumen_por_grupo(df_det, group_cols)

filas_originales = len(df_f)
columnas_total = df_f.shape[1]
variables_conteo_total = len(variables_seleccionadas)
registros_analizados = len(df_valid)
outliers_detectados = int(df_outliers["outlier_iqr"].sum())
pct_outliers = (100 * outliers_detectados / registros_analizados) if registros_analizados > 0 else 0

fila1 = st.columns(4)
with fila1[0]:
    st.metric("Filas originales", f"{filas_originales:,}")
with fila1[1]:
    st.metric("Columnas", f"{columnas_total:,}")
with fila1[2]:
    st.metric("Variables analizadas", f"{variables_conteo_total:,}")
with fila1[3]:
    st.metric("Registros analizados", f"{registros_analizados:,}")

fila2 = st.columns(2)
with fila2[0]:
    st.metric("Outliers univariados", f"{outliers_detectados:,}")
with fila2[1]:
    st.metric("% outliers univariados", f"{pct_outliers:.2f}%")

st.markdown("### Resumen univariado por variable")

if not res_var.empty:
    res_var_display = res_var.rename(columns={"pct_outliers": "% outliers"})
    columnas_resumen_var = [
        c for c in [
            "variable",
            "registros",
            "grupos_validos",
            "outliers",
            "% outliers",
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
    st.warning("No hay datos válidos para mostrar en el resumen univariado.")

st.markdown("### Top grupos con mayor incidencia de outliers univariados")
st.dataframe(res_grp.head(100), use_container_width=True)

# ==========================================================
# DIAGNÓSTICO VISUAL
# Se mantiene SOLO la dispersión semanal con filtro
# ==========================================================
st.markdown("### Dispersión semanal por variable")

if "SEMANA" in df_valid.columns:
    variables_dispersion = [v for v in VARIABLE_ORDER if v in df_valid["variable"].astype(str).unique().tolist()]

    if len(variables_dispersion) > 0:
        variables_scatter_sel = multiselect_visual_con_todo(
            "Selecciona variable para dispersión semanal",
            options=variables_dispersion,
            default_first=True,
            key="dispersion_multiselect"
        )

        if len(variables_scatter_sel) == 0:
            st.warning("Selecciona al menos una variable para la dispersión semanal.")
        else:
            df_scatter_multi = df_valid[
                df_valid["variable"].astype(str).isin(variables_scatter_sel)
            ].copy()

            df_scatter_multi["variable"] = pd.Categorical(
                df_scatter_multi["variable"],
                categories=VARIABLE_ORDER,
                ordered=True
            )

            df_scatter_multi = df_scatter_multi.sort_values(["SEMANA", "variable"])

            hover_cols = [
                c for c in [
                    "AÑO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
                    "variable", "valor_observado", "lim_inf", "lim_sup"
                ] if c in df_scatter_multi.columns
            ]

            fig_scatter = px.scatter(
                df_scatter_multi,
                x="SEMANA",
                y="valor_observado",
                color="variable",
                symbol="flag_outlier_iqr",
                category_orders={"variable": VARIABLE_ORDER},
                hover_data=hover_cols,
                title="Dispersión semanal de variables seleccionadas"
            )

            fig_scatter.update_layout(
                xaxis_title="Semana",
                yaxis_title="Valor observado",
                legend_title="Variable / Outlier"
            )

            st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("No hay variables con datos válidos para la dispersión semanal.")
else:
    st.warning("No existe la columna SEMANA para construir la dispersión semanal.")

# ==========================================================
# 2) ANÁLISIS BIVARIADO | MAHALANOBIS + IQR
# ==========================================================
st.subheader("2) Análisis bivariado con Mahalanobis")

st.caption(
    "Método usado: primero se calcula la distancia de Mahalanobis para cada relación bivariada y luego se aplica IQR sobre esa distancia."
)

df_biv = calcular_bivariado_mahalanobis_iqr(
    df=df_f,
    group_cols=group_cols,
    relations=BIO_RELATIONS,
    min_group_size=min_group_size,
    whisker=whisker
)

if not df_biv.empty:
    df_biv_valid = df_biv[df_biv["distancia_mahalanobis_biv"].notna()].copy()
    df_biv_out = df_biv_valid[df_biv_valid["anomalia_bivariante"] == 1].copy()
else:
    df_biv_valid = pd.DataFrame()
    df_biv_out = pd.DataFrame()

if not df_biv_valid.empty:
    resumen_biv = (
        df_biv_valid.groupby("relacion", dropna=False)
        .agg(
            registros=("relacion", "size"),
            anomalias_bivariantes=("anomalia_bivariante", "sum"),
            pct_anomalias=("anomalia_bivariante", lambda s: 100 * s.mean()),
            q1_dist_maha=("q1", "first"),
            q3_dist_maha=("q3", "first"),
            iqr_dist_maha=("iqr", "first"),
            lim_sup_dist_maha=("lim_sup", "first"),
        )
        .reset_index()
        .sort_values("relacion")
    )
    resumen_biv["pct_anomalias"] = resumen_biv["pct_anomalias"].round(4)

    st.markdown("### Resumen bivariado")
    st.dataframe(resumen_biv, use_container_width=True)

    st.markdown("### Top anomalías bivariadas")
    cols_biv_show = [
        c for c in [
            "AÑO", "SEMANA", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
            "relacion", "valor_target", "valor_source_lag",
            "distancia_mahalanobis_biv", "distancia_mahalanobis2_biv",
            "lim_sup"
        ] if c in df_biv_out.columns
    ]
    st.dataframe(
        df_biv_out.sort_values(
            ["distancia_mahalanobis_biv"],
            ascending=[False]
        )[cols_biv_show].head(100),
        use_container_width=True
    )

    relaciones_disponibles = sorted(df_biv_valid["relacion"].dropna().unique().tolist())
    relaciones_sel = multiselect_visual_con_todo(
        "Selecciona relación bivariada para visualizar",
        options=relaciones_disponibles,
        default_first=True,
        key="bivariado_relaciones_multiselect"
    )

    if len(relaciones_sel) == 0:
        st.warning("Selecciona al menos una relación bivariada para visualizar.")
    else:
        df_biv_plot = df_biv_valid[df_biv_valid["relacion"].isin(relaciones_sel)].copy()
        df_biv_plot["relacion"] = pd.Categorical(
            df_biv_plot["relacion"],
            categories=relaciones_disponibles,
            ordered=True
        )
        df_biv_plot = df_biv_plot.sort_values(["relacion", "SEMANA"])

        fig_biv = px.scatter(
            df_biv_plot,
            x="valor_source_lag",
            y="valor_target",
            color="flag_outlier_biv",
            facet_col="relacion",
            facet_col_wrap=2,
            hover_data=[
                c for c in [
                    "AÑO", "SEMANA", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
                    "distancia_mahalanobis_biv", "lim_sup"
                ] if c in df_biv_plot.columns
            ],
            category_orders={"relacion": relaciones_disponibles},
            title="Relaciones bivariadas seleccionadas"
        )

        fig_biv.update_layout(
            xaxis_title="Valor base (lag)",
            yaxis_title="Valor actual"
        )

        fig_biv.for_each_annotation(
            lambda a: a.update(text=a.text.split("=")[-1])
        )

        st.plotly_chart(fig_biv, use_container_width=True)
else:
    st.warning("No hay datos suficientes para evaluar el bivariado con Mahalanobis.")

# ==========================================================
# 3) ANÁLISIS BIVARIADO | CREMOSO, ROSADO Y AZUL
# ==========================================================
st.subheader("3) Análisis bivariado con Mahalanobis | Cremoso, Rosado y Azul")

st.caption(
    "Método usado: primero se calcula la distancia de Mahalanobis para cada relación bivariada entre cremoso, rosado y azul, y luego se aplica IQR sobre esa distancia."
)

df_biv_rip = calcular_bivariado_mahalanobis_iqr(
    df=df_f,
    group_cols=group_cols,
    relations=BIO_RELATIONS_RIPENING,
    min_group_size=min_group_size,
    whisker=whisker
)

if not df_biv_rip.empty:
    df_biv_rip_valid = df_biv_rip[df_biv_rip["distancia_mahalanobis_biv"].notna()].copy()
    df_biv_rip_out = df_biv_rip_valid[df_biv_rip_valid["anomalia_bivariante"] == 1].copy()
else:
    df_biv_rip_valid = pd.DataFrame()
    df_biv_rip_out = pd.DataFrame()

if not df_biv_rip_valid.empty:
    resumen_biv_rip = (
        df_biv_rip_valid.groupby("relacion", dropna=False)
        .agg(
            registros=("relacion", "size"),
            anomalias_bivariantes=("anomalia_bivariante", "sum"),
            pct_anomalias=("anomalia_bivariante", lambda s: 100 * s.mean()),
            q1_dist_maha=("q1", "first"),
            q3_dist_maha=("q3", "first"),
            iqr_dist_maha=("iqr", "first"),
            lim_sup_dist_maha=("lim_sup", "first"),
        )
        .reset_index()
        .sort_values("relacion")
    )
    resumen_biv_rip["pct_anomalias"] = resumen_biv_rip["pct_anomalias"].round(4)

    st.markdown("### Resumen bivariado | Cremoso, Rosado y Azul")
    st.dataframe(resumen_biv_rip, use_container_width=True)

    st.markdown("### Top anomalías bivariadas | Cremoso, Rosado y Azul")
    cols_biv_rip_show = [
        c for c in [
            "AÑO", "SEMANA", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
            "relacion", "valor_target", "valor_source_lag",
            "distancia_mahalanobis_biv", "distancia_mahalanobis2_biv",
            "lim_sup"
        ] if c in df_biv_rip_out.columns
    ]
    st.dataframe(
        df_biv_rip_out.sort_values(
            ["distancia_mahalanobis_biv"],
            ascending=[False]
        )[cols_biv_rip_show].head(100),
        use_container_width=True
    )

    relaciones_rip_disponibles = sorted(df_biv_rip_valid["relacion"].dropna().unique().tolist())
    relaciones_rip_sel = multiselect_visual_con_todo(
        "Selecciona relación bivariada para visualizar | Cremoso, Rosado y Azul",
        options=relaciones_rip_disponibles,
        default_first=True,
        key="bivariado_rip_relaciones_multiselect"
    )

    if len(relaciones_rip_sel) == 0:
        st.warning("Selecciona al menos una relación bivariada para visualizar en Cremoso, Rosado y Azul.")
    else:
        df_biv_rip_plot = df_biv_rip_valid[df_biv_rip_valid["relacion"].isin(relaciones_rip_sel)].copy()
        df_biv_rip_plot["relacion"] = pd.Categorical(
            df_biv_rip_plot["relacion"],
            categories=relaciones_rip_disponibles,
            ordered=True
        )
        df_biv_rip_plot = df_biv_rip_plot.sort_values(["relacion", "SEMANA"])

        fig_biv_rip = px.scatter(
            df_biv_rip_plot,
            x="valor_source_lag",
            y="valor_target",
            color="flag_outlier_biv",
            facet_col="relacion",
            facet_col_wrap=2,
            hover_data=[
                c for c in [
                    "AÑO", "SEMANA", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
                    "distancia_mahalanobis_biv", "lim_sup"
                ] if c in df_biv_rip_plot.columns
            ],
            category_orders={"relacion": relaciones_rip_disponibles},
            title="Relaciones bivariadas seleccionadas | Cremoso, Rosado y Azul"
        )

        fig_biv_rip.update_layout(
            xaxis_title="Valor base (lag)",
            yaxis_title="Valor actual"
        )

        fig_biv_rip.for_each_annotation(
            lambda a: a.update(text=a.text.split("=")[-1])
        )

        st.plotly_chart(fig_biv_rip, use_container_width=True)
else:
    st.warning("No hay datos suficientes para evaluar el bivariado con Mahalanobis en cremoso, rosado y azul.")

# ==========================================================
# INTERPRETACIÓN
# ==========================================================
st.subheader("Cómo interpreta esta app un outlier")

st.markdown(
    """
### Univariado
- Se analiza **una variable a la vez**.
- El IQR se aplica directamente sobre los valores observados de esa variable.
- Si el valor cae fuera de los límites IQR del grupo, se marca como outlier.

### Bivariado con Mahalanobis
- Se analizan **dos variables al mismo tiempo**.
- Primero se calcula la **distancia de Mahalanobis** para cada fila.
- Luego esa distancia se trata como una variable única de rareza.
- Finalmente se aplica **IQR** sobre esa distancia para identificar anomalías bivariadas.
"""
)

st.markdown(
    """
### Resumen del enfoque
- **Univariado** = IQR sobre la variable original.
- **Bivariado** = Mahalanobis sobre dos variables + IQR sobre la distancia resultante.
"""
)
