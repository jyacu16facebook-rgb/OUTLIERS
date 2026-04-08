import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from scipy.stats import chi2


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
]

BIO_RELATIONS = [
    {
        "nombre": "CUAJO_t vs FLORES_t-1",
        "target": "FRUTO CUAJADO",
        "source_lag": "FLORES_LAG1",
    },
    {
        "nombre": "VERDE_t vs CUAJO_t-1",
        "target": "FRUTO VERDE",
        "source_lag": "FRUTO CUAJADO_LAG1",
    },
    {
        "nombre": "VERDE_t vs FLORES_t-2",
        "target": "FRUTO VERDE",
        "source_lag": "FLORES_LAG2",
    },
]

# ==========================================================
# AJUSTE SOLICITADO:
# Mahalanobis solo con lógica biológica:
# CUAJO_t vs FLORES_t-1
# ==========================================================
MAHALANOBIS_FEATURES = [
    "FLORES_LAG1",
    "FRUTO CUAJADO",
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


def calcular_relaciones_bivariadas(
    df: pd.DataFrame,
    group_cols: list[str],
    min_group_size: int = 5,
    whisker: float = 1.5
) -> pd.DataFrame:
    resultados = []

    for rel in BIO_RELATIONS:
        target = rel["target"]
        source_lag = rel["source_lag"]
        nombre = rel["nombre"]

        if target not in df.columns or source_lag not in df.columns:
            continue

        temp = df.copy()
        ratio_col = f"RATIO__{target}__VS__{source_lag}"

        temp[ratio_col] = np.where(
            temp[source_lag].notna() & (temp[source_lag] > 0),
            temp[target] / temp[source_lag],
            np.nan
        )

        det = calcular_iqr_por_grupo(
            df=temp,
            group_cols=group_cols,
            value_col=ratio_col,
            min_group_size=min_group_size,
            whisker=whisker
        )

        if det.empty:
            continue

        det["relacion"] = nombre
        det["target"] = target
        det["source_lag"] = source_lag
        det["valor_target"] = det[target] if target in det.columns else np.nan
        det["valor_source_lag"] = det[source_lag] if source_lag in det.columns else np.nan
        det["ratio_relacion"] = det[ratio_col]
        det["anomalia_bivariante"] = det["outlier_iqr"]
        det["flag_outlier_biv"] = np.where(det["anomalia_bivariante"].eq(1), "OUTLIER", "NORMAL")
        resultados.append(det)

    if len(resultados) == 0:
        return pd.DataFrame()

    return pd.concat(resultados, ignore_index=True)


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
    return d2


def calcular_mahalanobis_biologico(
    df: pd.DataFrame,
    group_cols: list[str],
    min_group_size: int = 5,
    alpha: float = 0.975
) -> pd.DataFrame:
    needed = [c for c in MAHALANOBIS_FEATURES if c in df.columns]
    if len(needed) < 2:
        return pd.DataFrame()

    base_cols = list(dict.fromkeys(group_cols + ["SEMANA"] + needed))
    work = df[base_cols].copy()
    work = work.dropna(subset=needed)

    if work.empty:
        return pd.DataFrame()

    resultados = []

    for keys, sub in work.groupby(group_cols, dropna=False):
        sub = sub.copy()

        n = len(sub)
        p = len(needed)

        sub["n_maha"] = n
        sub["p_maha"] = p
        sub["grupo_valido_maha"] = n >= max(min_group_size, p + 2)

        if not sub["grupo_valido_maha"].iloc[0]:
            sub["distancia_mahalanobis2"] = np.nan
            sub["distancia_mahalanobis"] = np.nan
            sub["umbral_chi2"] = np.nan
            sub["anomalia_mahalanobis"] = 0
            sub["p_value_maha"] = np.nan
            sub["flag_outlier_maha"] = "NORMAL"
            resultados.append(sub)
            continue

        X = sub[needed].to_numpy(dtype=float)
        d2 = _mahalanobis_distances_group(X)

        umbral = chi2.ppf(alpha, df=p)
        pvals = 1 - chi2.cdf(d2, df=p)

        sub["distancia_mahalanobis2"] = d2
        sub["distancia_mahalanobis"] = np.sqrt(np.clip(d2, 0, None))
        sub["umbral_chi2"] = umbral
        sub["p_value_maha"] = pvals
        sub["anomalia_mahalanobis"] = np.where(d2 > umbral, 1, 0)
        sub["flag_outlier_maha"] = np.where(sub["anomalia_mahalanobis"].eq(1), "OUTLIER", "NORMAL")
        resultados.append(sub)

    if len(resultados) == 0:
        return pd.DataFrame()

    out = pd.concat(resultados, ignore_index=True)

    out["ratio_umbral_maha"] = np.where(
        out["umbral_chi2"].notna() & (out["umbral_chi2"] > 0),
        out["distancia_mahalanobis2"] / out["umbral_chi2"],
        np.nan
    )

    out["score_severidad_maha"] = np.clip((out["ratio_umbral_maha"] - 1) / 2.0, 0, 1)
    out["score_n_maha"] = np.clip(out["n_maha"] / 20.0, 0, 1)

    out["metrica_concordancia_maha"] = np.where(
        out["anomalia_mahalanobis"].eq(1),
        100 * (0.70 * out["score_severidad_maha"] + 0.30 * out["score_n_maha"]),
        0.0
    ).round(1)

    out["nivel_concordancia_maha"] = np.select(
        [
            out["anomalia_mahalanobis"].eq(0),
            out["metrica_concordancia_maha"] < 40,
            out["metrica_concordancia_maha"].between(40, 69.9999, inclusive="left"),
            out["metrica_concordancia_maha"] >= 70,
        ],
        [
            "NO APLICA",
            "BAJA",
            "MEDIA",
            "ALTA",
        ],
        default="NO APLICA"
    )

    return out


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
    )

    out["variable"] = pd.Categorical(out["variable"], categories=VARIABLE_ORDER, ordered=True)
    out = out.sort_values("variable")

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
        df_valid.groupby(group_cols + ["variable"], dropna=False, observed=False)
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

    out["variable"] = pd.Categorical(out["variable"], categories=VARIABLE_ORDER, ordered=True)

    out["concordancia_media"] = out["concordancia_media"].round(1)
    out["concordancia_max"] = out["concordancia_max"].round(1)
    out["pct_outliers"] = out["pct_outliers"].round(4)

    return out


def crear_boxplot_clasico_con_outliers_rojos(df_plot: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    variables = [v for v in VARIABLE_ORDER if v in df_plot["variable"].astype(str).unique().tolist()]

    for var in variables:
        sub = df_plot[df_plot["variable"].astype(str) == var].copy()

        fig.add_trace(
            go.Box(
                y=sub["valor_observado"],
                name=var,
                boxpoints=False,
                marker_color="rgba(70, 130, 180, 0.55)",
                line=dict(color="rgba(70, 130, 180, 1)"),
                fillcolor="rgba(70, 130, 180, 0.35)",
                hovertemplate=(
                    f"Variable: {var}<br>"
                    "Valor: %{y}<extra></extra>"
                )
            )
        )

        sub_out = sub[sub["outlier_iqr"] == 1].copy()
        if not sub_out.empty:
            fig.add_trace(
                go.Scatter(
                    x=[var] * len(sub_out),
                    y=sub_out["valor_observado"],
                    mode="markers",
                    name=f"Outliers - {var}",
                    marker=dict(
                        color="red",
                        size=7,
                        opacity=0.85,
                        line=dict(color="darkred", width=0.5)
                    ),
                    hovertemplate=(
                        f"Variable: {var}<br>"
                        "Valor: %{y}<extra></extra>"
                    )
                )
            )

    fig.update_layout(
        title="Boxplot clásico con outliers en rojo",
        xaxis_title="Variable",
        yaxis_title="Valor observado",
        showlegend=False
    )

    fig.update_xaxes(categoryorder="array", categoryarray=VARIABLE_ORDER)

    return fig


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

alpha_maha = st.sidebar.number_input(
    "Nivel chi-cuadrado Mahalanobis",
    min_value=0.90,
    max_value=0.999,
    value=0.975,
    step=0.005,
    format="%.3f"
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
# DETECCIÓN IQR UNIVARIANTE
# ==========================================================
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
df_valid["flag_outlier_iqr"] = np.where(df_valid["outlier_iqr"].eq(1), "OUTLIER", "NORMAL")
df_outliers = df_valid[df_valid["outlier_iqr"] == 1].copy()

# ==========================================================
# DETECCIÓN BIVARIADA SIMPLE
# ==========================================================
df_biv = calcular_relaciones_bivariadas(
    df=df_f,
    group_cols=group_cols,
    min_group_size=min_group_size,
    whisker=whisker
)

if not df_biv.empty:
    df_biv_valid = df_biv[df_biv["ratio_relacion"].notna()].copy()
    df_biv_out = df_biv_valid[df_biv_valid["anomalia_bivariante"] == 1].copy()
else:
    df_biv_valid = pd.DataFrame()
    df_biv_out = pd.DataFrame()

# ==========================================================
# DETECCIÓN MAHALANOBIS
# ==========================================================
df_maha = calcular_mahalanobis_biologico(
    df=df_f,
    group_cols=group_cols,
    min_group_size=min_group_size,
    alpha=alpha_maha
)

if not df_maha.empty:
    df_maha_valid = df_maha[df_maha["distancia_mahalanobis2"].notna()].copy()
    df_maha_out = df_maha_valid[df_maha_valid["anomalia_mahalanobis"] == 1].copy()
else:
    df_maha_valid = pd.DataFrame()
    df_maha_out = pd.DataFrame()

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
        df_outliers.groupby(["variable", "nivel_concordancia"], dropna=False, observed=False)
        .size()
        .reset_index(name="casos")
    )
    tabla_concordancia["variable"] = pd.Categorical(
        tabla_concordancia["variable"],
        categories=VARIABLE_ORDER,
        ordered=True
    )
    tabla_concordancia = tabla_concordancia.sort_values(["variable", "casos"], ascending=[True, False])

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
        fig_hist_conc.update_layout(
            xaxis_title="Concordancia",
            yaxis_title="Frecuencia"
        )
        st.plotly_chart(fig_hist_conc, use_container_width=True)
    else:
        st.warning("No hay outliers para mostrar la distribución de concordancia.")

# ==========================================================
# NUEVA VISTA AGREGADA: BOXPLOT CLÁSICO
# ==========================================================
st.markdown("#### Boxplot clásico con outliers en rojo")

if not df_valid.empty:
    fig_box_clasico = crear_boxplot_clasico_con_outliers_rojos(df_valid)
    st.plotly_chart(fig_box_clasico, use_container_width=True)
else:
    st.warning("No hay datos válidos para construir el boxplot clásico.")

# ==========================================================
# VISUALIZACIONES
# ==========================================================
st.subheader("Diagnóstico visual")

variables_plot = [v for v in VARIABLE_ORDER if v in df_valid["variable"].astype(str).dropna().unique().tolist()]

if len(variables_plot) == 0:
    st.warning("No hay variables con datos válidos para visualizar.")
else:
    fig_box = px.box(
        df_valid,
        x="variable",
        y="valor_observado",
        points="all",
        color="flag_outlier_iqr",
        category_orders={"variable": VARIABLE_ORDER},
        hover_data=[
            c for c in [
                "AÑO", "SEMANA", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
                "metrica_concordancia", "nivel_concordancia"
            ] if c in df_valid.columns
        ],
        title="Boxplot"
    )
    fig_box.update_layout(
        xaxis_title="Variable",
        yaxis_title="Valor observado"
    )
    st.plotly_chart(fig_box, use_container_width=True)

    if "SEMANA" in df_valid.columns:
        variable_scatter_sel = st.selectbox(
            "Selecciona variable para dispersión semanal",
            options=[v for v in VARIABLE_ORDER if v in df_valid["variable"].astype(str).unique().tolist()],
            index=0
        )

        df_scatter_var = df_valid[df_valid["variable"].astype(str) == variable_scatter_sel].copy()

        hover_cols = [
            c for c in [
                "AÑO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
                "valor_observado", "lim_inf", "lim_sup",
                "metrica_concordancia", "nivel_concordancia"
            ] if c in df_scatter_var.columns
        ]

        fig_scatter = px.scatter(
            df_scatter_var.sort_values("SEMANA"),
            x="SEMANA",
            y="valor_observado",
            color="flag_outlier_iqr",
            hover_data=hover_cols,
            title=f"Dispersión semanal - {variable_scatter_sel}"
        )
        fig_scatter.update_layout(
            xaxis_title="Semana",
            yaxis_title="Valor observado"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================================
# MÓDULO BIVARIADO ESENCIAL
# ==========================================================
st.subheader("Relaciones biológicas esenciales")

st.caption(
    "Relaciones puntuales evaluadas con ratio e IQR: "
    "CUAJO_t vs FLORES_t-1, VERDE_t vs CUAJO_t-1 y VERDE_t vs FLORES_t-2."
)

if not df_biv_valid.empty:
    resumen_biv = (
        df_biv_valid.groupby("relacion", dropna=False)
        .agg(
            registros=("relacion", "size"),
            anomalias_bivariantes=("anomalia_bivariante", "sum"),
            pct_anomalias=("anomalia_bivariante", lambda s: 100 * s.mean()),
            concordancia_media=("metrica_concordancia", lambda s: s[s > 0].mean() if (s > 0).any() else 0)
        )
        .reset_index()
        .sort_values("relacion")
    )
    resumen_biv["pct_anomalias"] = resumen_biv["pct_anomalias"].round(4)
    resumen_biv["concordancia_media"] = resumen_biv["concordancia_media"].round(1)

    st.markdown("### Resumen bivariante")
    st.dataframe(resumen_biv, use_container_width=True)

    st.markdown("### Top anomalías bivariantes")
    cols_biv_show = [
        c for c in [
            "AÑO", "SEMANA", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
            "relacion", "valor_target", "valor_source_lag", "ratio_relacion",
            "lim_inf", "lim_sup", "metrica_concordancia", "nivel_concordancia"
        ] if c in df_biv_out.columns
    ]
    st.dataframe(
        df_biv_out.sort_values(["metrica_concordancia", "ratio_relacion"], ascending=[False, False])[cols_biv_show].head(100),
        use_container_width=True
    )

    relaciones_disponibles = sorted(df_biv_valid["relacion"].dropna().unique().tolist())
    relacion_sel = st.selectbox(
        "Selecciona relación biológica para visualizar",
        options=relaciones_disponibles,
        index=0
    )

    df_biv_plot = df_biv_valid[df_biv_valid["relacion"] == relacion_sel].copy()

    fig_biv = px.scatter(
        df_biv_plot,
        x="valor_source_lag",
        y="valor_target",
        color="flag_outlier_biv",
        hover_data=[
            c for c in [
                "AÑO", "SEMANA", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
                "ratio_relacion", "metrica_concordancia", "nivel_concordancia"
            ] if c in df_biv_plot.columns
        ],
        title=f"Relación biológica: {relacion_sel}"
    )
    fig_biv.update_layout(
        xaxis_title="Valor base (lag)",
        yaxis_title="Valor actual"
    )
    st.plotly_chart(fig_biv, use_container_width=True)
else:
    st.warning("No hay datos suficientes para evaluar relaciones biológicas con lag.")

# ==========================================================
# MÓDULO MAHALANOBIS ESENCIAL
# AJUSTADO SOLO A: CUAJO_t vs FLORES_t-1
# ==========================================================
st.subheader("Anomalías multivariables esenciales (Mahalanobis)")

st.caption(
    "Vista adicional de coherencia biológica multivariable usando "
    "FLORES_t-1 vs CUAJO_t dentro del mismo grupo."
)

if not df_maha_valid.empty:
    resumen_maha = pd.DataFrame({
        "modelo": ["FLORES_t-1 vs CUAJO_t"],
        "registros": [len(df_maha_valid)],
        "anomalias_mahalanobis": [int(df_maha_valid["anomalia_mahalanobis"].sum())],
        "pct_anomalias": [round(100 * df_maha_valid["anomalia_mahalanobis"].mean(), 4)],
        "concordancia_media": [round(df_maha_out["metrica_concordancia_maha"].mean(), 1) if not df_maha_out.empty else 0.0],
        "umbral_chi2_promedio": [round(df_maha_valid["umbral_chi2"].mean(), 4)],
    })

    st.markdown("### Resumen Mahalanobis")
    st.dataframe(resumen_maha, use_container_width=True)

    st.markdown("### Top anomalías Mahalanobis")
    cols_maha_show = [
        c for c in [
            "AÑO", "SEMANA", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
            "FLORES_LAG1", "FRUTO CUAJADO",
            "distancia_mahalanobis2", "distancia_mahalanobis",
            "umbral_chi2", "p_value_maha",
            "metrica_concordancia_maha", "nivel_concordancia_maha"
        ] if c in df_maha_out.columns
    ]
    st.dataframe(
        df_maha_out.sort_values(
            ["metrica_concordancia_maha", "distancia_mahalanobis2"],
            ascending=[False, False]
        )[cols_maha_show].head(100),
        use_container_width=True
    )

    col_mh1, col_mh2 = st.columns(2)

    with col_mh1:
        fig_maha_dist = px.scatter(
            df_maha_valid.sort_values("SEMANA") if "SEMANA" in df_maha_valid.columns else df_maha_valid.copy(),
            x="SEMANA" if "SEMANA" in df_maha_valid.columns else "distancia_mahalanobis2",
            y="distancia_mahalanobis2",
            color="flag_outlier_maha",
            hover_data=[
                c for c in [
                    "AÑO", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
                    "FLORES_LAG1", "FRUTO CUAJADO",
                    "umbral_chi2", "p_value_maha",
                    "metrica_concordancia_maha", "nivel_concordancia_maha"
                ] if c in df_maha_valid.columns
            ],
            title="Mahalanobis por semana"
        )

        if "SEMANA" in df_maha_valid.columns and not df_maha_valid["umbral_chi2"].isna().all():
            fig_maha_dist.add_hline(
                y=float(df_maha_valid["umbral_chi2"].dropna().mean()),
                line_dash="dash",
                annotation_text="Umbral"
            )

        fig_maha_dist.update_layout(
            xaxis_title="Semana",
            yaxis_title="Distancia Mahalanobis²"
        )
        st.plotly_chart(fig_maha_dist, use_container_width=True)

    with col_mh2:
        x_col = "FLORES_LAG1"
        y_col = "FRUTO CUAJADO"

        df_maha_plot = df_maha_valid.dropna(subset=[x_col, y_col]).copy()

        normales_maha = df_maha_plot[df_maha_plot["flag_outlier_maha"] == "NORMAL"].copy()
        outliers_maha = df_maha_plot[df_maha_plot["flag_outlier_maha"] == "OUTLIER"].copy()

        fig_maha_plane = go.Figure()

        # NORMALES: fondo más sutil para evitar efecto nube exagerada
        fig_maha_plane.add_trace(
            go.Scatter(
                x=normales_maha[x_col],
                y=normales_maha[y_col],
                mode="markers",
                name="NORMAL",
                marker=dict(
                    size=4,
                    opacity=0.22,
                    color="rgba(120,120,120,0.35)",
                    line=dict(width=0)
                ),
                hovertemplate=(
                    "Flores t-1: %{x}<br>"
                    "Cuajo t: %{y}<extra></extra>"
                )
            )
        )

        # OUTLIERS: encima, más visibles, pero sin burbujas grandes
        fig_maha_plane.add_trace(
            go.Scatter(
                x=outliers_maha[x_col],
                y=outliers_maha[y_col],
                mode="markers",
                name="OUTLIER",
                marker=dict(
                    size=6,
                    opacity=0.95,
                    color="rgba(135,206,250,0.95)",
                    line=dict(width=0)
                ),
                hovertemplate=(
                    "Flores t-1: %{x}<br>"
                    "Cuajo t: %{y}<extra></extra>"
                )
            )
        )

        fig_maha_plane.update_layout(
            title="Plano biológico: FLORES_t-1 vs CUAJO_t",
            xaxis_title="Flores t-1",
            yaxis_title="Cuajo t"
        )

        st.plotly_chart(fig_maha_plane, use_container_width=True)
else:
    st.warning("No hay datos suficientes para calcular Mahalanobis en esta selección.")

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

st.markdown("### Reglas adicionales agregadas de forma puntual")
st.markdown(
    """
- **Relaciones biológicas esenciales**:
  - **FRUTO CUAJADO_t vs FLORES_t-1**
  - **FRUTO VERDE_t vs FRUTO CUAJADO_t-1**
  - **FRUTO VERDE_t vs FLORES_t-2**

  Para estas relaciones se evalúa un **ratio biológico simple** y se aplica IQR al ratio dentro del grupo.

- **Mahalanobis esencial**:
  - **FLORES_t-1**
  - **FRUTO CUAJADO_t**

  Aquí no se evalúa un ratio, sino la **coherencia multivariable del punto completo**
  dentro de la nube histórica del grupo para la relación **CUAJO_t vs FLORES_t-1**.
"""
)
