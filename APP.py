with col_mh2:
    x_col = "FLORES_LAG1"
    y_col = "FRUTO CUAJADO"

    df_maha_plot = df_maha_valid.dropna(subset=[x_col, y_col]).copy()

    # Separar normales y outliers
    normales_maha = df_maha_plot[df_maha_plot["flag_outlier_maha"] == "NORMAL"].copy()
    outliers_maha = df_maha_plot[df_maha_plot["flag_outlier_maha"] == "OUTLIER"].copy()

    hover_cols_maha = [
        c for c in [
            "AÑO", "SEMANA", "ETAPA", "CAMPO", "TURNO", "VARIEDAD",
            "FLORES_LAG1", "FRUTO CUAJADO",
            "distancia_mahalanobis2", "p_value_maha",
            "metrica_concordancia_maha", "nivel_concordancia_maha"
        ] if c in df_maha_plot.columns
    ]

    fig_maha_plane = go.Figure()

    # 🔵 NORMALES (fondo)
    fig_maha_plane.add_trace(
        go.Scatter(
            x=normales_maha[x_col],
            y=normales_maha[y_col],
            mode="markers",
            name="NORMAL",
            marker=dict(
                size=5,
                opacity=0.35,
                color="#1f77b4",
                line=dict(width=0)
            ),
            customdata=normales_maha[hover_cols_maha].values if len(hover_cols_maha) > 0 else None,
            hovertemplate=(
                "Flores t-1: %{x}<br>"
                "Cuajo t: %{y}<br>"
                "<extra></extra>"
            )
        )
    )

    # 🔵 OUTLIERS (encima)
    fig_maha_plane.add_trace(
        go.Scatter(
            x=outliers_maha[x_col],
            y=outliers_maha[y_col],
            mode="markers",
            name="OUTLIER",
            marker=dict(
                size=7,
                opacity=0.95,
                color="#87CEFA",
                line=dict(width=0)
            ),
            customdata=outliers_maha[hover_cols_maha].values if len(hover_cols_maha) > 0 else None,
            hovertemplate=(
                "Flores t-1: %{x}<br>"
                "Cuajo t: %{y}<br>"
                "<extra></extra>"
            )
        )
    )

    fig_maha_plane.update_layout(
        title="Plano biológico: FLORES_t-1 vs CUAJO_t",
        xaxis_title="Flores t-1",
        yaxis_title="Cuajo t"
    )

    st.plotly_chart(fig_maha_plane, use_container_width=True)
