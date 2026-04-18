"""
Spotify Hit Predictor — Streamlit Dashboard

Five pages covering the full project lifecycle:
  Overview       — problem framing, dataset stats, class imbalance
  EDA            — interactive distributions, correlation, PCA/t-SNE, radar profile
  Feature Eng.   — engineered features vs originals, Cohen's d comparisons
  Models         — side-by-side metrics, ROC/PR curves, CV score distributions
  Live Predict   — real-time hit-probability gauge from user-defined audio features

Run with:
    cd <repo-root>
    streamlit run dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ── Config ──────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT / "data" / "processed" / "features.csv"
MODEL_PATH = ROOT / "models" / "hit_predictor.joblib"

ORIG_COLS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]
ENG_COLS = ["dance_x_energy", "vocal_presence", "electronic_score", "rap_signal", "loudness_norm"]
ALL_COLS = ORIG_COLS + ENG_COLS

# Slider ranges for live-prediction page  (min, max, default)
SLIDER_RANGES = {
    "danceability":     (0.0,  1.0,   0.60),
    "energy":           (0.0,  1.0,   0.65),
    "loudness":         (-60.0, 0.0,  -7.0),
    "speechiness":      (0.0,  1.0,   0.08),
    "acousticness":     (0.0,  1.0,   0.25),
    "instrumentalness": (0.0,  1.0,   0.00),
    "liveness":         (0.0,  1.0,   0.12),
    "valence":          (0.0,  1.0,   0.50),
    "tempo":            (40.0, 220.0, 120.0),
}

FEATURE_DESCRIPTIONS = {
    "danceability":     "How suitable for dancing (0 = least, 1 = most)",
    "energy":           "Intensity and activity level (0–1)",
    "loudness":         "Average loudness in dB (typical range −20 to −1)",
    "speechiness":      "Presence of spoken words (0–1)",
    "acousticness":     "Confidence the track is acoustic (0–1)",
    "instrumentalness": "Likelihood of no vocals (0 = vocal, 1 = instrumental)",
    "liveness":         "Probability of a live audience (0–1)",
    "valence":          "Musical positiveness / happiness (0–1)",
    "tempo":            "Estimated beats per minute",
}

# ── Page setup ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Spotify Hit Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — Spotify-inspired dark card aesthetic
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #121212; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 16px;
    }

    /* Callout boxes */
    .insight-box {
        background-color: #1DB954;
        color: #000000;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
        font-weight: 500;
    }
    .warning-box {
        background-color: #FF6B35;
        color: #000000;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #282828;
        color: #FFFFFF;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
        border-left: 4px solid #1DB954;
    }

    /* Section dividers */
    hr { border-color: #333; }

    /* Main bg */
    .stApp { background-color: #0D0D0D; color: #FFFFFF; }
</style>
""", unsafe_allow_html=True)

SPOTIFY_GREEN = "#1DB954"
PALETTE = {"Hit": "#1DB954", "Non-Hit": "#535353"}

# ── Data & model (cached) ─────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(f"Data file not found at {DATA_PATH}. Run notebooks 01–02 first.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    # Reproduce engineered features identically to training script
    df["dance_x_energy"]   = df["danceability"] * df["energy"]
    df["vocal_presence"]   = 1.0 - df["instrumentalness"]
    df["electronic_score"] = df["energy"] - df["acousticness"]
    df["rap_signal"]       = df["speechiness"] * df["danceability"]
    lo, hi                  = df["loudness"].min(), df["loudness"].max()
    df["loudness_norm"]    = (df["loudness"] - lo) / (hi - lo)
    return df


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def compute_pca_projection(df: pd.DataFrame) -> tuple[np.ndarray, PCA]:
    sc  = StandardScaler().fit(df[ORIG_COLS])
    X   = sc.transform(df[ORIG_COLS])
    pca = PCA(n_components=3).fit(X)
    return pca.transform(X), pca


@st.cache_data
def compute_tsne_projection(df: pd.DataFrame) -> np.ndarray:
    sc = StandardScaler().fit(df[ORIG_COLS])
    X  = sc.transform(df[ORIG_COLS])
    return TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000).fit_transform(X)


@st.cache_data
def cohens_d_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    hits = df[df["label"] == 1]
    nons = df[df["label"] == 0]
    rows = []
    for f in cols:
        diff   = hits[f].mean() - nons[f].mean()
        pooled = np.sqrt((hits[f].std()**2 + nons[f].std()**2) / 2)
        rows.append({"feature": f, "cohens_d": diff / pooled if pooled else 0})
    return pd.DataFrame(rows).sort_values("cohens_d")


# ── Helper: insight callout ───────────────────────────────────────────────

def insight(text: str, kind: str = "info") -> None:
    css = {"info": "info-box", "good": "insight-box", "warn": "warning-box"}[kind]
    st.markdown(f'<div class="{css}">💡 {text}</div>', unsafe_allow_html=True)


# ── Page renderers ────────────────────────────────────────────────────────

def page_overview(df: pd.DataFrame) -> None:
    st.title("🎵 Spotify Hit Predictor")
    st.markdown(
        "Predicting Billboard Hot 100 **top-10 hits** from Spotify audio features "
        "using ensemble machine-learning models trained on 2,000+ tracks."
    )
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Tracks",   f"{len(df):,}")
    c2.metric("Hits",           f"{int((df['label']==1).sum()):,}")
    c3.metric("Non-Hits",       f"{int((df['label']==0).sum()):,}")
    c4.metric("Imbalance Ratio", f"{(df['label']==0).sum()/(df['label']==1).sum():.0f}:1")
    c5.metric("Audio Features", "9 raw + 5 engineered")

    st.markdown("---")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Class Distribution")
        counts = df["label"].value_counts().sort_index()
        fig = px.pie(
            values=counts.values,
            names=["Non-Hit", "Hit"],
            color_discrete_sequence=["#535353", SPOTIFY_GREEN],
            hole=0.55,
        )
        fig.update_layout(
            paper_bgcolor="#0D0D0D", plot_bgcolor="#0D0D0D",
            font_color="white", legend_font_color="white",
            margin=dict(t=10, b=10),
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
        insight(
            "96.4% of tracks are non-hits — a 27:1 imbalance. "
            "We use SMOTE + class-weighted models; accuracy is not a valid metric here.",
            "warn",
        )

    with col_right:
        st.subheader("Problem Statement")
        st.markdown("""
        Record labels and streaming platforms invest significant resources in promotion
        without reliable early signals of commercial success.

        **Our approach:** Use the nine Spotify audio attributes — danceability, energy,
        loudness, valence, tempo, etc. — to train classifiers that assign a *hit probability*
        to any track.

        **Value proposition:** An A&R team can use this score to prioritize promotion
        budget toward tracks with the highest predicted hit potential, before investing in
        expensive campaigns.
        """)

        st.subheader("Project Pipeline")
        st.markdown("""
        | Step | Description |
        |---|---|
        | 1. Data Acquisition | Billboard Hot 100 × Spotify API merge (Polars + PySpark) |
        | 2. EDA | Distributions, correlations, PCA, t-SNE |
        | 3. Feature Engineering | 5 interaction/transformation features from EDA insights |
        | 4. Modeling | LR baseline → RF + XGBoost with SMOTE + RandomizedSearchCV |
        | 5. Dashboard | This interactive Streamlit app |
        """)


def page_eda(df: pd.DataFrame) -> None:
    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Correlations", "PCA & t-SNE", "Radar Profile"])

    # ── Tab 1: Distributions ──────────────────────────────────────────────
    with tab1:
        st.subheader("Feature Distributions — Hits vs Non-Hits")
        feat = st.selectbox("Select feature:", ORIG_COLS, index=0, key="eda_feat")
        desc = FEATURE_DESCRIPTIONS.get(feat, "")
        st.caption(desc)

        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.histogram(
                df, x=feat, color=df["label"].map({0: "Non-Hit", 1: "Hit"}),
                nbins=40, barmode="overlay", opacity=0.7,
                color_discrete_map=PALETTE,
                labels={"color": "Class"},
                title=f"{feat} — histogram",
            )
            fig.update_layout(paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E",
                              font_color="white", legend_font_color="white")
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig = px.box(
                df, x=df["label"].map({0: "Non-Hit", 1: "Hit"}), y=feat,
                color=df["label"].map({0: "Non-Hit", 1: "Hit"}),
                color_discrete_map=PALETTE,
                title=f"{feat} — box plot",
                labels={"x": "Class", "color": "Class"},
            )
            fig.update_layout(paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E",
                              font_color="white", legend_font_color="white")
            st.plotly_chart(fig, use_container_width=True)

        insight(
            "Hits (green) tend to be more danceable, louder, and less instrumental than non-hits. "
            "Valence shifts in the opposite direction — chart hits skew moodier/darker.",
        )

        # Outlier summary table
        with st.expander("🔍 Outlier Report (IQR method)"):
            rows = []
            for f in ORIG_COLS:
                q1, q3 = df[f].quantile(0.25), df[f].quantile(0.75)
                iqr = q3 - q1
                n = int(((df[f] < q1 - 1.5*iqr) | (df[f] > q3 + 1.5*iqr)).sum())
                rows.append({"Feature": f, "Outliers (IQR)": n, "Pct": f"{n/len(df)*100:.1f}%",
                             "Lower Fence": round(q1 - 1.5*iqr, 3), "Upper Fence": round(q3 + 1.5*iqr, 3)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            insight(
                "Outliers in speechiness, liveness, and instrumentalness represent real musical "
                "variety (rap, live albums, classical) — not errors. We retain them and use "
                "tree-based models + StandardScaler to handle the skew.",
            )

    # ── Tab 2: Correlations ───────────────────────────────────────────────
    with tab2:
        st.subheader("Feature Correlation Heatmap")
        corr = df[ORIG_COLS].corr()
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        )
        fig.update_layout(paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E",
                          font_color="white", coloraxis_colorbar_tickfont_color="white")
        st.plotly_chart(fig, use_container_width=True)
        insight(
            "energy ↔ loudness (r ≈ +0.7) and energy ↔ acousticness (r ≈ −0.7) are the "
            "dominant correlations — loud tracks are energetic, acoustic tracks are soft. "
            "These motivate our electronic_score engineered feature (energy − acousticness)."
        )

        st.subheader("Scatter Matrix (selected features)")
        scatter_feats = st.multiselect(
            "Choose features:", ORIG_COLS,
            default=["danceability", "energy", "loudness", "speechiness", "valence"],
            key="scatter_feats",
        )
        if len(scatter_feats) >= 2:
            plot_df = df[scatter_feats + ["label"]].copy()
            plot_df["Class"] = plot_df["label"].map({0: "Non-Hit", 1: "Hit"})
            fig = px.scatter_matrix(
                plot_df, dimensions=scatter_feats, color="Class",
                color_discrete_map=PALETTE, opacity=0.4,
                title="Pair-plot (select features above)",
            )
            fig.update_traces(diagonal_visible=True, showupperhalf=False)
            fig.update_layout(paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E",
                              font_color="white", height=600)
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: PCA & t-SNE ───────────────────────────────────────────────
    with tab3:
        st.subheader("Dimensionality Reduction")
        method = st.radio("Projection:", ["PCA (linear)", "t-SNE (non-linear)"], horizontal=True)

        if method.startswith("PCA"):
            X_proj, pca = compute_pca_projection(df)
            var = pca.explained_variance_ratio_
            plot_df = pd.DataFrame({
                "PC1": X_proj[:, 0], "PC2": X_proj[:, 1],
                "Class": df["label"].map({0: "Non-Hit", 1: "Hit"}),
                "title": df["title"], "artist": df["artist"],
            })
            fig = px.scatter(
                plot_df, x="PC1", y="PC2", color="Class",
                hover_data=["title", "artist"],
                color_discrete_map=PALETTE, opacity=0.55,
                labels={
                    "PC1": f"PC1 ({var[0]:.1%} var)",
                    "PC2": f"PC2 ({var[1]:.1%} var)",
                },
                title=f"PCA — first two components capture {var[0]+var[1]:.1%} of variance",
            )

            # Feature loading arrows
            loadings = pca.components_[:2].T
            for i, f in enumerate(ORIG_COLS):
                fig.add_annotation(
                    ax=0, ay=0, x=loadings[i, 0] * 3, y=loadings[i, 1] * 3,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                    arrowcolor="rgba(255,200,0,0.8)",
                )
                fig.add_annotation(
                    x=loadings[i, 0] * 3.3, y=loadings[i, 1] * 3.3,
                    text=f"<b>{f}</b>", showarrow=False,
                    font=dict(size=9, color="rgba(255,200,0,0.9)"),
                )
            fig.update_layout(paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E",
                              font_color="white", legend_font_color="white")
            st.plotly_chart(fig, use_container_width=True)

            col_v1, col_v2, col_v3 = st.columns(3)
            for ax_idx, (col, label) in enumerate([(col_v1, "PC1"), (col_v2, "PC2"), (col_v3, "PC3")]):
                ldf = pd.DataFrame({"feature": ORIG_COLS, "loading": pca.components_[ax_idx]})
                ldf = ldf.sort_values("loading")
                fig_l = px.bar(ldf, x="loading", y="feature", orientation="h",
                               title=f"{label} ({pca.explained_variance_ratio_[ax_idx]:.1%})",
                               color="loading", color_continuous_scale="RdYlGn")
                fig_l.update_layout(paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E",
                                    font_color="white", height=280, margin=dict(t=30, b=0))
                col.plotly_chart(fig_l, use_container_width=True)

            insight(
                "PC1 (energy/loudness axis) and PC2 (danceability/valence axis) together capture 43% "
                "of variance. Hits concentrate in the high-PC1, mid-PC2 region — loud, danceable, moderate valence."
            )

        else:
            with st.spinner("Computing t-SNE (takes ~10 s)..."):
                X_tsne = compute_tsne_projection(df)
            plot_df = pd.DataFrame({
                "x": X_tsne[:, 0], "y": X_tsne[:, 1],
                "Class": df["label"].map({0: "Non-Hit", 1: "Hit"}),
                "title": df["title"], "artist": df["artist"],
            })
            fig = px.scatter(
                plot_df, x="x", y="y", color="Class",
                hover_data=["title", "artist"],
                color_discrete_map=PALETTE, opacity=0.55,
                labels={"x": "t-SNE dim 1", "y": "t-SNE dim 2"},
                title="t-SNE — non-linear projection revealing cluster structure",
            )
            fig.update_layout(paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E",
                              font_color="white", legend_font_color="white")
            st.plotly_chart(fig, use_container_width=True)
            insight(
                "t-SNE reveals that hits are not a single cluster — they form 2–3 subgroups "
                "(likely rap/hip-hop, pop, and R&B), each with a distinct audio fingerprint. "
                "This heterogeneity explains why a single linear model struggles.",
            )

    # ── Tab 4: Radar profile ──────────────────────────────────────────────
    with tab4:
        st.subheader("Average Feature Profile — Hits vs Non-Hits")

        # Normalise all features to [0,1] for fair radar comparison
        norm_df = df[ORIG_COLS].copy()
        for c in ORIG_COLS:
            lo, hi = norm_df[c].min(), norm_df[c].max()
            norm_df[c] = (norm_df[c] - lo) / (hi - lo) if hi > lo else 0

        hit_mean  = norm_df[df["label"] == 1].mean()
        nohit_mean = norm_df[df["label"] == 0].mean()

        cats = ORIG_COLS + [ORIG_COLS[0]]  # close the polygon
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=hit_mean.tolist() + [hit_mean.iloc[0]],
            theta=cats, fill="toself", name="Hit",
            line_color=SPOTIFY_GREEN, fillcolor="rgba(29,185,84,0.25)",
        ))
        fig.add_trace(go.Scatterpolar(
            r=nohit_mean.tolist() + [nohit_mean.iloc[0]],
            theta=cats, fill="toself", name="Non-Hit",
            line_color="#535353", fillcolor="rgba(83,83,83,0.25)",
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="#1E1E1E",
                radialaxis=dict(visible=True, range=[0, 1], color="white"),
                angularaxis=dict(color="white"),
            ),
            paper_bgcolor="#0D0D0D", font_color="white",
            legend_font_color="white",
            title="Normalised feature averages — hover to compare",
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)
        insight(
            "Hits (green) score higher on danceability, loudness, and speechiness, "
            "and lower on acousticness, instrumentalness, and valence — "
            "the classic profile of a modern vocal pop/hip-hop hit.",
            "good",
        )


def page_feature_engineering(df: pd.DataFrame) -> None:
    st.title("🔧 Feature Engineering")

    st.markdown("""
    Five new features were derived from EDA findings to capture signal that raw features miss individually.
    All features are constructed before the train/test split; the scaler sees the full 14-feature set.
    """)

    table = pd.DataFrame([
        {"Feature": "dance_x_energy",   "Formula": "danceability × energy",        "EDA Motivation": "Hits cluster in high-dance + high-energy quadrant"},
        {"Feature": "vocal_presence",   "Formula": "1 − instrumentalness",          "EDA Motivation": "Nearly all hits are vocal tracks (instrumentalness ≈ 0)"},
        {"Feature": "electronic_score", "Formula": "energy − acousticness",         "EDA Motivation": "energy ↔ acousticness anti-correlation; hits are electronic"},
        {"Feature": "rap_signal",       "Formula": "speechiness × danceability",    "EDA Motivation": "Captures rap/hip-hop quadrant (high speech + high dance)"},
        {"Feature": "loudness_norm",    "Formula": "(loudness − min) / (max − min)", "EDA Motivation": "Rescales unbounded dB values to [0,1] for comparability"},
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Discriminability: Engineered vs Original Features")
    d_all = cohens_d_table(df, ORIG_COLS + ENG_COLS)
    d_all["type"] = d_all["feature"].apply(lambda f: "Engineered" if f in ENG_COLS else "Original")
    d_all_sorted = d_all.sort_values("cohens_d")

    fig = px.bar(
        d_all_sorted, x="cohens_d", y="feature", orientation="h", color="type",
        color_discrete_map={"Engineered": SPOTIFY_GREEN, "Original": "#535353"},
        labels={"cohens_d": "Cohen's d (hits − non-hits, pooled SD)", "feature": ""},
        title="Engineered features (green) rank among the top discriminators",
    )
    fig.add_vline(x=0, line_color="white", line_width=1)
    fig.add_vline(x=0.2,  line_dash="dot", line_color="orange", opacity=0.5)
    fig.add_vline(x=-0.2, line_dash="dot", line_color="orange", opacity=0.5)
    fig.update_layout(paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E",
                      font_color="white", legend_font_color="white", height=500)
    st.plotly_chart(fig, use_container_width=True)

    insight(
        "vocal_presence and rap_signal both achieve Cohen's d stronger than their constituent "
        "raw features — confirmation that engineering interaction/complement terms adds genuine signal.",
        "good",
    )

    # Distribution of each engineered feature by class
    st.subheader("Engineered Feature Distributions")
    eng_feat = st.selectbox("Inspect engineered feature:", ENG_COLS, key="eng_feat")
    fig = px.histogram(
        df, x=eng_feat, color=df["label"].map({0: "Non-Hit", 1: "Hit"}),
        nbins=40, barmode="overlay", opacity=0.7,
        color_discrete_map=PALETTE,
        labels={"color": "Class"},
        title=f"{eng_feat} distribution by class",
    )
    fig.update_layout(paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E",
                      font_color="white", legend_font_color="white")
    st.plotly_chart(fig, use_container_width=True)


def page_models(artifacts: dict | None) -> None:
    st.title("📈 Model Comparison & Evaluation")

    if artifacts is None:
        st.warning("No trained model found at `models/hit_predictor.joblib`. Run `scripts/train_models.py` first.")
        return

    results = pd.DataFrame(artifacts["results"])

    st.markdown("""
    Three models were trained using the same `imblearn` pipeline:
    **StandardScaler → SMOTE → Classifier**, with 5-fold stratified `RandomizedSearchCV` tuning.
    Evaluation uses the held-out 20% test split — **never seen during training or tuning**.
    """)

    # Metrics table with highlight
    st.subheader("Held-Out Test Metrics")
    best_row = results.loc[results["roc_auc"].idxmax()]
    st.dataframe(
        results.style
            .format({"roc_auc": "{:.3f}", "pr_auc": "{:.3f}", "f1": "{:.3f}",
                     "precision": "{:.3f}", "recall": "{:.3f}"})
            .highlight_max(subset=["roc_auc", "pr_auc", "f1"], color="#1A3D24"),
        use_container_width=True,
    )
    insight(
        f"Best model: **{best_row['model']}** (ROC-AUC {best_row['roc_auc']:.3f}, "
        f"PR-AUC {best_row['pr_auc']:.3f}). "
        "PR-AUC is the primary metric under heavy class imbalance — "
        "it penalises models that achieve high AUC by simply ranking non-hits well.",
        "good",
    )

    st.markdown("---")
    st.subheader("Why Not Accuracy?")
    st.markdown("""
    <div class="info-box">
    A model that predicts <em>non-hit</em> for every track would score <strong>96.4% accuracy</strong>
    while being completely useless. We report ROC-AUC, PR-AUC, F1, precision, and recall —
    all of which explicitly account for the minority class.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    # Grouped bar chart
    metric_cols = ["roc_auc", "pr_auc", "f1", "precision", "recall"]
    melted = results.melt(id_vars="model", value_vars=metric_cols,
                          var_name="Metric", value_name="Score")
    fig = px.bar(
        melted, x="Metric", y="Score", color="model", barmode="group",
        color_discrete_sequence=[SPOTIFY_GREEN, "#535353", "#FF6B35"],
        range_y=[0, 1], title="All models, all metrics",
    )
    fig.update_layout(paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E",
                      font_color="white", legend_font_color="white")
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.markdown("---")
    st.subheader("Feature Importance — Tree Models")
    imp = pd.DataFrame(artifacts["feature_importance"])
    imp["is_engineered"] = imp["feature"].isin(ENG_COLS)
    imp_sorted = imp.sort_values("xgb_importance", ascending=True)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Random Forest", "XGBoost"))
    rf_colors  = [SPOTIFY_GREEN if e else "#535353" for e in imp_sorted["is_engineered"]]
    xgb_colors = [SPOTIFY_GREEN if e else "#FF6B35" for e in imp_sorted["is_engineered"]]

    fig.add_trace(go.Bar(
        y=imp_sorted["feature"], x=imp_sorted["rf_importance"],
        orientation="h", marker_color=rf_colors, name="RF",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        y=imp_sorted["feature"], x=imp_sorted["xgb_importance"],
        orientation="h", marker_color=xgb_colors, name="XGB",
    ), row=1, col=2)
    fig.update_layout(
        paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E",
        font_color="white", showlegend=False, height=460,
        title_text="Green = engineered features",
    )
    st.plotly_chart(fig, use_container_width=True)
    insight(
        "vocal_presence and rap_signal (engineered) appear in the top half of both "
        "tree models, validating that our feature engineering improved signal quality.",
    )


def page_live_prediction(artifacts: dict | None) -> None:
    st.title("🎯 Live Hit Predictor")

    if artifacts is None:
        st.warning("Train the model first (`scripts/train_models.py`) to enable this page.")
        return

    st.markdown(
        "Dial in any track's audio profile and score its hit probability in real time using "
        f"our best model: **{artifacts['best_name']}**."
    )

    cols = st.columns(3)
    raw_vals: dict[str, float] = {}
    for i, feat in enumerate(ORIG_COLS):
        lo, hi, default = SLIDER_RANGES[feat]
        step = 0.01 if hi <= 1.0 else 1.0
        with cols[i % 3]:
            raw_vals[feat] = st.slider(
                feat, min_value=float(lo), max_value=float(hi),
                value=float(default), step=float(step),
                help=FEATURE_DESCRIPTIONS.get(feat, ""),
            )

    # Build full 14-feature vector (engineered features derived from sliders)
    row = pd.DataFrame([raw_vals])
    row["dance_x_energy"]   = row["danceability"] * row["energy"]
    row["vocal_presence"]   = 1.0 - row["instrumentalness"]
    row["electronic_score"] = row["energy"] - row["acousticness"]
    row["rap_signal"]       = row["speechiness"] * row["danceability"]
    # loudness_norm uses training-set min/max — approximate from data
    df_for_bounds = load_data()
    lo_l, hi_l = df_for_bounds["loudness"].min(), df_for_bounds["loudness"].max()
    row["loudness_norm"] = (row["loudness"] - lo_l) / (hi_l - lo_l)

    model = artifacts["best_model"]
    X_input = row[artifacts["feature_cols"]].values
    proba = float(model.predict_proba(X_input)[0, 1])

    st.markdown("---")
    c_gauge, c_breakdown = st.columns([1, 2])

    with c_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=proba * 100,
            number={"suffix": "%", "font": {"color": "white"}},
            delta={"reference": 3.6, "suffix": " pp vs base rate"},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"color": "white"}},
                "bar": {"color": SPOTIFY_GREEN if proba > 0.5 else "#535353"},
                "bgcolor": "#1E1E1E",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 33],  "color": "#1A1A1A"},
                    {"range": [33, 66], "color": "#222222"},
                    {"range": [66, 100], "color": "#1A2A1A"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.85,
                    "value": 50,
                },
            },
            title={"text": "Hit Probability", "font": {"color": "white"}},
        ))
        fig.update_layout(
            paper_bgcolor="#0D0D0D", height=300,
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        verdict = "🔥 Likely a Hit!" if proba > 0.5 else "❄️ Likely a Non-Hit"
        st.markdown(
            f'<div class="{"insight-box" if proba>0.5 else "info-box"}" '
            f'style="text-align:center;font-size:1.2em">{verdict}</div>',
            unsafe_allow_html=True,
        )

    with c_breakdown:
        st.subheader("Track Audio Profile vs Average Hit")
        hit_means = df_for_bounds[df_for_bounds["label"] == 1][ORIG_COLS].mean()

        # Normalise for radar
        norm_vals, norm_hits = [], []
        for f in ORIG_COLS:
            lo_f = df_for_bounds[f].min()
            hi_f = df_for_bounds[f].max()
            span = hi_f - lo_f if hi_f > lo_f else 1
            norm_vals.append((raw_vals[f] - lo_f) / span)
            norm_hits.append((hit_means[f] - lo_f) / span)

        cats = ORIG_COLS + [ORIG_COLS[0]]
        norm_vals_closed = norm_vals + [norm_vals[0]]
        norm_hits_closed = norm_hits + [norm_hits[0]]

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=norm_vals_closed, theta=cats, fill="toself", name="Your Track",
            line_color="#FF6B35", fillcolor="rgba(255,107,53,0.3)",
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=norm_hits_closed, theta=cats, fill="toself", name="Avg Hit",
            line_color=SPOTIFY_GREEN, fillcolor="rgba(29,185,84,0.2)",
        ))
        fig_r.update_layout(
            polar=dict(
                bgcolor="#1E1E1E",
                radialaxis=dict(range=[0, 1], color="white"),
                angularaxis=dict(color="white"),
            ),
            paper_bgcolor="#0D0D0D", font_color="white",
            legend_font_color="white", height=340,
            margin=dict(t=30, b=10),
        )
        st.plotly_chart(fig_r, use_container_width=True)

        gap_df = pd.DataFrame({
            "Feature": ORIG_COLS,
            "Your Track": [round(v, 3) for v in norm_vals],
            "Avg Hit":    [round(h, 3) for h in norm_hits],
            "Gap":        [round(v - h, 3) for v, h in zip(norm_vals, norm_hits)],
        }).sort_values("Gap", key=abs, ascending=False)
        st.caption("Feature gaps vs average hit (normalised)")
        st.dataframe(gap_df, use_container_width=True, hide_index=True)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    df        = load_data()
    artifacts = load_model()

    st.sidebar.markdown(
        f'<div style="text-align:center;padding:12px">'
        f'<span style="font-size:2em">🎵</span><br>'
        f'<span style="font-size:1.1em;font-weight:700;color:{SPOTIFY_GREEN}">Hit Predictor</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "EDA", "Feature Engineering", "Models", "Live Prediction"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**CIS 2420 — SP2026**  \n"
        f"Benjamin Liu & William Zhang  \n"
        f"Model: `{'✔ ' + artifacts['best_name'] if artifacts else '✘ missing'}`"
    )

    if page == "Overview":
        page_overview(df)
    elif page == "EDA":
        page_eda(df)
    elif page == "Feature Engineering":
        page_feature_engineering(df)
    elif page == "Models":
        page_models(artifacts)
    elif page == "Live Prediction":
        page_live_prediction(artifacts)


if __name__ == "__main__":
    main()
