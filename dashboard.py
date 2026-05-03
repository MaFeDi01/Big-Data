"""
Spotify MPD - Streamlit Dashboard
==================================
Run locally: streamlit run dashboard.py
Requires: pip install streamlit pandas plotly databricks-sql-connector
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Spotify MPD Recommendation Pipeline",
    page_icon="🎵",
    layout="wide"
)

DATA_DIR = os.path.dirname(__file__)

CATALOG = "spotify_project"
SCHEMA  = "mpd"
VOL     = f"/Volumes/{CATALOG}/{SCHEMA}/outputs"

# ── Model metrics (hardcoded from pipeline output) ───────────────────────────
METRICS = {
    "Precision@10": {"ALS only": 0.0300, "After Re-ranking": 0.0415},
    "Recall@10":    {"ALS only": 0.0135, "After Re-ranking": 0.0280},
    "NDCG@10":      {"ALS only": 0.0301, "After Re-ranking": 0.0480},
}
BASELINE_PRECISION = 0.0621
LONGTAIL_COVERAGE  = 22.0
RMSE               = 6.4787
TOTAL_ROWS         = 66_346_428
UNIQUE_PLAYLISTS   = 1_000_000
UNIQUE_TRACKS      = 2_262_292
UNIQUE_ARTISTS     = 295_860

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Dashboard", [
    "Dashboard 1: Pipeline Analytics",
    "Dashboard 2: Recommendations"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### Databricks Connection")

connect_btn = False
host = http_path = token = ""

if "data" in st.session_state:
    st.sidebar.success("Connected to Databricks")
    if st.sidebar.button("Disconnect", use_container_width=True):
        del st.session_state["data"]
        del st.session_state["data_source"]
        st.rerun()
else:
    with st.sidebar.expander("Connect to Databricks", expanded=True):
        host = st.text_input(
            "Workspace URL",
            value="dbc-6b2f8d9c-9761.cloud.databricks.com",
            help="Your Databricks workspace hostname (without https://)"
        )
        http_path = st.text_input(
            "HTTP Path",
            placeholder="/sql/1.0/warehouses/xxxx",
            help="SQL Warehouse → Connection details → HTTP Path"
        )
        token = st.text_input(
            "Access Token",
            type="password",
            placeholder="dapi...",
            help="User Settings → Developer → Access Tokens"
        )
        connect_btn = st.button("Connect & Load Data", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** Spotify MPD Recommendation Pipeline")
st.sidebar.markdown("**Authors:** Diessner, Fiedler, Ryciuk, Leonetti Luparini, De Tuddo")
st.sidebar.markdown("**Dataset:** 66.3M rows · 1M playlists · 2.26M tracks")

# ── Data loading ─────────────────────────────────────────────────────────────
def run_query(conn, sql):
    cursor = conn.cursor()
    cursor.execute(sql)
    cols = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    cursor.close()
    return pd.DataFrame(rows, columns=cols)

@st.cache_resource
def get_connection(host, http_path, token):
    from databricks import sql as dbsql
    return dbsql.connect(
        server_hostname=host.replace("https://", "").strip("/"),
        http_path=http_path.strip(),
        access_token=token.strip()
    )

@st.cache_data(show_spinner="Loading data from Databricks...")
def load_original_tracks_db(host, http_path, token):
    conn = get_connection(host, http_path, token)
    return run_query(conn, f"""
        SELECT f.playlist_idx, f.track_idx, tp.track_name, tp.artist_name, tp.popularity_tier
        FROM (
            SELECT DISTINCT playlist_idx, track_idx
            FROM read_files('{VOL}/mpd_modeling/', format => 'parquet')
        ) f
        INNER JOIN (
            SELECT DISTINCT playlist_idx
            FROM read_files('{VOL}/recommendations_reranked/', format => 'parquet')
        ) r ON f.playlist_idx = r.playlist_idx
        LEFT JOIN read_files('{VOL}/track_index/', format => 'parquet') ti
            ON f.track_idx = ti.track_idx
        LEFT JOIN (
            SELECT track_id, track_name, artist_name, popularity_tier
            FROM read_files('{VOL}/track_popularity/', format => 'parquet')
        ) tp ON ti.track_id = tp.track_id
    """)

@st.cache_data(show_spinner="Loading data from Databricks...")
def load_from_databricks(host, http_path, token):
    conn = get_connection(host, http_path, token)

    top_tracks = run_query(conn, f"""
        SELECT track_name, artist_name, playlist_count, popularity_tier
        FROM read_files('{VOL}/track_popularity/', format => 'parquet')
        ORDER BY playlist_count DESC
        LIMIT 50
    """)

    tier_dist = run_query(conn, f"""
        SELECT popularity_tier, COUNT(*) AS track_count
        FROM read_files('{VOL}/track_popularity/', format => 'parquet')
        GROUP BY popularity_tier
    """)

    track_info = run_query(conn, f"""
        SELECT ti.track_idx, ti.track_id, tp.track_name, tp.artist_name,
               tp.playlist_count, tp.popularity_tier
        FROM read_files('{VOL}/track_index/', format => 'parquet') ti
        LEFT JOIN read_files('{VOL}/track_popularity/', format => 'parquet') tp
            ON ti.track_id = tp.track_id
    """)

    recs = run_query(conn, f"""
        SELECT playlist_idx, track_idx, score, new_rank, popularity_tier
        FROM read_files('{VOL}/recommendations_reranked/', format => 'parquet')
    """)

    playlist_idx = run_query(conn, f"""
        SELECT * FROM read_files('{VOL}/playlist_index/', format => 'parquet')
        LIMIT 500
    """)

    return top_tracks, tier_dist, track_info, recs, playlist_idx

@st.cache_data
def load_from_csv():
    top_tracks   = pd.read_csv(os.path.join(DATA_DIR, "export_top_tracks.csv"))
    tier_dist    = pd.read_csv(os.path.join(DATA_DIR, "export_tier_distribution.csv"))
    track_info   = pd.read_csv(os.path.join(DATA_DIR, "export_track_info.csv"))
    recs         = pd.read_csv(os.path.join(DATA_DIR, "export_recommendations.csv"))
    playlist_idx = pd.read_csv(os.path.join(DATA_DIR, "export_playlist_index.csv"))
    return top_tracks, tier_dist, track_info, recs, playlist_idx

# ── Resolve data source ───────────────────────────────────────────────────────
data_loaded = False
data_source = None

if connect_btn and host and http_path and token:
    try:
        top_tracks, tier_dist, track_info, recs, playlist_idx = load_from_databricks(host, http_path, token)
        original_tracks = load_original_tracks_db(host, http_path, token)
        st.session_state["data"] = (top_tracks, tier_dist, track_info, recs, playlist_idx)
        st.session_state["original_tracks"] = original_tracks
        st.session_state["data_source"] = "databricks"
    except Exception as e:
        st.sidebar.error(f"Connection failed: {e}")

if "data" in st.session_state:
    top_tracks, tier_dist, track_info, recs, playlist_idx = st.session_state["data"]
    original_tracks = st.session_state.get("original_tracks", pd.DataFrame())
    data_loaded = True
    data_source = st.session_state.get("data_source", "databricks")
else:
    try:
        top_tracks, tier_dist, track_info, recs, playlist_idx = load_from_csv()
        orig_path = os.path.join(DATA_DIR, "export_original_tracks.csv")
        original_tracks = pd.read_csv(orig_path) if os.path.exists(orig_path) else pd.DataFrame()
        data_loaded = True
        data_source = "csv"
    except FileNotFoundError:
        data_loaded = False
        original_tracks = pd.DataFrame()

if data_source == "csv":
    st.sidebar.info("Using local CSV exports. Connect to Databricks for live data.")
elif data_source == "databricks":
    st.sidebar.success("Live data from Databricks")

# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD 1 — PIPELINE ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
if page == "Dashboard 1: Pipeline Analytics":
    st.title("Dashboard 1: Pipeline Analytics")
    st.markdown("Overview of the dataset, popularity distribution, and model evaluation results.")

    # ── KPI row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",       f"{TOTAL_ROWS:,}")
    c2.metric("Unique Playlists", f"{UNIQUE_PLAYLISTS:,}")
    c3.metric("Unique Tracks",    f"{UNIQUE_TRACKS:,}")
    c4.metric("Unique Artists",   f"{UNIQUE_ARTISTS:,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    # ── Chart 1: Popularity Tier Distribution ────────────────────────────────
    with col1:
        st.subheader("Chart 1: Track Popularity Tier Distribution")
        if data_loaded:
            tier_order = {"High": 0, "Medium": 1, "Low": 2}
            tier_dist_sorted = tier_dist.sort_values(
                "popularity_tier", key=lambda x: x.map(tier_order)
            )
            colors = {"High": "#1DB954", "Medium": "#FFA500", "Low": "#E8E8E8"}
            fig1 = px.pie(
                tier_dist_sorted,
                values="track_count",
                names="popularity_tier",
                color="popularity_tier",
                color_discrete_map=colors,
                hole=0.4
            )
            fig1.update_traces(textinfo="percent+label+value")
            fig1.update_layout(
                legend_title="Popularity Tier",
                margin=dict(t=20, b=20)
            )
            st.plotly_chart(fig1, use_container_width=True)
            st.caption("Low tier = long-tail (< 4 playlists). 59% of all unique tracks are long-tail.")
        else:
            st.warning("No data available. Connect to Databricks or add local CSV exports.")

    # ── Chart 2: Top 10 Artists ───────────────────────────────────────────────
    with col2:
        st.subheader("Chart 2: Top 10 Artists by Playlist Appearances")
        if data_loaded:
            top_artists = top_tracks.groupby("artist_name")["playlist_count"] \
                .sum().reset_index() \
                .sort_values("playlist_count", ascending=False).head(10)
            fig2 = px.bar(
                top_artists,
                x="playlist_count",
                y="artist_name",
                orientation="h",
                color="playlist_count",
                color_continuous_scale="Greens",
                labels={"playlist_count": "Total Playlist Appearances", "artist_name": "Artist"}
            )
            fig2.update_layout(
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False,
                margin=dict(t=20, b=20)
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Aggregated playlist appearances across all tracks per artist.")
        else:
            st.warning("No data available.")

    col3, col4 = st.columns(2)

    # ── Chart 3: Model Metrics Comparison ────────────────────────────────────
    with col3:
        st.subheader("Chart 3: Model Evaluation — ALS vs. Re-Ranking")
        metrics_df = pd.DataFrame([
            {"Metric": m, "Model": model, "Score": score}
            for m, vals in METRICS.items()
            for model, score in vals.items()
        ])
        fig3 = px.bar(
            metrics_df,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            color_discrete_map={"ALS only": "#636EFA", "After Re-ranking": "#1DB954"},
            text_auto=".4f"
        )
        fig3.update_traces(textposition="outside")
        fig3.update_layout(
            yaxis_title="Score",
            legend_title="",
            margin=dict(t=20, b=20)
        )
        fig3.add_hline(
            y=BASELINE_PRECISION,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Popularity Baseline ({BASELINE_PRECISION:.4f})",
            annotation_position="top left"
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(f"Re-ranking improves all metrics. RMSE: {RMSE:.4f}. Red line = popularity baseline.")

    # ── Chart 4: Top 20 Tracks ────────────────────────────────────────────────
    with col4:
        st.subheader("Chart 4: Top 20 Tracks by Playlist Count")
        if data_loaded:
            top20 = top_tracks.head(20).sort_values("playlist_count", ascending=True)
            top20["label"] = top20["track_name"] + " — " + top20["artist_name"]
            fig4 = px.bar(
                top20,
                x="playlist_count",
                y="label",
                orientation="h",
                color="playlist_count",
                color_continuous_scale="Blues",
                labels={"playlist_count": "Playlist Count", "label": "Track"}
            )
            fig4.update_layout(
                coloraxis_showscale=False,
                margin=dict(t=20, b=20)
            )
            st.plotly_chart(fig4, use_container_width=True)
            st.caption("HUMBLE. by Kendrick Lamar leads with 45,394 playlist appearances.")
        else:
            st.warning("No data available.")

# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD 2 — INTERACTIVE RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════
else:
    st.title("Dashboard 2: Interactive Recommendations")
    st.markdown("Select a playlist to see its top-10 personalized track recommendations.")

    if not data_loaded:
        st.error("No data available. Connect to Databricks or add local CSV exports.")
        st.stop()

    # Merge track names into recommendations
    recs_named = recs.merge(
        track_info[["track_idx", "track_name", "artist_name", "playlist_count"]],
        on="track_idx", how="left"
    )

    # Playlist selector
    available = sorted(recs_named["playlist_idx"].unique())
    selected = st.selectbox(
        "Select Playlist ID",
        options=available,
        format_func=lambda x: f"Playlist {x}"
    )

    user_recs = recs_named[recs_named["playlist_idx"] == selected] \
        .sort_values("new_rank").reset_index(drop=True)

    # ── Original Playlist Tracks ──────────────────────────────────────────────
    if not original_tracks.empty:
        orig = original_tracks[original_tracks["playlist_idx"] == selected].copy()
        if not orig.empty:
            st.markdown("---")
            st.subheader(f"Original Tracks in Playlist {selected} ({len(orig)} tracks)")
            tier_colors = {"High": "🟢", "Medium": "🟡", "Low": "🔵"}
            orig["Tier"] = orig["popularity_tier"].map(lambda t: f"{tier_colors.get(t, '⚪')} {t}")
            st.dataframe(
                orig[["track_name", "artist_name", "Tier"]].rename(
                    columns={"track_name": "Track", "artist_name": "Artist"}
                ),
                use_container_width=True, hide_index=True
            )

    st.markdown("---")
    col1, col2 = st.columns(2)

    # ── Chart 1: Recommended Tracks Table ────────────────────────────────────
    with col1:
        st.subheader("Chart 1: Top-10 Recommended Tracks")
        tier_colors = {"High": "🟢", "Medium": "🟡", "Low": "🔵"}
        display_df = user_recs[["new_rank", "track_name", "artist_name", "popularity_tier", "score"]].copy()
        display_df.columns = ["Rank", "Track", "Artist", "Tier", "Score"]
        display_df["Tier"] = display_df["Tier"].map(
            lambda t: f"{tier_colors.get(t, '⚪')} {t}"
        )
        display_df["Score"] = display_df["Score"].round(4)
        display_df["Rank"] = display_df["Rank"].astype(int)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption("🔵 Low tier = long-tail track. 2 guaranteed per playlist via slot-based re-ranking.")

    # ── Chart 2: Tier Distribution of Recommendations ────────────────────────
    with col2:
        st.subheader("Chart 2: Tier Distribution of Recommendations")
        tier_counts = user_recs["popularity_tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        colors = {"High": "#1DB954", "Medium": "#FFA500", "Low": "#1E90FF"}
        fig_tier = px.pie(
            tier_counts,
            values="Count",
            names="Tier",
            color="Tier",
            color_discrete_map=colors,
            hole=0.4
        )
        fig_tier.update_traces(textinfo="percent+label")
        fig_tier.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig_tier, use_container_width=True)

    col3, col4 = st.columns(2)

    # ── Chart 3: ALS Score per Track ─────────────────────────────────────────
    with col3:
        st.subheader("Chart 3: ALS Score per Recommended Track")
        user_recs["label"] = user_recs["track_name"].str[:30] + "..."
        tier_color_map = {"High": "#1DB954", "Medium": "#FFA500", "Low": "#1E90FF"}
        fig_scores = px.bar(
            user_recs.sort_values("new_rank"),
            x="label",
            y="score",
            color="popularity_tier",
            color_discrete_map=tier_color_map,
            labels={"label": "Track", "score": "ALS Score", "popularity_tier": "Tier"},
            text_auto=".3f"
        )
        fig_scores.update_layout(
            xaxis_tickangle=-45,
            margin=dict(t=20, b=80),
            legend_title="Tier"
        )
        st.plotly_chart(fig_scores, use_container_width=True)
        st.caption("Low-tier tracks have lower raw ALS scores but are guaranteed 2 slots via re-ranking.")

    # ── Chart 4: Overall Long-tail Coverage ──────────────────────────────────
    with col4:
        st.subheader("Chart 4: Long-tail Coverage — Before vs. After Re-ranking")
        coverage_df = pd.DataFrame({
            "Stage": ["ALS only", "After Re-ranking"],
            "Long-tail Coverage (%)": [0.0, LONGTAIL_COVERAGE]
        })
        fig_cov = px.bar(
            coverage_df,
            x="Stage",
            y="Long-tail Coverage (%)",
            color="Stage",
            color_discrete_map={"ALS only": "#636EFA", "After Re-ranking": "#1DB954"},
            text_auto=".1f"
        )
        fig_cov.update_traces(textposition="outside")
        fig_cov.update_layout(
            yaxis=dict(range=[0, 35]),
            showlegend=False,
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_cov, use_container_width=True)
        st.caption("Slot-based re-ranking guarantees 22% long-tail exposure across all users.")
