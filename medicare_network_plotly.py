import csv
import math
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
from networkx.algorithms import community
from collections import defaultdict

# ── Install: pip install streamlit plotly networkx pandas ──

st.set_page_config(
    page_title="Medicare Denial Network",
    page_icon="🏥",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
[data-testid="stSidebar"] { background-color: #0f1117; border-right: 1px solid #1e2330; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
.metric-card {
    background: #161b22; border: 1px solid #1e2330;
    border-radius: 8px; padding: 16px 20px; text-align: center;
}
.metric-card .value {
    font-family: 'IBM Plex Mono', monospace; font-size: 2rem;
    font-weight: 600; color: #58a6ff; line-height: 1;
}
.metric-card .label {
    font-size: 0.75rem; color: #8b949e;
    text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FILE_PATH = "Medicare_FFS_CERT_2025.csv"

PALETTE = [
    "#82131C", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#6A4C93", "#1982C4", "#8AC926",
    "#FF595E", "#FFCA3A",
]

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parameters")
    st.markdown("---")
    similarity_threshold = st.slider(
        "Similarity threshold", 0.40, 0.95, 0.75, 0.05,
        help="Minimum cosine similarity to draw an edge between two providers."
    )
    top_k_edges = st.slider(
        "Max edges per node", 1, 15, 5, 1,
        help="Each provider keeps only its top-K strongest connections."
    )
    label_min_denials = st.slider(
        "Label providers with >= N denials", 0, 500, 50, 10,
        help="Only providers above this raw denial count get a name label."
    )
    use_tfidf = st.toggle("TF-IDF weighting", value=True,
        help="Down-weights error codes common to all providers.")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#484f58;line-height:1.9'>"
        "Node size = raw denial volume<br>"
        "Edge thickness = similarity score<br>"
        "Hover edges to see similarity<br>"
        "Scroll to zoom · drag to pan<br>"
        "Color = community cluster"
        "</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# DATA PIPELINE  (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_and_deduplicate(path):
    raw_rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw_rows.append(row)
    seen, data = set(), []
    for row in raw_rows:
        key = frozenset(row.items())
        if key not in seen:
            seen.add(key)
            data.append(row)
    return raw_rows, data


@st.cache_data
def build_profiles(data, use_tfidf):
    provider_patterns = defaultdict(lambda: defaultdict(int))
    provider_totals   = defaultdict(int)

    for row in data:
        provider = row["Provider Type"]
        error    = row["Error Code"]
        if not error or error.strip() in ("-", ""):
            continue
        provider_patterns[provider][error] += 1
        provider_totals[provider]          += 1

    providers   = list(provider_patterns.keys())
    N_providers = len(providers)
    all_errors  = set(e for p in provider_patterns.values() for e in p)

    error_doc_freq = defaultdict(int)
    for provider, errors in provider_patterns.items():
        for error in errors:
            error_doc_freq[error] += 1

    provider_profiles = {}
    for provider in providers:
        total, profile = provider_totals[provider], {}
        for error, count in provider_patterns[provider].items():
            tf = count / total
            if use_tfidf:
                idf    = math.log(N_providers / error_doc_freq[error])
                weight = tf * idf
            else:
                weight = tf
            if weight > 0:
                profile[error] = weight
        provider_profiles[provider] = profile

    return (
        providers,
        dict(provider_patterns),
        dict(provider_totals),
        provider_profiles,
        all_errors,
        dict(error_doc_freq),
    )


def cosine_similarity(vec1, vec2):
    keys = set(vec1) | set(vec2)
    dot  = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in keys)
    mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v**2 for v in vec2.values()))
    return 0.0 if mag1 == 0 or mag2 == 0 else dot / (mag1 * mag2)


def build_graph(providers, profiles, provider_totals, threshold, top_k):
    G = nx.Graph()
    for p in providers:
        G.add_node(p, total_denials=provider_totals[p])

    candidate_edges = []
    for i in range(len(providers)):
        for j in range(i + 1, len(providers)):
            p1, p2 = providers[i], providers[j]
            sim = cosine_similarity(profiles[p1], profiles[p2])
            if sim >= threshold:
                candidate_edges.append((p1, p2, sim))

    node_edge_count = defaultdict(int)
    for p1, p2, sim in sorted(candidate_edges, key=lambda x: -x[2]):
        if node_edge_count[p1] < top_k and node_edge_count[p2] < top_k:
            G.add_edge(p1, p2, weight=sim)
            node_edge_count[p1] += 1
            node_edge_count[p2] += 1
    return G


# ─────────────────────────────────────────────
# PLOTLY FIGURE
# ─────────────────────────────────────────────
def build_plotly_figure(
    G_main, node_community, communities_list,
    label_min_denials, threshold, top_k
):
    pos = nx.kamada_kawai_layout(G_main, weight="weight")

    edges        = list(G_main.edges(data=True))
    edge_weights = [d["weight"] for _, _, d in edges]
    max_w        = max(edge_weights) if edge_weights else 1

    # ── Edge lines ─────────────────────────────
    edge_traces = []
    for (u, v, d) in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w       = d["weight"]
        alpha   = round(0.25 + 0.55 * (w / max_w), 2)
        width   = 1.0 + 3.5 * (w / max_w)
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color=f"rgba(168,218,220,{alpha})"),
            hoverinfo="none",
            showlegend=False,
        ))

    # ── Invisible midpoint markers for edge hover ──
    mid_x, mid_y, mid_hover = [], [], []
    for (u, v, d) in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        mid_hover.append(
            f"<b>{u}</b><br>"
            f"<b>{v}</b><br>"
            f"Similarity: <b>{d['weight']:.4f}</b>"
        )

    edge_hover_trace = go.Scatter(
        x=mid_x,
        y=mid_y,
        mode="markers",
        marker=dict(size=14, color="rgba(0,0,0,0)"),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=mid_hover,
        showlegend=False,
    )

    # ── Nodes ──────────────────────────────────
    node_x, node_y = [], []
    node_colors, node_sizes = [], []
    node_labels, node_hover = [], []

    for node in G_main.nodes():
        x, y    = pos[node]
        denials = G_main.nodes[node]["total_denials"]
        color   = PALETTE[node_community.get(node, 0) % len(PALETTE)]
        size    = max(8, min(40, denials * 0.04))
        label   = (
            (node[:20] + "...") if len(node) > 20 else node
        ) if denials >= label_min_denials else ""

        node_x.append(x)
        node_y.append(y)
        node_colors.append(color)
        node_sizes.append(size)
        node_labels.append(label)
        node_hover.append(
            f"<b>{node}</b><br>"
            f"Denials: {denials:,}<br>"
            f"Cluster: {node_community.get(node, 0) + 1}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color="#ffffff"),
        ),
        text=node_labels,
        textposition="top center",
        textfont=dict(color="white", size=8),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=node_hover,
        showlegend=False,
    )

    # ── Cluster legend entries ─────────────────
    legend_traces = [
        go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=10, color=PALETTE[i % len(PALETTE)]),
            name=f"Cluster {i+1}  ({len(comm)} providers)",
            showlegend=True,
        )
        for i, comm in enumerate(communities_list)
    ]

    # ── Assemble ───────────────────────────────
    fig = go.Figure(
        data=edge_traces + [edge_hover_trace, node_trace] + legend_traces,
        layout=go.Layout(
            paper_bgcolor="#0D1117",
            plot_bgcolor="#0D1117",
            font=dict(color="white"),
            title=dict(
                text=(
                    f"Medicare FFS CERT 2025 — Provider Denial Pattern Network<br>"
                    f"<sup>threshold={threshold}  |  top-{top_k} edges/node  |  "
                    f"{G_main.number_of_nodes()} providers  |  "
                    f"hover edges for similarity scores</sup>"
                ),
                font=dict(color="white", size=14),
                x=0.01,
            ),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            hovermode="closest",
            hoverlabel=dict(
                bgcolor="#161B22",
                bordercolor="#30363D",
                font=dict(color="white", size=12),
            ),
            legend=dict(
                bgcolor="#161B22",
                bordercolor="#30363D",
                borderwidth=1,
                font=dict(color="white", size=9),
                itemsizing="constant",
                x=1.01,
            ),
            margin=dict(l=10, r=160, t=80, b=10),
            height=720,
            dragmode="zoom",
        )
    )

    fig.update_layout(
    dragmode="zoom",
    modebar=dict(
        bgcolor="#161B22",
        color="#8b949e",
        activecolor="#58a6ff",
    )
)

    return fig


# ─────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────
st.markdown("# Medicare FFS CERT — Denial Pattern Network")
st.markdown(
    "<p style='color:#8b949e;font-size:0.88rem;margin-top:-8px;margin-bottom:24px'>"
    "Providers connected by similarity of error-code denial patterns &nbsp;|&nbsp; "
    "TF-IDF + proportional normalisation &nbsp;|&nbsp; "
    "Scroll to zoom &nbsp;·&nbsp; drag to pan &nbsp;·&nbsp; hover edges for weights"
    "</p>",
    unsafe_allow_html=True,
)

with st.spinner("Loading and deduplicating..."):
    try:
        raw_rows, data = load_and_deduplicate(FILE_PATH)
    except FileNotFoundError:
        st.error(f"File not found:\n`{FILE_PATH}`\n\nUpdate FILE_PATH at the top of the script.")
        st.stop()

with st.spinner("Building normalised provider profiles..."):
    providers, provider_patterns, provider_totals, profiles, all_errors, error_doc_freq = \
        build_profiles(data, use_tfidf)

with st.spinner("Computing similarities and detecting clusters..."):
    G = build_graph(providers, profiles, provider_totals, similarity_threshold, top_k_edges)
    components       = sorted(nx.connected_components(G), key=len, reverse=True)
    G_main           = G.copy()
    communities_list = sorted(
        community.greedy_modularity_communities(G_main), key=len, reverse=True
    )
    node_community = {
        node: i for i, comm in enumerate(communities_list) for node in comm
    }

# Metrics
c1, c2, c3, c4, c5 = st.columns(5)
for col, val, label in [
    (c1, len(raw_rows),            "Raw rows"),
    (c2, len(data),                "After dedup"),
    (c3, len(providers),           "Providers"),
    (c4, G_main.number_of_edges(), "Edges drawn"),
    (c5, len(communities_list),    "Clusters"),
]:
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="value">{val}</div>'
        f'<div class="label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# Graph + cluster panel
graph_col, cluster_col = st.columns([3, 1])

with graph_col:
    with st.spinner("Rendering network..."):
        fig = build_plotly_figure(
            G_main, node_community, communities_list,
            label_min_denials, similarity_threshold, top_k_edges
        )
        st.plotly_chart(fig, use_container_width=True, config={
            "scrollZoom": True,
            "displayModeBar": True,
        })
        
with cluster_col:
    st.markdown("### Clusters")
    st.caption("Expand a cluster to see its providers and top error codes.")
    for i, comm in enumerate(communities_list):
        color = PALETTE[i % len(PALETTE)]
        with st.expander(f"Cluster {i+1}  .  {len(comm)} providers", expanded=(i == 0)):
            cluster_errors = defaultdict(int)
            for provider in comm:
                for error, count in provider_patterns.get(provider, {}).items():
                    cluster_errors[error] += count
            top_errors = sorted(cluster_errors.items(), key=lambda x: -x[1])[:5]

            st.markdown(
                "<div style='font-size:0.74rem;color:#8b949e;margin-bottom:6px'>"
                "Top error codes</div>",
                unsafe_allow_html=True,
            )
            for error, count in top_errors:
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"font-size:0.78rem;font-family:monospace;"
                    f"padding:3px 0;border-bottom:1px solid #1e2330'>"
                    f"<span style='color:{color}'>{error}</span>"
                    f"<span style='color:#8b949e'>{count:,}</span></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:0.74rem;color:#8b949e;margin-bottom:4px'>"
                "Providers</div>",
                unsafe_allow_html=True,
            )
            for provider in sorted(comm):
                denials = provider_totals.get(provider, 0)
                st.markdown(
                    f"<div style='font-size:0.74rem;padding:2px 0;"
                    f"border-bottom:1px solid #1e2330;color:#c9d1d9'>"
                    f"{provider} "
                    f"<span style='color:#484f58'>({denials:,} denials)</span></div>",
                    unsafe_allow_html=True,
                )

# Error code table
st.markdown("---")
st.markdown("### Error code breakdown")
rows = []
for error in sorted(all_errors):
    total_count    = sum(provider_patterns[p].get(error, 0) for p in providers)
    provider_count = error_doc_freq.get(error, 0)
    rows.append({
        "Error code":         error,
        "Total denials":      total_count,
        "Providers affected": provider_count,
        "% of providers":     f"{100 * provider_count / len(providers):.1f}%",
    })
df = pd.DataFrame(rows).sort_values("Total denials", ascending=False)
st.dataframe(df, use_container_width=True, hide_index=True)
