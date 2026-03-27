"""
🦠 Fake News Spread Simulator
SIR-based misinformation propagation model on scale-free networks.
"""

import streamlit as st
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import random
import math
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Spread Simulator",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root Palette ── */
:root {
    --bg-primary:    #04060f;
    --bg-card:       #0b0f1e;
    --bg-card2:      #0f1527;
    --accent-cyan:   #00e5ff;
    --accent-red:    #ff3366;
    --accent-green:  #00ff88;
    --accent-purple: #b44fff;
    --accent-orange: #ff8c00;
    --text-main:     #e8edf5;
    --text-muted:    #7a8aab;
    --border:        rgba(0,229,255,0.15);
    --glow-cyan:     0 0 20px rgba(0,229,255,0.35);
    --glow-red:      0 0 20px rgba(255,51,102,0.35);
    --glow-green:    0 0 20px rgba(0,255,136,0.35);
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-main) !important;
}

/* ── Streamlit App Background ── */
.stApp {
    background: radial-gradient(ellipse at 20% 20%, rgba(0,229,255,0.04) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(180,79,255,0.04) 0%, transparent 50%),
                var(--bg-primary) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #08091a 0%, #0b0f1e 100%) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-main) !important; }

/* ── Header Block ── */
.hero-block {
    background: linear-gradient(135deg, #08091a 0%, #0f1527 50%, #08091a 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-block::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), var(--accent-purple), transparent);
}
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.4rem; font-weight: 900;
    background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-purple) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.1;
}
.hero-subtitle {
    font-family: 'Orbitron', monospace;
    font-size: 0.9rem; font-weight: 400;
    color: var(--accent-cyan); letter-spacing: 0.15em;
    margin: 0.4rem 0 0.8rem 0; text-transform: uppercase;
}
.hero-desc {
    font-size: 0.95rem; color: var(--text-muted);
    max-width: 650px; line-height: 1.6; margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,229,255,0.1); border: 1px solid rgba(0,229,255,0.3);
    color: var(--accent-cyan); border-radius: 20px;
    padding: 0.2rem 0.8rem; font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace; margin-right: 0.4rem; margin-top: 0.6rem;
}

/* ── Metric Cards ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px; padding: 1.2rem 1.4rem;
    position: relative; overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-card.red  { border-color: rgba(255,51,102,0.3);  box-shadow: var(--glow-red); }
.metric-card.green{ border-color: rgba(0,255,136,0.3);   box-shadow: var(--glow-green); }
.metric-card.cyan { border-color: rgba(0,229,255,0.3);   box-shadow: var(--glow-cyan); }
.metric-card.purple{ border-color: rgba(180,79,255,0.3); box-shadow: 0 0 20px rgba(180,79,255,0.35); }
.metric-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.3rem; }
.metric-value { font-family: 'Orbitron', monospace; font-size: 2rem; font-weight: 700; line-height: 1; }
.metric-sub   { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem; }
.metric-card.red    .metric-label { color: var(--accent-red); }
.metric-card.green  .metric-label { color: var(--accent-green); }
.metric-card.cyan   .metric-label { color: var(--accent-cyan); }
.metric-card.purple .metric-label { color: var(--accent-purple); }

/* ── Section Card ── */
.section-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px; padding: 1.5rem;
    margin-bottom: 1rem;
}
.section-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem; font-weight: 700;
    color: var(--accent-cyan); letter-spacing: 0.12em;
    text-transform: uppercase; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.5rem;
}

/* ── Tabs override ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card2) !important;
    border-bottom: 1px solid var(--border) !important;
    border-radius: 10px 10px 0 0; gap: 0.2rem; padding: 0.3rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important; border-radius: 8px !important;
    padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,229,255,0.1) !important;
    color: var(--accent-cyan) !important;
    box-shadow: 0 0 12px rgba(0,229,255,0.2) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    padding: 1.5rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,229,255,0.15) 0%, rgba(180,79,255,0.15) 100%) !important;
    border: 1px solid var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important; font-size: 0.8rem !important;
    letter-spacing: 0.08em !important; border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,229,255,0.3) 0%, rgba(180,79,255,0.3) 100%) !important;
    box-shadow: var(--glow-cyan) !important; transform: translateY(-1px);
}

/* ── Sliders ── */
.stSlider > div > div > div > div { background: var(--accent-cyan) !important; }

/* ── Sidebar controls ── */
.sidebar-section {
    background: rgba(0,229,255,0.04);
    border: 1px solid rgba(0,229,255,0.1);
    border-radius: 10px; padding: 1rem; margin-bottom: 1rem;
}
.sidebar-header {
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem; font-weight: 700;
    color: var(--accent-cyan); letter-spacing: 0.15em;
    text-transform: uppercase; margin-bottom: 0.8rem;
}

/* ── Info/Alert boxes ── */
.info-box {
    background: rgba(0,229,255,0.06);
    border-left: 3px solid var(--accent-cyan);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem; font-size: 0.85rem; margin: 0.8rem 0;
}
.warn-box {
    background: rgba(255,51,102,0.06);
    border-left: 3px solid var(--accent-red);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem; font-size: 0.85rem; margin: 0.8rem 0;
}
.success-box {
    background: rgba(0,255,136,0.06);
    border-left: 3px solid var(--accent-green);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem; font-size: 0.85rem; margin: 0.8rem 0;
}

/* ── Legend Pills ── */
.legend-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.05);
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.78rem; margin: 3px;
}
.legend-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

/* ── Table ── */
.stDataFrame { border-radius: 10px !important; }

/* ── Footer ── */
.footer {
    margin-top: 3rem; padding: 1.5rem;
    border-top: 1px solid var(--border);
    text-align: center; font-size: 0.78rem; color: var(--text-muted);
}
.footer a { color: var(--accent-cyan); text-decoration: none; }

/* ── Spinner & progress ── */
.stSpinner > div { border-color: var(--accent-cyan) !important; }

/* ── Number Input ── */
.stNumberInput input { background: var(--bg-card2) !important; border-color: var(--border) !important; color: var(--text-main) !important; }

/* ── Selectbox ── */
.stSelectbox > div > div { background: var(--bg-card2) !important; border-color: var(--border) !important; }

/* ── Checkbox ── */
.stCheckbox > label { color: var(--text-main) !important; }
</style>
""", unsafe_allow_html=True)


# ─── CONSTANTS ───────────────────────────────────────────────────────────────
STATE_S   = 0   # Susceptible
STATE_I   = 1   # Infected / Believer
STATE_R   = 2   # Recovered
STATE_SK  = 3   # Skeptic
STATE_FC  = 4   # Fact-checker

COLOR_MAP = {
    STATE_S:  "#00e5ff",
    STATE_I:  "#ff3366",
    STATE_R:  "#00ff88",
    STATE_SK: "#b44fff",
    STATE_FC: "#ff8c00",
}
LABEL_MAP = {
    STATE_S: "Susceptible", STATE_I: "Believers",
    STATE_R: "Recovered",   STATE_SK: "Skeptics", STATE_FC: "Fact-checkers",
}


# ─── SIMULATION ENGINE ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_network(N: int, m: int, seed: int = 42):
    """Generate Barabási–Albert scale-free network."""
    G = nx.barabasi_albert_graph(N, m, seed=seed)
    return G

@st.cache_data(show_spinner=False)
def compute_centrality(N: int, m: int, seed: int = 42):
    G = build_network(N, m, seed)
    deg = dict(G.degree())
    bet = nx.betweenness_centrality(G, normalized=True, seed=seed)
    return deg, bet

def assign_roles(G, skeptic_frac, fc_frac, initial_infected_frac, seed=42):
    rng = np.random.default_rng(seed)
    N   = G.number_of_nodes()
    nodes = list(G.nodes())
    rng.shuffle(nodes)

    n_sk = int(N * skeptic_frac)
    n_fc = int(N * fc_frac)
    n_i  = max(1, int(N * initial_infected_frac))

    states = {n: STATE_S for n in G.nodes()}
    for n in nodes[:n_sk]:
        states[n] = STATE_SK
    for n in nodes[n_sk:n_sk + n_fc]:
        states[n] = STATE_FC

    susceptible = [n for n in G.nodes() if states[n] == STATE_S]
    rng.shuffle(susceptible)
    for n in susceptible[:n_i]:
        states[n] = STATE_I

    return states

def sir_step(G, states, beta, gamma, fc_gamma_mult, skeptic_beta_mult):
    new_states = states.copy()
    for node in G.nodes():
        s = states[node]
        if s == STATE_I:
            # Recovery
            g = gamma * fc_gamma_mult if any(
                states[nb] == STATE_FC for nb in G.neighbors(node)
            ) else gamma
            if np.random.random() < g:
                new_states[node] = STATE_R
            else:
                # Try to infect neighbours
                for nb in G.neighbors(node):
                    nb_s = states[nb]
                    if nb_s in (STATE_S, STATE_SK):
                        eff_beta = beta * (skeptic_beta_mult if nb_s == STATE_SK else 1.0)
                        if np.random.random() < eff_beta:
                            new_states[nb] = STATE_I
        elif s == STATE_FC:
            # Fact-checkers recover infected neighbours quickly
            for nb in G.neighbors(node):
                if states[nb] == STATE_I:
                    if np.random.random() < gamma * fc_gamma_mult:
                        new_states[nb] = STATE_R
    return new_states

def run_simulation(G, states_init, beta, gamma, T,
                   fc_gamma_mult, skeptic_beta_mult,
                   remove_hubs_pct=0.0, seed=42):
    """Run full simulation; optionally remove top hub nodes."""
    np.random.seed(seed)
    Gw = G.copy()
    states = states_init.copy()

    if remove_hubs_pct > 0.0:
        degrees = dict(Gw.degree())
        threshold = np.percentile(list(degrees.values()),
                                  100 - remove_hubs_pct * 100)
        hubs = [n for n, d in degrees.items() if d >= threshold]
        Gw.remove_nodes_from(hubs)
        # Remove hub nodes from states
        for h in hubs:
            states.pop(h, None)

    history = []
    N = Gw.number_of_nodes()
    if N == 0:
        return pd.DataFrame(), {}

    for t in range(T):
        counts = {STATE_S: 0, STATE_I: 0, STATE_R: 0, STATE_SK: 0, STATE_FC: 0}
        for s in states.values():
            counts[s] += 1
        history.append({
            "t": t,
            "S":  counts[STATE_S]  / N * 100,
            "I":  counts[STATE_I]  / N * 100,
            "R":  counts[STATE_R]  / N * 100,
            "SK": counts[STATE_SK] / N * 100,
            "FC": counts[STATE_FC] / N * 100,
        })
        if counts[STATE_I] == 0:
            # Fill remaining steps
            for tt in range(t + 1, T):
                history.append({**history[-1], "t": tt})
            break
        states = sir_step(Gw, states, beta, gamma, fc_gamma_mult, skeptic_beta_mult)

    df = pd.DataFrame(history)
    peak_inf  = df["I"].max()
    final_rec = df["R"].iloc[-1]
    # Basic R0 estimate: beta * avg_degree / gamma
    avg_deg = 2 * Gw.number_of_edges() / N if N > 0 else 0
    r0 = (beta * avg_deg) / gamma if gamma > 0 else float("inf")

    metrics = {
        "peak_infection":  round(peak_inf, 2),
        "final_recovered": round(final_rec, 2),
        "r0":              round(r0, 2),
        "nodes_in_sim":    N,
    }
    return df, metrics


# ─── PLOT HELPERS ─────────────────────────────────────────────────────────────

PLOT_COLORS = {
    "S":  "#00e5ff", "I": "#ff3366",
    "R":  "#00ff88", "SK": "#b44fff", "FC": "#ff8c00",
}
PLOT_LABELS = {
    "S": "Susceptible", "I": "Believers / Infected",
    "R": "Recovered",   "SK": "Skeptics", "FC": "Fact-checkers",
}

def make_timeseries_fig(df_base, df_int=None, title="SIR Dynamics"):
    fig = go.Figure()
    dashes = {"base": "solid", "int": "dash"}

    for col in ["S", "I", "R", "SK", "FC"]:
        fig.add_trace(go.Scatter(
            x=df_base["t"], y=df_base[col],
            name=PLOT_LABELS[col] + (" (baseline)" if df_int is not None else ""),
            mode="lines",
            line=dict(color=PLOT_COLORS[col], width=2.5),
            fill="tozeroy" if col == "I" else "none",
            fillcolor="rgba(255,51,102,0.06)" if col == "I" else "rgba(0,0,0,0)",
        ))

    if df_int is not None:
        for col in ["S", "I", "R"]:
            fig.add_trace(go.Scatter(
                x=df_int["t"], y=df_int[col],
                name=PLOT_LABELS[col] + " (intervened)",
                mode="lines",
                line=dict(color=PLOT_COLORS[col], width=2, dash="dash"),
            ))

    fig.update_layout(
        title=dict(text=title, font=dict(family="Orbitron", size=14, color="#00e5ff")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,15,30,0.8)",
        font=dict(family="DM Sans", color="#e8edf5"),
        xaxis=dict(title="Time Step", gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title="% of Network", gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.1)"),
        legend=dict(bgcolor="rgba(11,15,30,0.8)", bordercolor="rgba(0,229,255,0.2)", borderwidth=1),
        hovermode="x unified",
        height=420,
        margin=dict(l=50, r=30, t=60, b=50),
    )
    return fig

def make_network_fig(G, states, title="Network State"):
    pos = nx.spring_layout(G, seed=42, k=1.5 / math.sqrt(G.number_of_nodes()))

    # Build edge traces
    edge_x, edge_y = [], []
    for u, v in G.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.4, color="rgba(255,255,255,0.12)"),
        hoverinfo="none", name="Edges",
    )

    # Node traces (one per state for legend)
    node_traces = []
    groups = {st: ([], [], []) for st in LABEL_MAP}
    for node in G.nodes():
        if node not in pos:
            continue
        st = states.get(node, STATE_S)
        x, y = pos[node]
        deg  = G.degree(node)
        groups[st][0].append(x)
        groups[st][1].append(y)
        groups[st][2].append(f"Node {node}<br>State: {LABEL_MAP[st]}<br>Degree: {deg}")

    for st, (xs, ys, texts) in groups.items():
        if not xs:
            continue
        node_traces.append(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(
                size=6, color=COLOR_MAP[st],
                line=dict(width=0.5, color="rgba(0,0,0,0.5)"),
                opacity=0.9,
            ),
            text=texts, hoverinfo="text",
            name=LABEL_MAP[st],
        ))

    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        title=dict(text=title, font=dict(family="Orbitron", size=13, color="#00e5ff")),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,9,26,0.95)",
        font=dict(family="DM Sans", color="#e8edf5"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(bgcolor="rgba(11,15,30,0.8)", bordercolor="rgba(0,229,255,0.2)", borderwidth=1),
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig

def make_bar_comparison(metrics_base, metrics_int, labels=("Baseline", "Intervened")):
    cats = ["Peak Infection (%)", "Final Recovered (%)", "R₀"]
    vals_b = [metrics_base["peak_infection"], metrics_base["final_recovered"], metrics_base["r0"]]
    vals_i = [metrics_int["peak_infection"],  metrics_int["final_recovered"],  metrics_int["r0"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(name=labels[0], x=cats, y=vals_b,
                          marker_color="#ff3366", opacity=0.85))
    fig.add_trace(go.Bar(name=labels[1], x=cats, y=vals_i,
                          marker_color="#00ff88", opacity=0.85))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,15,30,0.8)",
        font=dict(family="DM Sans", color="#e8edf5"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        legend=dict(bgcolor="rgba(11,15,30,0.8)"),
        height=360,
        margin=dict(l=40, r=20, t=40, b=50),
        title=dict(text="Strategy Comparison", font=dict(family="Orbitron", size=13, color="#00e5ff")),
    )
    return fig

def make_degree_dist_fig(G):
    degrees = sorted([d for _, d in G.degree()], reverse=True)
    unique, counts = np.unique(degrees, return_counts=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=unique, y=counts,
        marker=dict(
            color=counts,
            colorscale=[[0, "#0b0f1e"], [0.5, "#00e5ff"], [1, "#b44fff"]],
            showscale=False,
        ),
        name="Degree count",
    ))
    fig.update_layout(
        title=dict(text="Degree Distribution (Power-Law BA)", font=dict(family="Orbitron", size=12, color="#00e5ff")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,15,30,0.8)",
        font=dict(family="DM Sans", color="#e8edf5"),
        xaxis=dict(title="Degree", gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(title="Count",  gridcolor="rgba(255,255,255,0.07)"),
        height=280, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig

def make_centrality_scatter(deg_dict, bet_dict, states=None):
    nodes = list(deg_dict.keys())
    x = [deg_dict[n] for n in nodes]
    y = [bet_dict[n] for n in nodes]
    cols = [COLOR_MAP.get(states.get(n, STATE_S), "#00e5ff") if states else "#00e5ff" for n in nodes]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=5, color=cols, opacity=0.75,
                    line=dict(width=0.3, color="rgba(0,0,0,0.4)")),
        text=[f"Node {n}<br>Degree: {deg_dict[n]}<br>Betweenness: {bet_dict[n]:.4f}" for n in nodes],
        hoverinfo="text",
    ))
    fig.update_layout(
        title=dict(text="Degree vs Betweenness Centrality", font=dict(family="Orbitron", size=12, color="#00e5ff")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,15,30,0.8)",
        font=dict(family="DM Sans", color="#e8edf5"),
        xaxis=dict(title="Degree", gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(title="Betweenness", gridcolor="rgba(255,255,255,0.07)"),
        height=320, margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="font-family:'Orbitron',monospace;font-size:1.1rem;font-weight:900;
                background:linear-gradient(135deg,#00e5ff,#b44fff);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                margin-bottom:0.3rem;">⚙️ Parameters</div>
    <div style="font-size:0.75rem;color:#7a8aab;margin-bottom:1.2rem;">
    Configure the simulation below</div>
    """, unsafe_allow_html=True)

    # Network
    st.markdown('<div class="sidebar-header">🌐 Network</div>', unsafe_allow_html=True)
    N = st.slider("Network Size (N nodes)", 100, 1000, 300, 50,
                  help="Number of agents in the social network")
    m = st.slider("BA Attachment (m)", 3, 10, 5,
                  help="Each new node connects to m existing nodes (controls density)")
    st.divider()

    # Infection dynamics
    st.markdown('<div class="sidebar-header">🦠 Infection Dynamics</div>', unsafe_allow_html=True)
    beta  = st.slider("Infection Probability (β)", 0.01, 0.30, 0.08, 0.01,
                       help="Probability a Believer infects a Susceptible neighbour per step")
    gamma = st.slider("Recovery Probability (γ)", 0.01, 0.10, 0.03, 0.01,
                       help="Natural recovery probability per step")
    init_infected = st.slider("Initial Infected (%)", 1, 5, 2,
                               help="Percentage of nodes initially infected")
    T = st.slider("Simulation Steps (T)", 50, 200, 100, 10)
    st.divider()

    # Agent roles
    st.markdown('<div class="sidebar-header">👥 Agent Roles</div>', unsafe_allow_html=True)
    skeptic_frac = st.slider("Skeptic Fraction", 0.00, 0.30, 0.10, 0.01,
                              help="Fraction of agents resistant to infection (lower β)")
    fc_frac      = st.slider("Fact-checker Fraction", 0.00, 0.20, 0.05, 0.01,
                              help="Fraction of agents that accelerate recovery of neighbours")
    st.divider()

    # Interventions
    st.markdown('<div class="sidebar-header">🛡️ Interventions</div>', unsafe_allow_html=True)
    fc_boost     = st.checkbox("🚀 Fact-checker Boost", value=False,
                                help="Multiply fact-checker γ by boost multiplier")
    fc_mult      = st.slider("FC Boost Multiplier (×γ)", 2.0, 5.0, 3.0, 0.5,
                              disabled=not fc_boost)
    remove_hubs  = st.checkbox("🎯 Remove High-Centrality Hubs", value=False,
                                help="Simulate removing top-degree spreaders")
    hub_pct      = st.slider("Hub Removal Top-%", 0.05, 0.20, 0.10, 0.01,
                              disabled=not remove_hubs)
    st.divider()

    run_btn = st.button("▶  RUN SIMULATION", use_container_width=True)

    st.markdown("""
    <div style="margin-top:1rem;font-size:0.72rem;color:#7a8aab;line-height:1.6;">
    🔬 <b style="color:#00e5ff;">Model:</b> Discrete-time SIR on BA network<br>
    📐 <b style="color:#b44fff;">Network:</b> Barabási–Albert scale-free<br>
    ⚡ <b style="color:#00ff88;">Cache:</b> Results cached for speed
    </div>
    """, unsafe_allow_html=True)


# ─── HERO HEADER ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-block">
  <div class="hero-subtitle">🦠 Epidemiological Modelling · Social Networks · Misinformation</div>
  <div class="hero-title">Fake News Spread<br>Simulator</div>
  <p class="hero-desc" style="margin-top:0.7rem;">
    Interactive SIR-based model on Barabási–Albert scale-free networks with interventions.
    Model misinformation propagation with <b style="color:#00e5ff;">Believers</b>,
    <b style="color:#b44fff;">Skeptics</b>, and <b style="color:#ff8c00;">Fact-checkers</b>.
    Simulate &amp; Mitigate Fake News Propagation.
  </p>
  <div style="margin-top:0.8rem;">
    <span class="hero-badge">SIR Model</span>
    <span class="hero-badge">Scale-Free Network</span>
    <span class="hero-badge">Agent-Based</span>
    <span class="hero-badge">Real-Time Viz</span>
    <span class="hero-badge">Intervention Analysis</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Legend row
st.markdown("""
<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:1rem;">
  <span class="legend-pill"><span class="legend-dot" style="background:#00e5ff;box-shadow:0 0 6px #00e5ff;"></span>Susceptible</span>
  <span class="legend-pill"><span class="legend-dot" style="background:#ff3366;box-shadow:0 0 6px #ff3366;"></span>Believers</span>
  <span class="legend-pill"><span class="legend-dot" style="background:#00ff88;box-shadow:0 0 6px #00ff88;"></span>Recovered</span>
  <span class="legend-pill"><span class="legend-dot" style="background:#b44fff;box-shadow:0 0 6px #b44fff;"></span>Skeptics</span>
  <span class="legend-pill"><span class="legend-dot" style="background:#ff8c00;box-shadow:0 0 6px #ff8c00;"></span>Fact-checkers</span>
</div>
""", unsafe_allow_html=True)


# ─── SESSION STATE ────────────────────────────────────────────────────────────
if "sim_done"     not in st.session_state: st.session_state.sim_done     = False
if "df_base"      not in st.session_state: st.session_state.df_base      = None
if "df_int"       not in st.session_state: st.session_state.df_int       = None
if "metrics_base" not in st.session_state: st.session_state.metrics_base = {}
if "metrics_int"  not in st.session_state: st.session_state.metrics_int  = {}
if "G"            not in st.session_state: st.session_state.G            = None
if "states_init"  not in st.session_state: st.session_state.states_init  = None


# ─── RUN SIMULATION ──────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("⚙️  Building network & running simulation..."):
        G = build_network(N, m, seed=42)
        states_init = assign_roles(G, skeptic_frac, fc_frac,
                                   init_infected / 100, seed=42)
        fc_gamma_mult = fc_mult if fc_boost else 1.0
        skeptic_beta_mult = 0.25  # Skeptics have 75% lower infection rate

        # Baseline
        df_base, metrics_base = run_simulation(
            G, states_init, beta, gamma, T,
            fc_gamma_mult=1.0, skeptic_beta_mult=skeptic_beta_mult,
            remove_hubs_pct=0.0,
        )

        # Intervened
        df_int, metrics_int = run_simulation(
            G, states_init, beta, gamma, T,
            fc_gamma_mult=fc_gamma_mult,
            skeptic_beta_mult=skeptic_beta_mult,
            remove_hubs_pct=hub_pct if remove_hubs else 0.0,
        )

        st.session_state.update({
            "sim_done": True, "G": G, "states_init": states_init,
            "df_base": df_base, "df_int": df_int,
            "metrics_base": metrics_base, "metrics_int": metrics_int,
            "fc_gamma_mult": fc_gamma_mult,
            "fc_boost": fc_boost, "remove_hubs": remove_hubs,
        })

    st.success("✅ Simulation complete! Explore the tabs below.")


# ─── TABS ────────────────────────────────────────────────────────────────────
tab_sim, tab_res, tab_int, tab_net = st.tabs([
    "📈  Simulation",
    "📊  Results & Metrics",
    "🛡️  Interventions",
    "🕸️  Network Viz",
])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sim:
    if not st.session_state.sim_done:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;">
          <div style="font-size:3rem;margin-bottom:1rem;">🦠</div>
          <div style="font-family:'Orbitron',monospace;font-size:1.2rem;color:#00e5ff;margin-bottom:0.5rem;">
            Ready to Simulate
          </div>
          <div style="color:#7a8aab;font-size:0.9rem;">
            Configure parameters in the sidebar, then click <b style="color:#00e5ff;">▶ RUN SIMULATION</b>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_b = st.session_state.df_base
        df_i = st.session_state.df_int
        mb   = st.session_state.metrics_base
        mi   = st.session_state.metrics_int

        # ── Quick KPI row ──
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card red">
              <div class="metric-label">🔴 Peak Infection</div>
              <div class="metric-value" style="color:#ff3366;">{mb['peak_infection']:.1f}%</div>
              <div class="metric-sub">of network at once</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            delta = mb['peak_infection'] - mi['peak_infection']
            st.markdown(f"""
            <div class="metric-card green">
              <div class="metric-label">📉 Intervention Reduction</div>
              <div class="metric-value" style="color:#00ff88;">{delta:.1f}%</div>
              <div class="metric-sub">peak infection drop</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card purple">
              <div class="metric-label">🧮 R₀ Estimate</div>
              <div class="metric-value" style="color:#b44fff;">{mb['r0']:.2f}</div>
              <div class="metric-sub">basic reproduction number</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card cyan">
              <div class="metric-label">✅ Final Recovered</div>
              <div class="metric-value" style="color:#00e5ff;">{mb['final_recovered']:.1f}%</div>
              <div class="metric-sub">of network (baseline)</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Time-series plot ──
        show_int = st.session_state.fc_boost or st.session_state.remove_hubs
        fig_ts = make_timeseries_fig(
            df_b, df_i if show_int else None,
            title="SIR Dynamics — Baseline" + (" vs Intervened" if show_int else ""),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        if show_int:
            st.markdown("""
            <div class="info-box">
            ℹ️ <b>Dashed lines</b> show the intervened scenario. Comparing solid vs dashed reveals
            how much your selected interventions reduce the infection peak and speed of recovery.
            </div>""", unsafe_allow_html=True)

        # ── Stacked area ──
        with st.expander("📊 Stacked Area View (Baseline)"):
            fig_area = go.Figure()
            for col, label, clr in [
                ("FC", "Fact-checkers", "#ff8c00"),
                ("SK", "Skeptics",      "#b44fff"),
                ("R",  "Recovered",     "#00ff88"),
                ("I",  "Believers",     "#ff3366"),
                ("S",  "Susceptible",   "#00e5ff"),
            ]:
                fig_area.add_trace(go.Scatter(
                    x=df_b["t"], y=df_b[col], name=label,
                    mode="lines", stackgroup="one",
                    line=dict(width=0.5, color=clr),
                    fillcolor=clr.replace("#", "rgba(").replace(")", ",0.7)")
                              if False else clr,
                ))
            fig_area.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(11,15,30,0.8)",
                font=dict(family="DM Sans", color="#e8edf5"),
                xaxis=dict(title="Time Step", gridcolor="rgba(255,255,255,0.07)"),
                yaxis=dict(title="% of Network", gridcolor="rgba(255,255,255,0.07)"),
                height=320, margin=dict(l=40, r=20, t=20, b=40),
                legend=dict(bgcolor="rgba(11,15,30,0.8)"),
            )
            st.plotly_chart(fig_area, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — RESULTS & METRICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_res:
    if not st.session_state.sim_done:
        st.info("Run the simulation first to see results.")
    else:
        mb  = st.session_state.metrics_base
        mi  = st.session_state.metrics_int
        G   = st.session_state.G
        sts = st.session_state.states_init

        # ── Metrics table ──
        st.markdown('<div class="section-title">📋 Simulation Metrics</div>', unsafe_allow_html=True)

        delta_peak = mb["peak_infection"] - mi["peak_infection"]
        delta_rec  = mi["final_recovered"] - mb["final_recovered"]
        r0_label   = "🔴 Epidemic" if mb["r0"] > 1 else "🟢 Contained"

        df_metrics = pd.DataFrame({
            "Metric": ["Peak Infection", "Final Recovered", "R₀ Estimate",
                       "Nodes Simulated", "Network Edges"],
            "Baseline": [
                f"{mb['peak_infection']:.2f}%",
                f"{mb['final_recovered']:.2f}%",
                f"{mb['r0']:.2f} ({r0_label})",
                str(mb["nodes_in_sim"]),
                str(G.number_of_edges()),
            ],
            "Intervened": [
                f"{mi['peak_infection']:.2f}%",
                f"{mi['final_recovered']:.2f}%",
                f"{mi['r0']:.2f}",
                str(mi["nodes_in_sim"]),
                "—",
            ],
            "Δ Change": [
                f"▼ {delta_peak:.2f}%" if delta_peak > 0 else f"▲ {abs(delta_peak):.2f}%",
                f"▲ {delta_rec:.2f}%"  if delta_rec  > 0 else f"▼ {abs(delta_rec):.2f}%",
                "—", "—", "—",
            ],
        })
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

        # ── Centrality analysis ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">🎯 Centrality Analysis</div>', unsafe_allow_html=True)

        col_left, col_right = st.columns(2)
        with col_left:
            deg_dict, bet_dict = compute_centrality(N, m, seed=42)
            fig_cent = make_centrality_scatter(deg_dict, bet_dict, sts)
            st.plotly_chart(fig_cent, use_container_width=True)

        with col_right:
            fig_deg = make_degree_dist_fig(G)
            st.plotly_chart(fig_deg, use_container_width=True)

        # ── Top spreaders table ──
        st.markdown('<div class="section-title">🚨 Top Spreader Nodes</div>', unsafe_allow_html=True)
        top_n = 15
        sorted_deg = sorted(deg_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        df_top = pd.DataFrame([{
            "Node ID": n,
            "Degree": d,
            "Betweenness": f"{bet_dict[n]:.4f}",
            "Initial State": LABEL_MAP.get(sts.get(n, STATE_S), "?"),
            "Spreader Risk": "🔴 HIGH" if d >= np.percentile(list(deg_dict.values()), 85) else
                             "🟡 MED"  if d >= np.percentile(list(deg_dict.values()), 60) else "🟢 LOW",
        } for n, d in sorted_deg])
        st.dataframe(df_top, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="info-box">
        🔬 <b>Scale-free networks</b> follow a power-law degree distribution — a few 
        <b style="color:#ff3366;">hub nodes</b> have disproportionately many connections and 
        dominate information spread. Targeting these hubs is the most effective containment strategy.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — INTERVENTIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_int:
    if not st.session_state.sim_done:
        st.info("Run the simulation first to compare interventions.")
    else:
        mb = st.session_state.metrics_base
        mi = st.session_state.metrics_int
        G  = st.session_state.G
        sts_init = st.session_state.states_init

        # ── Bar comparison ──
        st.markdown('<div class="section-title">⚔️ Baseline vs Intervened Comparison</div>',
                    unsafe_allow_html=True)
        fig_bar = make_bar_comparison(mb, mi)
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Multi-strategy analysis ──
        st.markdown('<div class="section-title">🧪 Multi-Strategy Sweep</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        ℹ️ Running 4 strategies: <b>No intervention</b>, <b>FC Boost ×2</b>, 
        <b>Hub Removal 10%</b>, and <b>Combined</b>. This may take a moment...
        </div>""", unsafe_allow_html=True)

        fc_gamma_mult_base = 1.0
        fc_gamma_mult_int  = st.session_state.get("fc_gamma_mult", 3.0)
        skeptic_beta_mult  = 0.25

        with st.spinner("Running strategy sweep..."):
            strats = [
                ("No Intervention",     fc_gamma_mult_base, 0.00),
                ("FC Boost ×2",         2.0,                0.00),
                ("Hub Removal 10%",     fc_gamma_mult_base, 0.10),
                ("FC Boost + Hub Rem.", 2.0,                0.10),
            ]
            strat_results = []
            for label, fcm, hrp in strats:
                _, met = run_simulation(
                    G, sts_init, beta, gamma, T,
                    fc_gamma_mult=fcm, skeptic_beta_mult=skeptic_beta_mult,
                    remove_hubs_pct=hrp,
                )
                strat_results.append({
                    "Strategy": label,
                    "Peak Infection (%)": met["peak_infection"],
                    "Final Recovered (%)": met["final_recovered"],
                    "R₀": met["r0"],
                })

        df_strats = pd.DataFrame(strat_results)

        # Grouped bar
        fig_strat = go.Figure()
        colors_strat = ["#ff3366", "#ff8c00", "#00e5ff", "#00ff88"]
        for i, col in enumerate(["Peak Infection (%)", "Final Recovered (%)"]):
            for j, row in df_strats.iterrows():
                if i == 0:
                    fig_strat.add_trace(go.Bar(
                        name=row["Strategy"],
                        x=[col], y=[row[col]],
                        marker_color=colors_strat[j], opacity=0.85,
                        legendgroup=row["Strategy"],
                    ))
                else:
                    fig_strat.add_trace(go.Bar(
                        name=row["Strategy"],
                        x=[col], y=[row[col]],
                        marker_color=colors_strat[j], opacity=0.85,
                        legendgroup=row["Strategy"], showlegend=False,
                    ))

        fig_strat.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(11,15,30,0.8)",
            font=dict(family="DM Sans", color="#e8edf5"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
            yaxis=dict(title="%", gridcolor="rgba(255,255,255,0.07)"),
            legend=dict(bgcolor="rgba(11,15,30,0.8)"),
            height=380, margin=dict(l=40, r=20, t=30, b=40),
            title=dict(text="All Strategies: Peak Infection & Final Recovered",
                       font=dict(family="Orbitron", size=12, color="#00e5ff")),
        )
        st.plotly_chart(fig_strat, use_container_width=True)
        st.dataframe(df_strats.style.format({
            "Peak Infection (%)":  "{:.2f}",
            "Final Recovered (%)": "{:.2f}",
            "R₀":                  "{:.2f}",
        }), use_container_width=True, hide_index=True)

        # ── Research insights ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">📚 Research Insights</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div class="section-card">
            <b style="color:#00e5ff;">🎯 Hub Targeting</b><br><br>
            Removing or quarantining the top 10% highest-degree nodes in scale-free networks
            can reduce peak infection by <b style="color:#00ff88;">30–50%</b>, because these 
            hubs serve as critical amplifiers in information cascades.<br><br>
            <i style="color:#7a8aab;">Ref: Pastor-Satorras & Vespignani (2001); 
            arXiv:2401.11524</i>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown("""
            <div class="section-card">
            <b style="color:#ff8c00;">🚀 Fact-checker Networks</b><br><br>
            Increasing the fact-checker recovery multiplier (γ×2–5) simulates 
            coordinated debunking campaigns. Combined with hub removal, this strategy 
            achieves the fastest epidemic decline and lowest final infection rate.<br><br>
            <i style="color:#7a8aab;">Ref: Vosoughi et al., Science 2018; 
            Nature Misinformation Review 2024</i>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="section-card" style="margin-top:0.5rem;">
        <b style="color:#b44fff;">📐 R₀ Interpretation</b><br><br>
        <b>R₀ > 1</b> → <span style="color:#ff3366;">Epidemic spreads</span> &nbsp;|&nbsp;
        <b>R₀ = 1</b> → <span style="color:#ff8c00;">Endemic equilibrium</span> &nbsp;|&nbsp;
        <b>R₀ < 1</b> → <span style="color:#00ff88;">Epidemic dies out</span><br><br>
        In this model: R₀ ≈ β × ⟨k⟩ / γ, where ⟨k⟩ is the average node degree.
        On scale-free networks the effective threshold is much lower, making
        herd immunity harder to achieve without targeted interventions.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — NETWORK VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_net:
    if not st.session_state.sim_done:
        st.info("Run the simulation first to visualize the network.")
    else:
        G   = st.session_state.G
        sts = st.session_state.states_init
        df_b = st.session_state.df_base

        st.markdown("""
        <div class="info-box">
        🕸️ Each node is coloured by its <b>initial role</b>. Hover over nodes to inspect degree 
        and state. For large networks (N > 400) a sample sub-graph is shown for readability.
        </div>""", unsafe_allow_html=True)

        # Sub-sample for large networks
        Gv = G
        sts_v = sts
        if G.number_of_nodes() > 400:
            sample_nodes = random.sample(list(G.nodes()), 400)
            Gv = G.subgraph(sample_nodes).copy()
            sts_v = {n: sts[n] for n in Gv.nodes()}
            st.warning(f"⚠️ Showing 400-node sub-graph (full network: {G.number_of_nodes()} nodes).")

        # Layout selector
        layout_choice = st.selectbox(
            "Layout algorithm",
            ["Spring (Fruchterman-Reingold)", "Kamada-Kawai", "Spectral"],
        )
        if layout_choice.startswith("Spring"):
            pos = nx.spring_layout(Gv, seed=42, k=1.5 / math.sqrt(Gv.number_of_nodes()))
        elif layout_choice.startswith("Kamada"):
            try:
                pos = nx.kamada_kawai_layout(Gv)
            except Exception:
                pos = nx.spring_layout(Gv, seed=42)
        else:
            pos = nx.spectral_layout(Gv)

        # Rebuild net fig with chosen layout
        edge_x, edge_y = [], []
        for u, v in Gv.edges():
            if u in pos and v in pos:
                x0, y0 = pos[u]; x1, y1 = pos[v]
                edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

        fig_net = go.Figure()
        fig_net.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=0.4, color="rgba(255,255,255,0.1)"),
            hoverinfo="none", name="Edges",
        ))

        groups = {st: ([], [], []) for st in LABEL_MAP}
        for node in Gv.nodes():
            if node not in pos: continue
            st_n = sts_v.get(node, STATE_S)
            x, y = pos[node]
            groups[st_n][0].append(x)
            groups[st_n][1].append(y)
            groups[st_n][2].append(f"Node {node}<br>State: {LABEL_MAP[st_n]}<br>Degree: {Gv.degree(node)}")

        # Node size = log degree
        for st_n, (xs, ys, texts) in groups.items():
            if not xs: continue
            node_ids = [n for n in Gv.nodes() if sts_v.get(n, STATE_S) == st_n and n in pos]
            sizes = [max(5, min(18, 3 * math.log1p(Gv.degree(n)))) for n in node_ids]
            fig_net.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(
                    size=sizes, color=COLOR_MAP[st_n], opacity=0.88,
                    line=dict(width=0.5, color="rgba(0,0,0,0.5)"),
                ),
                text=texts, hoverinfo="text", name=LABEL_MAP[st_n],
            ))

        fig_net.update_layout(
            title=dict(text=f"Network Initial State  ({Gv.number_of_nodes()} nodes, {Gv.number_of_edges()} edges)",
                       font=dict(family="Orbitron", size=13, color="#00e5ff")),
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(8,9,26,0.95)",
            font=dict(family="DM Sans", color="#e8edf5"),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(bgcolor="rgba(11,15,30,0.8)", bordercolor="rgba(0,229,255,0.2)", borderwidth=1),
            height=550, margin=dict(l=10, r=10, t=55, b=10),
        )
        st.plotly_chart(fig_net, use_container_width=True)

        # ── Network stats ──
        st.markdown('<div class="section-title">📐 Network Properties</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        avg_deg   = 2 * G.number_of_edges() / G.number_of_nodes()
        max_deg   = max(dict(G.degree()).values())
        try:
            avg_clust = nx.average_clustering(G)
        except Exception:
            avg_clust = 0.0
        try:
            n_comp = nx.number_connected_components(G)
        except Exception:
            n_comp = 1

        for col, label, val, color in [
            (col1, "Avg Degree ⟨k⟩",    f"{avg_deg:.2f}",   "cyan"),
            (col2, "Max Degree",         str(max_deg),       "red"),
            (col3, "Clustering Coeff",   f"{avg_clust:.4f}", "green"),
            (col4, "Connected Components", str(n_comp),      "purple"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card {color}">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value" style="font-size:1.6rem;">{val}</div>
                </div>""", unsafe_allow_html=True)


# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <div style="margin-bottom:0.5rem;">
    <span style="color:#00e5ff;font-family:'Orbitron',monospace;font-weight:700;">🦠 Fake News Spread Simulator</span>
    &nbsp;·&nbsp; SIR-based misinformation model on scale-free networks
  </div>
  <div style="margin-bottom:0.4rem;">
    Inspired by:
    <a href="https://arxiv.org/abs/2401.11524" target="_blank">arXiv:2401.11524</a> ·
    <a href="https://www.nature.com/articles/s41586-018-0101-z" target="_blank">Vosoughi et al., Science 2018</a> ·
    <a href="https://doi.org/10.1103/PhysRevLett.86.3200" target="_blank">Pastor-Satorras & Vespignani, PRL 2001</a> ·
    Nature Misinformation Review 2024
  </div>
  <div style="color:#4a5a7a;">
    Built with Streamlit · NetworkX · Plotly · NumPy · Pandas
    &nbsp;|&nbsp; Model: Discrete-time SIR on Barabási–Albert networks
  </div>
</div>
""", unsafe_allow_html=True)
