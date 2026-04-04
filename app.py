import streamlit as st
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random
import math
import warnings
warnings.filterwarnings('ignore')

# Core states: 0=Susceptible, 1=Infected(Believers), 2=Recovered, 3=Skeptics, 4=Fact-checkers
STATES = {0: '#00e5ff', 1: '#ff3366', 2: '#00ff88', 3: '#b44fff', 4: '#ff8c00'}
LABELS = {0: 'Susceptible', 1: 'Believers', 2: 'Recovered', 3: 'Skeptics', 4: 'Fact-checkers'}

@st.cache_data(show_spinner=False)
def build_network(N, m, seed=42):
    """Generate Barabasi-Albert scale-free network."""
    G = nx.barabasi_albert_graph(N, m, seed=seed)
    return G

@st.cache_data(show_spinner=False)
def compute_centrality(N, m, seed=42):
    G = build_network(N, m, seed)
    deg = dict(G.degree)
    bet = nx.betweenness_centrality(G, normalized=True, seed=seed)
    return deg, bet

def assign_roles(G, skeptic_frac, fc_frac, initial_infected_frac, seed=42):
    """Assign initial roles to nodes."""
    rng = np.random.default_rng(seed)
    N = G.number_of_nodes()
    nodes = list(G.nodes)
    rng.shuffle(nodes)
    n_sk = int(N * skeptic_frac)
    n_fc = int(N * fc_frac)
    ni = max(1, int(N * initial_infected_frac))
    
    states = {n: 0 for n in G.nodes}
    for n in nodes[:n_sk]: states[n] = 3
    for n in nodes[n_sk:n_sk+n_fc]: states[n] = 4
    
    susceptible = [n for n in G.nodes if states[n] == 0]
    rng.shuffle(susceptible)
    for n in susceptible[:ni]: states[n] = 1
    return states

def sir_step(G, states, beta, gamma, fc_gamma_mult, skeptic_beta_mult):
    """Single SIR simulation step."""
    new_states = states.copy()
    for node in G.nodes:
        s = states[node]
        if s == 1:
            for nb in G.neighbors(node):
                nbs = states[nb]
                if nbs in [0, 3]:
                    eff_beta = beta * skeptic_beta_mult if nbs == 3 else beta
                    if np.random.random() < eff_beta:
                        new_states[nb] = 1
            g = gamma * fc_gamma_mult if any(states[nb] == 4 for nb in G.neighbors(node)) else gamma
            if np.random.random() < g:
                new_states[node] = 2
    return new_states

def run_simulation(G, states_init, beta, gamma, T, fc_gamma_mult, skeptic_beta_mult, remove_hubs_pct=0.0, seed=42):
    """Run full SIR simulation."""
    np.random.seed(seed)
    Gw = G.copy()
    states = states_init.copy()
    
    if remove_hubs_pct > 0.0:
        degrees = dict(Gw.degree)
        threshold = np.percentile(list(degrees.values()), 100 - remove_hubs_pct*100)
        hubs = [n for n, d in degrees.items() if d >= threshold]
        Gw.remove_nodes_from(hubs)
        for h in hubs: states.pop(h, None)
    
    history = []
    N = Gw.number_of_nodes()
    if N == 0: return pd.DataFrame(), {}
    
    for t in range(T):
        counts = {0:0, 1:0, 2:0, 3:0, 4:0}
        for s in states.values(): counts[s] += 1
        history.append({'t': t, 'S': counts[0]/N*100, 'I': counts[1]/N*100, 'R': counts[2]/N*100, 
                       'SK': counts[3]/N*100, 'FC': counts[4]/N*100})
        states = sir_step(Gw, states, beta, gamma, fc_gamma_mult, skeptic_beta_mult)
    
    df = pd.DataFrame(history)
    peak_inf = df['I'].max()
    final_rec = df['R'].iloc[-1]
    avg_deg = 2 * Gw.number_of_edges() / N if N > 0 else 0
    r0 = beta * avg_deg / gamma if gamma > 0 else float('inf')
    
    metrics = {'peak_infection': peak_inf, 'final_recovered': final_rec, 'r0': r0, 'nodes_in_sim': N}
    return df, metrics

def make_time_series_fig(df_base, df_int=None, title="SIR Dynamics"):
    fig = go.Figure()
    state_cols = ['S', 'I', 'R', 'SK', 'FC']
    state_ids = [0,1,2,3,4]
    
    for i, col in enumerate(state_cols):
        fig.add_trace(go.Scatter(x=df_base['t'], y=df_base[col], name=LABELS[state_ids[i]],
                                 line=dict(color=STATES[state_ids[i]], width=2.5)))
        if df_int is not None:
            fig.add_trace(go.Scatter(x=df_int['t'], y=df_int[col], name=f"{LABELS[state_ids[i]]} (Int)",
                                     line=dict(color=STATES[state_ids[i]], width=2, dash='dash')))
    
    fig.update_layout(title=title, xaxis_title="Time Step", yaxis_title="% of Network",
                      height=400)
    return fig

def make_network_fig(G, states, title="Network State"):
    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#555'), 
                           hoverinfo='none', mode='lines')
    node_traces = []
    
    for state in [0,1,2,3,4]:
        x_nodes, y_nodes = [], []
        for node in G.nodes():
            if states.get(node, 0) == state:
                x_nodes.append(pos[node][0])
                y_nodes.append(pos[node][1])
        if x_nodes:
            node_traces.append(go.Scatter(x=x_nodes, y=y_nodes, mode='markers',
                                          marker=dict(size=10, color=STATES[state]),
                                          name=LABELS[state]))
    
    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(title=title, showlegend=True,
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      height=500)
    return fig

# Simple UI
st.title("Fake News Spread Simulator")
st.write("SIR model on scale-free networks with Skeptics & Fact-checkers")

with st.sidebar:
    st.header("Parameters")
    N = st.slider("Network Size", 100, 500, 200)
    m = st.slider("BA m", 2, 8, 3)
    beta = st.slider("β (Infection)", 0.01, 0.2, 0.08, 0.01)
    gamma = st.slider("γ (Recovery)", 0.01, 0.1, 0.03, 0.01)
    init_frac = st.slider("Initial Infected %", 0.5, 5, 1)/100
    T = st.slider("Steps", 50, 150, 100)
    skeptic_frac = st.slider("Skeptics %", 0.0, 0.2, 0.1)
    fc_frac = st.slider("Fact-checkers %", 0.0, 0.15, 0.05)
    fc_boost = st.checkbox("Boost FC")
    fc_mult = st.slider("FC Mult", 2.0, 5.0, 3.0)
    remove_hubs = st.checkbox("Remove Hubs")
    hub_pct = st.slider("Hub %", 5, 20, 10)

if st.button("Run"):
    with st.spinner("Simulating..."):
        G = build_network(N, m)
        states = assign_roles(G, skeptic_frac, fc_frac, init_frac)
        
        df_base, m_base = run_simulation(G, states, beta, gamma, T, 1.0, 0.25)
        fc_g = fc_mult if fc_boost else 1.0
        hub_p = hub_pct/100 if remove_hubs else 0
        df_int, m_int = run_simulation(G, states, beta, gamma, T, fc_g, 0.25, hub_p)
        
        st.session_state.update({
            'df_base': df_base, 'df_int': df_int,
            'm_base': m_base, 'm_int': m_int,
            'G': G, 'states': states
        })
        st.success("Complete!")

if 'df_base' in st.session_state:
    tab1, tab2, tab3 = st.tabs(["Dynamics", "Metrics", "Network"])
    
    with tab1:
        fig = make_time_series_fig(st.session_state['df_base'], st.session_state['df_int'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        df_m = pd.DataFrame({
            ' ': ['Peak I %', 'Final R %', 'R0'],
            'Base': [f"{st.session_state['m_base']['peak_infection']:.1f}",
                    f"{st.session_state['m_base']['final_recovered']:.1f}",
                    f"{st.session_state['m_base']['r0']:.2f}"],
            'Int': [f"{st.session_state['m_int']['peak_infection']:.1f}",
                   f"{st.session_state['m_int']['final_recovered']:.1f}",
                   f"{st.session_state['m_int']['r0']:.2f}"]
        })
        st.dataframe(df_m)
    
    with tab3:
        fig_net = make_network_fig(st.session_state['G'], st.session_state['states'])
        st.plotly_chart(fig_net, use_container_width=True)

st.markdown("**Exam Notes:** Discrete SIR on BA scale-free net. Skeptics: β×0.25. FC boost neighbor recovery. Interventions cut peak spread. R0=β×⟨k⟩/γ.")
