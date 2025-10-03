import plotly.graph_objects as go
import plotly.io as pio
import networkx as nx

def frame_from_graph(G, pos):
    x_nodes = [pos[n][0] for n in G.nodes()]
    y_nodes = [pos[n][1] for n in G.nodes()]
    x_edges, y_edges = [], []
    for u, v in G.edges():
        x_edges += [pos[u][0], pos[v][0], None]
        y_edges += [pos[u][1], pos[v][1], None]
    edge_trace = go.Scatter(x=x_edges, y=y_edges, mode="lines")
    node_trace = go.Scatter(x=x_nodes, y=y_nodes, mode="markers+text",
                            text=list(G.nodes()), textposition="top center")
    return [edge_trace, node_trace]


# --- Build a base graph and a sequence of snapshots (timesteps) ---
G0 = nx.Graph()
G0.add_nodes_from(["A", "B", "C", "D"])
G0.add_edges_from([("A","B"), ("B","C")])

# Make a few edits over time:
G1 = G0.copy(); G1.add_edge("C","D")              # add edge
G2 = G1.copy(); G2.remove_edge("A","B")          # remove edge
G3 = G2.copy(); G3.add_edge("A","D"); G3.add_edge("A","C")

# Build snapshots as in Option A
G_list = [G0, G1, G2, G3]  # supply from above or rebuild
pos = nx.spring_layout(G_list[0], seed=42)       # fixed layout

fig = go.Figure(data=frame_from_graph(G_list[0], pos))
fig.frames = [go.Frame(data=frame_from_graph(G, pos), name=str(i))
              for i, G in enumerate(G_list)]

fig.update_layout(
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {"label": "Play", "method": "animate", "args": [None, {"fromcurrent": True}]},
            {"label": "Pause", "method": "animate", "args": [[None], {"mode": "immediate"}]},
        ],
    }],
    sliders=[{
        "steps": [{"method": "animate", "label": str(i),
                   "args": [[str(i)], {"mode": "immediate"}]} for i in range(len(G_list))]
    }],
    showlegend=False, width=800, height=600
)

pio.renderers.default = "browser"
fig.show()