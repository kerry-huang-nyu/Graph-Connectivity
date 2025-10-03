# pip install dash dash-cytoscape networkx
import dash
from dash import html, dcc, Input, Output
import dash_cytoscape as cyto
import networkx as nx

# --- Build a base graph and a sequence of snapshots (timesteps) ---
G0 = nx.Graph()
G0.add_nodes_from(["A", "B", "C", "D"])
G0.add_edges_from([("A","B"), ("B","C")])

# Make a few edits over time:
G1 = G0.copy(); G1.add_edge("C","D")              # add edge
G2 = G1.copy(); G2.remove_edge("A","B")          # remove edge
G3 = G2.copy(); G3.add_edge("A","D"); G3.add_edge("A","C")

SNAPSHOTS = [G0, G1, G2, G3]

def to_elements(G: nx.Graph):
    nodes = [{"data": {"id": str(n), "label": str(n)}} for n in G.nodes()]
    edges = [{"data": {"source": str(u), "target": str(v)}} for u, v in G.edges()]
    return nodes + edges

app = dash.Dash(__name__)
app.layout = html.Div([
    cyto.Cytoscape(
        id="cy",
        elements=to_elements(SNAPSHOTS[0]),
        layout={"name": "cose"},
        style={"width": "100%", "height": "600px"},
        stylesheet=[{"selector": "node", "style": {"label": "data(label)"}}],
    ),
    html.Div(style={"marginTop":"12px"}, children=[
        dcc.Slider(id="step", min=0, max=len(SNAPSHOTS)-1, step=1, value=0,
                   marks={i: str(i) for i in range(len(SNAPSHOTS))}),
        html.Div(id="info", style={"marginTop":"6px"})
    ])
])

@app.callback(
    Output("cy", "elements"),
    Output("info", "children"),
    Input("step", "value"),
)
def update_elements(k):
    G = SNAPSHOTS[k]
    # Example: run an algorithm each step (e.g., connected components)
    comps = list(nx.connected_components(G))
    info = f"Step {k}: {len(comps)} connected component(s): " + \
           ", ".join("{" + ", ".join(sorted(c)) + "}" for c in comps)
    return to_elements(G), info

if __name__ == "__main__":
    app.run(debug=True)