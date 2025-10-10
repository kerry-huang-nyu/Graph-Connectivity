# pip install dash dash-cytoscape networkx
import dash
from dash import html, dcc, Input, Output
import dash_cytoscape as cyto
import networkx as nx
from collections import Counter 

import sys 
sys.path.append("..")
from graph import Graph, Edge, GraphAlgorithm, SimpleRingAlgorithm, step_algorithm

# ---- Helpers to convert to/from NX/Dash ----
def graph_to_nx_multigraph(graph: Graph) -> nx.MultiGraph:
    G = nx.MultiGraph()
    G.add_nodes_from([str(i) for i in range(graph.n)])
    seen = Counter()
    for e in graph.edges:
        n1, n2, p = e.node1, e.node2, e.prob
        if n1 > n2:
            n1, n2 = n2, n1
        seen[(n1, n2)] += 1
        key = f"{n1}-{n2}-{seen[(n1, n2)]}"
        G.add_edge(str(n1), str(n2), key=key, label=str(p))
    return G

def build_snapshots(algo: GraphAlgorithm, graph: Graph):
    snaps = [graph_to_nx_multigraph(graph)]
    for _ in step_algorithm(algo, graph):
        snaps.append(graph_to_nx_multigraph(graph))
    return snaps

def get_fixed_positions(first_graph: nx.MultiGraph):
    _raw = nx.circular_layout(first_graph)
    SCALE = 250
    return {n: {"x": float(x) * SCALE, "y": float(y) * SCALE} for n, (x, y) in _raw.items()}

def serialize_graph(G: nx.MultiGraph):
    return {
        "nodes": [str(n) for n in G.nodes()],
        "edges": [
            {"u": str(u), "v": str(v), "k": str(k), "label": str(data.get("label", k))}
            for u, v, k, data in G.edges(keys=True, data=True)
        ],
    }

def serialize_snapshots(snaps: list[nx.MultiGraph]):
    return [serialize_graph(G) for G in snaps]

def to_elements_from_serialized(serial_G: dict, pos: dict):
    nodes = [{"data": {"id": nid, "label": nid}, "position": pos[nid]} for nid in serial_G["nodes"]]
    edges = [{"data": {"id": f'{e["u"]}-{e["v"]}-{e["k"]}',
                       "source": e["u"], "target": e["v"], "label": e["label"]}} for e in serial_G["edges"]]
    return nodes + edges

# superseded by graph_to_nx_multigraph(graph)
# def graph_conversion(graph: Graph) -> nx.MultiGraph: 
#     #get the edge and the number of vertices 
#     G = nx.MultiGraph() 
#     G.add_nodes_from([str(i) for i in graph.n]) 
# 
#     seen = Counter() 
#     for edge in graph.edges: 
#         n1, n2, p = edge
#         if n1 > n2: #swap so the first element is smaller 
#             n1, n2 = n2, n1 
#         seen[(n1, n2)] += 1 #add another edge between the two nodes 
# 
#         #the key is set as the count of the edge between the 2 endpoints 
#         G.add_edge(str(n1), str(n2), key=f'{n1}-{n2}-{seen[(n1,n2)]}', label=f'{p}')
#     return G
# 
# superseded by build_snapshots(graph) above
# def build_snapshots(graph: Graph): 
#     snapshots = [graph_conversion(graph)]
#     for step in step_algorithm(graph): 
#         snapshots.append(graph_conversion(graph))
#     return snapshots
# 
# superseded by get_fixed_positions(first_graph) above
# def get_fixed_positions(snapshots: list[nx.MultiGraph]):
#     _raw_pos = nx.circular_layout(snapshots[0])  # dict: node -> (x, y) in [-1, 1]
#     SCALE = 250
#     POS = {n: {"x": float(x) * SCALE, "y": float(y) * SCALE}
#         for n, (x, y) in _raw_pos.items()}
#     return POS 
# 
# superseded by to_elements_from_serialized(serial_G, pos)
# def to_elements(G: nx.MultiGraph):
#     # Nodes with fixed positions (preset layout)
#     nodes = [
#         {"data": {"id": str(n), "label": str(n)}, "position": POS[str(n)]}
#         for n in G.nodes()
#     ]
# 
#     # Edges: include a unique 'id' per parallel edge (use the MultiGraph key)
#     edges = []
#     for u, v, k, data in G.edges(keys=True, data=True):
#         edge_label = data.get("label", k)
#         edges.append({
#             "data": {
#                 "id": f"{u}-{v}-{k}",
#                 "source": str(u),
#                 "target": str(v),
#                 "label": str(edge_label),
#             }
#         })
#     return nodes + edges

app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Store(id="snapshots-store"),
    dcc.Store(id="pos-store"),
    html.Button("Simulate", id="simulate-btn", n_clicks=0),
    cyto.Cytoscape(
        id="cy",
        elements=[],
        layout={"name": "preset"},
        autoRefreshLayout=True,
        style={"width": "100%", "height": "600px"},
        autoungrabify=True,
        responsive=True,
        zoomingEnabled=True,
        userZoomingEnabled=True,
        minZoom=0.8,
        maxZoom=1,
        stylesheet=[
            {"selector": "node", "style": {
                "label": "data(label)",
                "background-color": "#000000",
            }},
            {"selector": "edge[label = '1']", "style": {
                "curve-style": "bezier",
                "control-point-step-size": 40,
                "line-opacity": 0.9,
                "label": "data(label)",
                "text-rotation": "autorotate",
                "text-margin-y": -8,
                "font-size": 12,
                "text-background-color": "#ffffff",
                "text-background-opacity": 0.75,
                "text-background-shape": "roundrectangle",
                "text-wrap": "wrap",
                "line-style": "solid",
                "line-color": "#000000"
            }},
            {"selector": "edge[label > 0][label < 1]", "style": {
                "curve-style": "bezier",
                "control-point-step-size": 40,
                "line-opacity": 0.9,
                "label": "data(label)",
                "text-rotation": "autorotate",
                "text-margin-y": -8,
                "font-size": 12,
                "text-background-color": "#ffffff",
                "text-background-opacity": 0.75,
                "text-background-shape": "roundrectangle",
                "text-wrap": "wrap",
                "line-style": "dotted",
                "line-dash-pattern": [10, 100],
                "line-color": "#555555"
            }},
            {"selector": "edge[label = '0']", "style": {
                "curve-style": "bezier",
                "control-point-step-size": 40,
                "line-opacity": 0.3,
                "label": "data(label)",
                "text-rotation": "autorotate",
                "text-margin-y": -8,
                "font-size": 12,
                "text-background-color": "#ffffff",
                "text-background-opacity": 0.75,
                "text-background-shape": "roundrectangle",
                "text-wrap": "wrap",
                "line-style": "solid",
                "line-color": "#cccccc"
            }},
        ],
    ),
    html.Div(style={"marginTop": "12px"}, children=[
        dcc.Slider(id="step", min=0, max=0, step=1, value=0, marks={0: "0"}),
        html.Div(id="info", style={"marginTop": "6px"})
    ])
])

@app.callback(
    Output("cy", "elements"),
    Output("info", "children"),
    Input("step", "value"),
    Input("snapshots-store", "data"),
    Input("pos-store", "data"),
)
def update_elements(k, serialized_snaps, pos):
    if not serialized_snaps or not pos:
        return [], "Click Simulate to generate a graph."
    k = int(k or 0)
    k = max(0, min(k, len(serialized_snaps) - 1))
    serial_G = serialized_snaps[k]
    # compute components for info panel
    H = nx.Graph()
    H.add_nodes_from(serial_G["nodes"])
    H.add_edges_from([(e["u"], e["v"]) for e in serial_G["edges"]])
    comps = list(nx.connected_components(H))
    info = (
        f"Step {k}: {len(comps)} connected component(s): " +
        ", ".join("{" + ", ".join(sorted(c)) + "}" for c in comps)
    )
    return to_elements_from_serialized(serial_G, pos), info

@app.callback(
    Output("step", "max"),
    Output("step", "marks"),
    Input("snapshots-store", "data"),
)
def sync_slider(serialized_snaps):
    if not serialized_snaps:
        return 0, {0: "0"}
    n = len(serialized_snaps)
    return n - 1, {i: str(i) for i in range(n)}

@app.callback(
    Output("snapshots-store", "data"),
    Output("pos-store", "data"),
    Input("simulate-btn", "n_clicks"),
    prevent_initial_call=False,
)
def simulate_graph(_):
    # Define your edge list for simulation; adjust as desired
    edge_lst = [
        Edge(0, 1, 0.9),
        Edge(1, 2, 0.2),
        Edge(2, 3, 0.8),
        Edge(3, 4, 0.3),
        Edge(4, 5, 0.5),
        Edge(5, 0, 0.5),
    ]
    g = Graph(edge_lst)
    algo = SimpleRingAlgorithm(len(edge_lst) - 1)
    snaps = build_snapshots(algo, g)               # list[nx.MultiGraph]
    pos = get_fixed_positions(snaps[0])      # fixed positions from first snapshot
    return serialize_snapshots(snaps), pos

if __name__ == "__main__":
    app.run(debug=True)