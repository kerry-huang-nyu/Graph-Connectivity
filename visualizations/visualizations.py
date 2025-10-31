# pip install dash dash-cytoscape networkx
import dash
from dash import html, dcc, Input, Output, State
import dash_cytoscape as cyto
import networkx as nx
from collections import Counter 
import json

import sys 
sys.path.append("..")
from src.graph import *

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

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.layout = html.Div([
    dcc.Store(id="snapshots-store"),
    dcc.Store(id="pos-store"),
    dcc.Store(id="edge-store", data=[
        {"start": 0, "end": 1, "prob": 0.2},
        {"start": 0, "end": 1, "prob": 0.8},
        {"start": 2, "end": 1, "prob": 0.2},
        {"start": 2, "end": 1, "prob": 0.9},
        {"start": 3, "end": 2, "prob": 0.1},
        {"start": 0, "end": 1, "prob": 0.01},
        {"start": 0, "end": 3, "prob": 0.4},
    ]),
    html.Div([
        dcc.Input(id="start-node", type="number", placeholder="Start node", style={"width": "80px"}),
        dcc.Input(id="end-node", type="number", placeholder="End node", style={"width": "80px", "marginLeft": "8px"}),
        dcc.Input(id="prob", type="number", placeholder="Probability", min=0, max=1, step=0.01, style={"width": "100px", "marginLeft": "8px"}),
        html.Button("Add Edge", id="add-edge-btn", n_clicks=0, style={"marginLeft": "8px"})
    ], style={"marginBottom": "10px"}),
    html.Button("Simulate", id="simulate-btn", n_clicks=0),
    html.Button("Run Monte Carlo", id="metrics-btn", n_clicks=0, style={"marginLeft": "8px"}),
    html.Div(
        style={"display": "flex", "alignItems": "flex-start"},
        children=[
            html.Div(
                style={"flex": "1 1 70%"},
                children=[
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
                    ]),
                ],
            ),
            html.Div(
                id="metrics",
                style={
                    "display": "block",
                    "flex": "0 0 30%",
                    "maxHeight": "600px",
                    "overflowY": "auto",
                    "marginLeft": "12px",
                    "padding": "10px",
                    "border": "1px solid #ddd",
                    "borderRadius": "6px",
                    "backgroundColor": "#fafafa",
                    "whiteSpace": "pre-wrap",
                    "fontFamily": "monospace"
                },
                children=[
                    html.H4("Monte Carlo Metrics"),
                    html.Div("Click 'Run Monte Carlo' to compute metrics on the current example graph.")
                ]
            )
        ]
    )
])

@app.callback(
    Output("edge-store", "data"),
    Input("add-edge-btn", "n_clicks"),
    Input("cy", "tapEdgeData"),
    State("start-node", "value"),
    State("end-node", "value"),
    State("prob", "value"),
    State("edge-store", "data"),
    prevent_initial_call=True
)
def upsert_edges(n_add, tap_edge, start, end, prob, edges):
    # Initialize list
    edges = list(edges or [])
    # Determine which input fired
    triggered = getattr(dash, "ctx", None)
    trig_id = triggered.triggered_id if triggered else None
    # Handle Add Edge
    if trig_id == "add-edge-btn":
        if start is None or end is None or prob is None:
            raise dash.exceptions.PreventUpdate
        u, v = sorted([int(start), int(end)])
        p = round(float(prob), 6)
        edges.append({"start": u, "end": v, "prob": p})
        return edges
    # Handle Delete via tapEdgeData
    if trig_id == "cy":
        if not tap_edge or not edges:
            raise dash.exceptions.PreventUpdate
        try:
            u_raw = int(tap_edge.get("source"))
            v_raw = int(tap_edge.get("target"))
            lbl = tap_edge.get("label")
            p_raw = float(lbl) if lbl is not None else None
        except (ValueError, TypeError):
            raise dash.exceptions.PreventUpdate
        if p_raw is None:
            raise dash.exceptions.PreventUpdate

        # Canonicalize endpoints (undirected) and round probability for stable matching
        u, v = sorted([u_raw, v_raw])
        p = round(p_raw, 6)

        for i, e in enumerate(list(edges)):
            eu, ev = sorted([int(e.get("start")), int(e.get("end"))])
            ep = round(float(e.get("prob", -1)), 6)
            if eu == u and ev == v and ep == p:
                del edges[i]
                break
        return edges
    # If something else triggered, do nothing
    raise dash.exceptions.PreventUpdate

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
    H = nx.MultiGraph()
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
    Output("edge-store", "data", allow_duplicate=True),
    Input("edge-store", "data"),
    prevent_initial_call=True
)
def normalize_edges(edges):
    if not edges:
        raise dash.exceptions.PreventUpdate
    normalized = []
    for e in edges:
        try:
            u, v = sorted([int(e["start"]), int(e["end"])])
            p = round(float(e["prob"]), 6)
            normalized.append({"start": u, "end": v, "prob": p})
        except Exception:
            continue
    # Only update if something actually changed
    if normalized != edges:
        return normalized
    raise dash.exceptions.PreventUpdate

@app.callback(
    Output("snapshots-store", "data"),
    Output("pos-store", "data"),
    Input("simulate-btn", "n_clicks"),
    Input("edge-store", "data"),
    prevent_initial_call=False,
)
def simulate_graph(_, edge_data):
    edge_lst = [Edge(e["start"], e["end"], e["prob"]) for e in edge_data]
    algo = DFS0CertificateAlgorithm()
    
    g = Graph(edge_lst)
    snaps = build_snapshots(algo, g)               # list[nx.MultiGraph]
    pos = get_fixed_positions(snaps[0])      # fixed positions from first snapshot
    return serialize_snapshots(snaps), pos

def simulate_monte_carlo(_):
    edge_lst = [
        Edge(0, 1, 0.2),
        Edge(0, 1, 0.8),
        Edge(0, 1, 0.2),
        Edge(1, 2, 0.2),
        Edge(1, 2, 0.9),
        Edge(3, 2, 0.1),
        Edge(1, 0, 0.01),
        Edge(3, 0, 0.4)
    ]
    algo = DFS0CertificateAlgorithm()
    metrics = monte_carlo(edge_lst, DFS0CertificateAlgorithm(len(edge_lst) - 1), 100, True)


@app.callback(
    Output("metrics", "children"),
    Input("metrics-btn", "n_clicks"),
    prevent_initial_call=False,
)
def run_metrics(n_clicks):
    # If not clicked yet, keep helper text
    if not n_clicks:
        return [
            html.H4("Monte Carlo Metrics"),
            html.Div("Click 'Run Monte Carlo' to compute metrics on the current example graph.")
        ]

    # Reuse the same synthetic graph/algo as simulate_graph().
    # If you later store edges/algo in dcc.Store, swap this out to read from there.
    edge_lst = [
        Edge(0, 1, 0.2),
        Edge(0, 1, 0.8),
        Edge(0, 1, 0.2),
        Edge(1, 2, 0.2),
        Edge(1, 2, 0.9),
        Edge(3, 2, 0.1),
        Edge(1, 0, 0.01),
        Edge(3, 0, 0.4),
    ]

    # Run Monte Carlo. Assumes monte_carlo is imported from src.graph via the wildcard import.
    # We pass a DFS0CertificateAlgorithm() instance; adjust trials/flags as your API expects.
    try:
        metrics = monte_carlo(edge_lst, Optimal1CertificateAlgorithm(), 10, True)
    except Exception as e:
        metrics = {"error": str(e)}

    # Pretty print whichever structure comes back (dict/list/tuple/etc.)
    def _pretty(x):
        try:
            return json.dumps(x, indent=2, ensure_ascii=False)
        except TypeError:
            # not JSON-serializable; fallback to repr
            return repr(x)

    return [
        html.H4("Monte Carlo Metrics"),
        html.Ul([
        html.Li(f"Simulations: {metrics.simulations}"),
        html.Li(f"Connected: {metrics.connected}"),
        html.Li(f"Disconnected: {metrics.disconnected}"),
        html.Li(f"Avg 1-certificate flips: {metrics.connected_flipped/metrics.connected:.2f}" if metrics.connected else "N/A"),
        html.Li(f"Avg 0-certificate flips: {metrics.disconnected_flipped/metrics.disconnected:.2f}" if metrics.disconnected else "N/A")
    ])
    ]

if __name__ == "__main__":
    app.run(debug=True)