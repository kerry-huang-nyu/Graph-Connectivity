from pyvis.network import Network

net = Network(cdn_resources='in_line', notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white")
net.add_node(1, label="Node 1")
net.add_node(2, label="Node 2")
net.add_edge(1, 2)
net.show("my_graph.html")


# import networkx as nx
# from pyvis.network import Network

# G = nx.barabasi_albert_graph(50, 2)
# centrality = nx.betweenness_centrality(G)

# net = Network(cdn_resources="in_line", notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white")
# for n in G.nodes():
#     net.add_node(n, label=str(n), value=centrality[n]*100)

# for u, v in G.edges():
#     net.add_edge(u, v)

# net.show("graph.html")