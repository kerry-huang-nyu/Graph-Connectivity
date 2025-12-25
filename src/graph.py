from __future__ import annotations #postpones type hint eval  
import copy
from functools import cache
import random
from typing import Optional
from . import dynamic_connectivity as dc
from collections import defaultdict


class Edge:
    def __init__(self, node1: int, node2: int, prob: float):
        self.node1 = node1
        self.node2 = node2
        self.prob = prob
    
    #comparison operator is only created to serve the probabilities 
    def __lt__(self, other: Edge): 
        return (self.prob, self.node1, self.node2) < (other.prob, other.node1, other.node2)
    
    # def __eq__(self, other: Edge):
    #     return self.prob == other.prob 

    def __repr__(self):
        return f"Edge({self.node1}, {self.node2}, {self.prob})"
    
    def flip(self):
        import random
        result = random.random() < self.prob
        if result: 
            self.prob = 1
        else:   
            self.prob = 0
        return result 

class CheckJoined: #Disjoint Set Union for connected components
    def __init__(self, n: int, edges: list[Edge]):
        self.parent = list(range(n)) #initially each node is its own parent
        self.rank = [1] * n #initially each node is in a set of size 1
        self.components = n #initially there are n components

        #we should add in edges that are for sure joined 
        for edge in edges: 
            if edge.prob == 1: 
                self.union(edge.node1, edge.node2)

    def find(self, u): #find with path compression 
        #ackerman's function at work, very efficient 
        #finds the group that the node belongs to 
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        rootU = self.find(u)
        rootV = self.find(v)
        if rootU != rootV:
            self.components -= 1
            if self.rank[rootU] > self.rank[rootV]:
                self.parent[rootV] = rootU
            elif self.rank[rootU] < self.rank[rootV]:
                self.parent[rootU] = rootV
            else:
                self.parent[rootV] = rootU
                self.rank[rootU] += 1

    def connected(self) -> bool:
        return self.components == 1

class CheckDisjoined: 
    def __init__(self, n, edges: list[Edge]):
        self.graph = dc.DynamicConnectivity(n)

        #we need to initialize the self.graph with all the edges that are connected 
        seen = set() 
        for edge in edges:
            
            marker = edge.node1, edge.node2 
            if edge.node1 > edge.node2:
                marker = edge.node2, edge.node1

            if edge.prob != 0 and marker not in seen: #if the edge is definitely connected 
                self.graph.add_edge(edge.node1, edge.node2)
                seen.add(marker)

    def disunion(self, u, v): 
        self.graph.remove_edge(u, v)

    def connected(self) -> bool: 
        #currently just a placeholder with a dfs to check connectivity in the future it will be dynamically updated 
        return self.graph.get_number_of_connected_components() == 1 


class Graph:
    def __init__(self, edges: list[Edge]): #assume edges are 0 indexed 
        #helper function purely for initialization 
        def generate_adj_list() -> dict: 
            adj_list = [{} for i in range(self.n)]

            #don't store the probabilities, store the edges 
            for edge in self.edges: #assuming that this is an undirected graph
                if edge.node2 in adj_list[edge.node1]: 
                    adj_list[edge.node1][edge.node2].append(edge)
                    adj_list[edge.node2][edge.node1].append(edge)
                else:
                    adj_list[edge.node1][edge.node2] = [edge]
                    adj_list[edge.node2][edge.node1] = [edge]
            return adj_list
        
        self.edges = copy.deepcopy(edges) #we make a copy of the edges so that they are flipped 
        # adjacency list values interpretation 
        # 0-1 = probability of edge being valid 
        # once an edge is discovered, the probability collapses to either 0 or 1
        self.n = max((max(edge.node1, edge.node2) for edge in edges), default=0) + 1 # number of vertices always 1 vertex 
        self.adjlst = generate_adj_list()  #current state of the graph

        self.dsu = CheckJoined(self.n, self.edges) #DSU to check connectivity
        self.disjoined = CheckDisjoined(self.n, self.edges) #to check disjointedness
        self.edges_flipped = 0 #number of edges flipped

    def get_bundle_status(self, start: int, end: int) -> Optional[bool]:
        status = False #false indicates 0 certificate 
        #all disconnected edges gives a disconnected bundle 

        for edge in self.adjlst[start][end]:
            if 1 > edge.prob > 0: #indeterminate edges with no certain edges gives indeterminate 
                status = None 

            if edge.prob == 1: #any connected edge makes the entire bundle connected 
                status = True 
                break 

        return status
    
    def get_bundle_test_by_descending_prob(self, start: int, end: int) -> tuple[float, float]: 
        #return a status of expected value 

        lst = sorted(self.adjlst[start][end], reverse=True)
        #sort the lst by the probability 
        expected = 0
        prob = 1 

        for edge in lst: 
            if 1 > edge.prob > 0: 
                expected += prob
                prob *= (1 - edge.prob)

            if edge.prob == 1:
                return (0, 0) #probability it is 0 is 0 and expected tests is 0 
        
        return (prob, expected) #prob is now Q_i and expected is C_i


    def __repr__(self) -> str:
        return f"Graph({self.adjlst})"
    
    def flip(self, edge: Edge) -> bool:
        result = edge.flip() #flip the edge to be either 0 or 1 return true or false 
        self.edges_flipped += 1
        if result: #if the edge is valid 
            self.dsu.union(edge.node1, edge.node2) #update the DSU 
        else: #if the edge is invalid and there are no other untested edges between the two nodes 
            untested = False 
            for edges in self.adjlst[edge.node1].get(edge.node2, []):
                if edges.prob > 0: #if there is an untested edge or a true edge 
                    untested = True 
                    break 
            if untested == False:
                self.disjoined.disunion(edge.node1, edge.node2) #update the disjoined checker
        return result
    
    def connected(self) -> Optional[bool]:
        #check if the graph is connected dynamically 
        #self.disjoined.update(self)
        # print("DSU CONNECTED IS :", self.dsu.connected())
        # print("DISJOINED CONNECTED IS :", self.disjoined.connected())

        if self.dsu.connected():
            return True
        elif self.disjoined.connected():
            return None
        else:
            return False
        
    def get_edges(self) -> list[Edge]: #modifiable edges that we get 
        return self.edges 

class GraphFactory:
    """Factory for generating random multigraphs (bundled edges).

    Parameters correspond to the sliders you described:
      1) num_nodes: number of bundles / nodes in the base graph.
      2) bundle_size_std: stddev controlling how non-uniform bundle sizes are.
      3) bundle_size_mean: mean number of parallel edges per bundle.
      4) prob_std: stddev of edge probabilities.
      5) prob_mean: mean of edge probabilities.

    By default we generate a ring base graph, so the number of bundles
    equals num_nodes (pairs (i, i+1)). This matches your current ring focus.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def _clip01(self, x: float) -> float:
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

    def _sample_int(self, mean: float, std: float, min_value: int = 1) -> int:
        # Gaussian then round, with a floor so bundles don't disappear.
        val = int(round(self.rng.gauss(mean, std)))
        return max(min_value, val)

    def _sample_prob(self, mean: float, std: float) -> float:
        # Gaussian in R, clipped to [0, 1].
        sigfig = 2 #2 significant figures 
        return self._clip01(round(self.rng.gauss(mean, std), sigfig))

    def create(
        self,
        num_nodes: int,
        bundle_size_mean: float,
        bundle_size_std: float,
        prob_mean: float,
        prob_std: float,
        topology: str = "ring",
        erdos_p: float = 0.3,
    ) -> Graph:
        """Create a Graph with bundled/parallel edges.

        Args:
            num_nodes: number of vertices in the graph.
            bundle_size_mean: average bundle size (parallel edges per node-pair).
            bundle_size_std: stddev of bundle sizes.
            prob_mean: average probability of an edge being present.
            prob_std: stddev of edge probabilities.
            topology: 'ring' (default), 'path', 'complete', or 'erdos_renyi'.
            erdos_p: if topology == 'erdos_renyi', probability a base edge exists.
        """
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if bundle_size_mean <= 0:
            raise ValueError("bundle_size_mean must be positive")
        if bundle_size_std < 0 or prob_std < 0:
            raise ValueError("stddev parameters must be non-negative")

        # 1) Choose base (simple) edges = bundles.
        base_pairs: list[tuple[int, int]] = []
        topo = topology.lower()

        if topo == "ring":
            for i in range(num_nodes):
                base_pairs.append((i, (i + 1) % num_nodes))
        elif topo == "path":
            for i in range(num_nodes - 1):
                base_pairs.append((i, i + 1))
        elif topo == "complete":
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    base_pairs.append((i, j))
        elif topo == "erdos_renyi":
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if self.rng.random() < erdos_p:
                        base_pairs.append((i, j))
            # ensure at least a connected-ish skeleton if p is tiny
            if not base_pairs:
                base_pairs = [(i, i + 1) for i in range(num_nodes - 1)]
        else:
            raise ValueError(f"Unknown topology: {topology}")

        # 2) Sample bundle sizes and edge probabilities.
        edges: list[Edge] = []
        for (u, v) in base_pairs:
            k = self._sample_int(bundle_size_mean, bundle_size_std, min_value=1)
            for _ in range(k):
                p = self._sample_prob(prob_mean, prob_std)
                edges.append(Edge(u, v, p))

        return Graph(edges)


def make_graph(
    num_nodes: int,
    bundle_size_mean: float,
    bundle_size_std: float,
    prob_mean: float,
    prob_std: float,
    topology: str = "ring",
    seed: Optional[int] = None,
) -> Graph:
    """Convenience wrapper for one-off graph generation."""
    return GraphFactory(seed=seed).create(
        num_nodes=num_nodes,
        bundle_size_mean=bundle_size_mean,
        bundle_size_std=bundle_size_std,
        prob_mean=prob_mean,
        prob_std=prob_std,
        topology=topology,
    )

class GraphAlgorithm: 
    def __init__(self):
        pass
    
    def run(self, graph: Graph) -> Edge: #return the edge object to flip 
        print("Running base algorithm")
        pass #return the next edge to flip


class SimpleRingAlgorithm(GraphAlgorithm):
    def __init__(self, k: int):
        self.k = k #number of edges to be True 

    def run(self, graph: Graph) -> Edge:
        #return the edge with the highest probability 
        #assume that the graph is a ring (and not multigraph) we want to find the next edges to test
        
        #analyzes the graph, find the edge 
        #needs to find 2 broken edges or n-1 working edges 

        #1. find all edges
        edges = graph.get_edges()

        #2. find all edges that are not yet tested  
        untested_edges = [edge for edge in edges if edge.prob not in (0, 1)]
        num_true = sum(1 for edge in edges if edge.prob == 1)

        #3. find the correct value for k 
        target = self.k - num_true 
        sorted_edges = sorted(untested_edges, key=lambda x: x.prob, reverse=True)
        #print("Sorted edges:", sorted_edges)

        #4. return the edge 
        return sorted_edges[target-1] #return the edge with the kth highest probability
    

class AlwaysHighestAlgorithm(GraphAlgorithm):
    def __init__(self): #I mean I can definitely make this more efficient and just sort the edges once 
        pass  

    def run(self, graph: Graph) -> Edge:
        #sort the graph by the edges probabilty and always choose the highest likelihood 
        edges = graph.get_edges() 
        sortedges = [(edge.prob, i) for i, edge in enumerate(edges)]
        sortedges.sort()  #should automatically sort by edge probabilies 
        #sorted(edges, key = lambda x: x.probability)

        #find the first value that is not 1 
        #finds the first value that is greater than or equal to current value 
        from bisect import bisect_left 
        index = bisect_left(sortedges, (1, -1)) #search for any edge with a 1 percent probability 
        tupl = None 
        if (index - 1 >= 0): 
            tupl = sortedges[index-1]
        else: #some other exceptions here as well uncaught
            raise Exception("error, should have terminated already since I have tested all values ")
        output = edges[tupl[1]]
        return output
    

class Optimal1CertificateAlgorithm(GraphAlgorithm):
    def __init__(self): 
        pass  

    def run(self, graph: Graph) -> Edge:
        #sort the graph by the edges probabilty and always choose the highest likelihood 
        edges = graph.get_edges() 

        sortedges = [(edge.prob, i) for i, edge in enumerate(edges)]
        sortedges.sort()  #should automatically sort by edge probabilies 

        #go through the sorted edges and make sure you select the highest edge the in bundle that is not determined
        from bisect import bisect_left 
        index = bisect_left(sortedges, (1, -1)) #search for any edge with a certain probability 
        for i in range(index - 1, -1, -1): 
            edgeprob, loc = sortedges[i]
            edge = edges[loc]
            if graph.get_bundle_status(edge.node1, edge.node2) == None:  #still indeterminate 
                return edge 
        raise ValueError("I am supposed to return an answer but it turns out all bundles are already evaluated?")
    
class OptimalAlgorithm(GraphAlgorithm): 
    def __init__(self): 
        pass 
    
    @cache 
    def dp(self, edges:tuple[Edge]) -> (float, Edge): # the float indicates the expected number to flip, then edge is the optimal edge to flip 
        original = Graph(edges)

        if original.connected() != None: #either connected or not 
            return (0, None)  #already determined, so then no need to do anything anymore to further investigate 
        
        #assume that edges is already sorted 
        #go through the edges to find one that is unflipped 
        minflip = float('inf')
        edgeflip = None 

        for i in range(len(edges)): 
            edge = edges[i]

            #if we can have the edge switch to antoher 
            if edge.prob not in (0, 1): #so we are not deterministic yet 
                #flip to be true 
                newedges = list(copy.copy(edges))
                newedges[i] = Edge(edge.node1, edge.node2, 1) #flip to 1
                newedges.sort() 
                expected_true, _ = self.dp(tuple(newedges))

                #flip to be false 
                newedges = list(copy.copy(edges))
                newedges[i] = Edge(edge.node1, edge.node2, 0) #flip to 0
                newedges.sort() 
                expected_false, _ = self.dp(tuple(newedges))

                expected_flip = edge.prob * expected_true + (1 - edge.prob) * expected_false + 1 #+1 for the flip we just did

                if expected_flip < minflip:
                    minflip = expected_flip
                    edgeflip = edge 

        if edgeflip == None or minflip == float('inf'):
            raise ValueError
        return (minflip, edgeflip) 

    def run(self, graph:Graph) -> Edge: 
        edges = graph.get_edges() 
        newedges = copy.copy(edges)
        newedges.sort()

        expected_flips, edge = self.dp(tuple(newedges))

        #return the edge lying within the graph itself
        for edge_original in edges: 
            if edge_original.node1 == edge.node1 and edge_original.node2 == edge.node2 and edge_original.prob == edge.prob:
                return edge_original
        raise ValueError
    
    
class DFS0CertificateAlgorithm(GraphAlgorithm): 
    #the idea is always to choose the dfs 0 algorithm 
    def __init__(self): 
        pass  

    def run(self, graph: Graph) -> Edge:
        #get all of the buckets and choose the one that doesn't have a determination 
        #get all of the buckets and choose the bucket by the chance that we have a 0 

        lst = graph.get_edges() 
        tup = [graph.get_bundle_test_by_descending_prob(edge.node1, edge.node2) for edge in lst]
        zipped = list(zip(tup, lst))

        #prob/expected we want high probability 
        filtered = [(stats[0]/stats[1], edge) for stats, edge in zipped if stats[1] != 0]
        lst = sorted(filtered) #sort by 
        return lst[-1][-1]


class SmallestBundleAlgorithm(GraphAlgorithm):
    def __init__(self):
        pass 

    def run(self, graph: Graph): 
        #get all bundles that have not resolved yet 
        lst = graph.get_edges() 
        bundles = defaultdict(list) 

        for edge in lst: 
            if graph.get_bundle_status(edge.node1, edge.node2) == None: #still undefined 
                first, second = (edge.node1, edge.node2) if edge.node1 < edge.node2 else (edge.node2, edge.node1)
                bundles[(first, second)].append(edge)
        
        answer = [value for key, value in bundles.items()]
        answer.sort(key=len)

        answer[0].sort()

        return answer[0][-1]


        #get the edge that has the highest value of being 1 



def step_algorithm(algo: GraphAlgorithm, graph: Graph):
    while graph.connected() == None: 
        test_edge = algo.run(graph)
        result = graph.flip(test_edge)
        yield test_edge, result

def test_algorithm(algo: GraphAlgorithm, graph: Graph, print_steps: bool = False):
    while graph.connected() == None:
        if print_steps:
            print(graph)
        test_edge = algo.run(graph) 
        if print_steps: 
            print("Flipping edges:", test_edge)

        result = graph.flip(test_edge) 
        if print_steps:
            print("Result is", result)
    #print(graph.connected())
    
class Metrics:
    def __init__(self):
        self.simulations = 0
        self.connected = 0
        self.disconnected = 0 
        self.connected_flipped = 0
        self.disconnected_flipped = 0 

    def get_total_runs(self) -> int:
        return self.connected + self.disconnected 

    def update(self, graph: Graph):
        if graph.connected() == True:
            self.connected += 1
            self.connected_flipped += graph.edges_flipped
        else:
            self.disconnected += 1
            self.disconnected_flipped += graph.edges_flipped
        self.simulations += 1 

    def __repr__(self) -> str:
        return f"Metrics(simulations={self.simulations}, connected={self.connected}, disconnected={self.disconnected}, 1 certificate={self.connected_flipped/self.connected}, 0 certificate={self.disconnected_flipped/self.disconnected})"
    
def monte_carlo(edge_lst: list[Edge], algorithm: GraphAlgorithm, iterations:int =100, certificate:Optional[bool]=None) -> float: 
    #monte carlo with take the graph at the point your insert it and then run the algorithm on it, reporting metrics 
    metrics = Metrics()
    #metrics track the number of total flips reported by the graph since its creation, not since its insertion into monte carlo function

    condition = lambda metric: metric.get_total_runs() == iterations
    if certificate == True: #this means 1 certificate only 
        condition = lambda metric: metric.connected == iterations 
    elif certificate == False: 
        condition = lambda metric: metric.disconnected == iterations 

    while condition(metrics) == False:
        temp_graph = Graph(edge_lst)
        test_algorithm(algorithm, temp_graph)
        metrics.update(temp_graph)
        
    return metrics


if __name__ == "__main__": 
    # ------------------Factory example
    # factory = GraphFactory(seed=0)
    # g = factory.create(
    #     num_nodes=6,
    #     bundle_size_mean=3,
    #     bundle_size_std=1,
    #     prob_mean=0.6,
    #     prob_std=0.2,
    #     topology="ring",
    # )
    # test_algorithm(AlwaysHighestAlgorithm(), g, True)
    #------------------Example usage
    #edge_lst = [Edge(0, 1, 0.1), Edge(1, 2, 0.99), Edge(2, 3, 0.3), Edge(3, 4, 0.99), Edge(4, 0, 0.99)]
    #test_algorithm(SimpleRingAlgorithm(len(edge_lst) - 1), Graph(edge_lst), True)

    #print("\n starting always highest algorithm")

    #edge_lst = [Edge(0, 1, 0.1), Edge(1, 2, 0.99), Edge(2, 0, 0.8), Edge(0, 1, 0.0)]
    #test_algorithm(AlwaysHighestAlgorithm(), Graph(edge_lst), True)

    #------------------Test Monte Carlo 
    #edge_lst = [Edge(0, 1, 0.1), Edge(1, 2, 0.99), Edge(2, 3, 0.8), Edge(3, 4, 0.35), Edge(4, 0, 0.99)]
    #copies were not made, must create another instance of the edge list because a side effect of initialization is that 
    #the edge list is modified 
    # metrics = monte_carlo(edge_lst, SimpleRingAlgorithm(len(edge_lst) - 1), 100, True)
    # print(metrics)

    # edge_lst = [
    #     Edge(0, 1, 0.9),
    #     Edge(0, 1, 0.8),
    #     Edge(0, 1, 0.2),
    #     Edge(1, 2, 0.2),
    #     Edge(1, 2, 0.9),
    #     Edge(0, 2, 0.1),
    # ]
    # algo = Optimal1CertificateAlgorithm()
    # g = Graph(edge_lst)
    # test_algorithm(algo, g, True)


    # edge_lst = [
    #     Edge(0, 1, 0.9),
    #     Edge(0, 1, 0.8),
    #     Edge(0, 1, 0.2),
    #     Edge(1, 2, 0.2),
    #     Edge(1, 2, 0.9),
    #     Edge(0, 2, 0.1),
    # ]
    # algo = OptimalAlgorithm()
    # g = Graph(edge_lst)
    # test_algorithm(algo, g, True)

    edge_lst = [
        Edge(0, 1, 0.9),
        Edge(0, 1, 0.8),
        Edge(0, 1, 0.2),
        Edge(1, 2, 0.2),
        Edge(1, 2, 0.9),
        Edge(0, 2, 0.1),
    ]
    algo = OptimalAlgorithm()
    g = Graph(edge_lst)
    test_algorithm(algo, g, True)


    # ------------------Monte Carlo report over factory graphs
    # from dataclasses import dataclass

    # @dataclass
    # class GraphSpec:
    #     name: str
    #     num_nodes: int
    #     bundle_mean: float
    #     bundle_std: float
    #     prob_mean: float
    #     prob_std: float
    #     topology: str = "ring"

    # specs = [
    #     GraphSpec(
    #         name="G1: ring(10 nodes), bundle mean=3",
    #         num_nodes=10,
    #         bundle_mean=3,
    #         bundle_std=1,
    #         prob_mean=0.6,
    #         prob_std=0.2,
    #     ),
    #     GraphSpec(
    #         name="G2: ring(5 nodes), bundle mean=5",
    #         num_nodes=5,
    #         bundle_mean=5,
    #         bundle_std=1,
    #         prob_mean=0.6,
    #         prob_std=0.2,
    #     ),
    #     GraphSpec(
    #         name="G3: ring(3 nodes), bundle mean=2",
    #         num_nodes=3,
    #         bundle_mean=2,
    #         bundle_std=1,
    #         prob_mean=0.6,
    #         prob_std=0.2,
    #     ),
    # ]

    # def build_algorithms(n_nodes: int) -> list[tuple[str, GraphAlgorithm]]:
    #     # SimpleRingAlgorithm expects k = target #true edges in a SIMPLE ring.
    #     # We use n_nodes - 1, which matches the classic 1-certificate for an n-cycle.
    #     return [
    #         ("SimpleRingAlgorithm", SimpleRingAlgorithm(n_nodes - 1)),
    #         ("AlwaysHighestAlgorithm", AlwaysHighestAlgorithm()),
    #         ("Optimal1CertificateAlgorithm", Optimal1CertificateAlgorithm()),
    #         ("DFS0CertificateAlgorithm", DFS0CertificateAlgorithm()),
    #         ("SmallestBundleAlgorithm", SmallestBundleAlgorithm()),
    #     ]

    # def run_report(iterations: int = 1000):
    #     factory = GraphFactory(seed=0)
    #     for spec in specs:
    #         print("\n==============================")
    #         print(spec.name)
    #         print("(iterations =", iterations, ")")
    #         g = factory.create(
    #             num_nodes=spec.num_nodes,
    #             bundle_size_mean=spec.bundle_mean,
    #             bundle_size_std=spec.bundle_std,
    #             prob_mean=spec.prob_mean,
    #             prob_std=spec.prob_std,
    #             topology=spec.topology,
    #         )
    #         base_edges = g.get_edges()  # safe because Graph deepcopies internally

    #         algos = build_algorithms(spec.num_nodes)
    #         results = []
    #         for algo_name, algo in algos:
    #             m = monte_carlo(base_edges, algo, iterations)
    #             results.append((algo_name, m))

    #         # Pretty print
    #         header = f"{'Algorithm':30s} | {'Connected%':10s} | {'Avg flips (1-cert)':18s} | {'Avg flips (0-cert)':18s}"
    #         print(header)
    #         print("-" * len(header))
    #         for algo_name, m in results:
    #             total = m.get_total_runs() if m.get_total_runs() else 1
    #             connected_pct = 100.0 * m.connected / total
    #             avg1 = (m.connected_flipped / m.connected) if m.connected else float('nan')
    #             avg0 = (m.disconnected_flipped / m.disconnected) if m.disconnected else float('nan')
    #             print(f"{algo_name:30s} | {connected_pct:9.2f}% | {avg1:18.3f} | {avg0:18.3f}")

    # run_report(iterations=1000)