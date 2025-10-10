from __future__ import annotations #postpones type hint eval  
import copy
from typing import Optional
import dynamic_connectivity as dc

class Edge:
    def __init__(self, node1, node2, prob):
        self.node1 = node1
        self.node2 = node2
        self.prob = prob
    
    def __lt__(self, other: Edge): 
        return self.prob < other.prob 
    
    def __eq__(self, other: Edge):
        return self.prob == other.prob 

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
    def __init__(self, n):
        self.parent = list(range(n)) #initially each node is its own parent
        self.rank = [1] * n #initially each node is in a set of size 1
        self.components = n #initially there are n components

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
        for edge in edges:
            if edge.prob != 0: #if the edge is definitely connected 
                self.graph.add_edge(edge.node1, edge.node2)

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
        
        self.edges = edges
        # adjacency list values interpretation 
        # 0-1 = probability of edge being valid 
        # once an edge is discovered, the probability collapses to either 0 or 1
        self.n = max(max(edge.node1, edge.node2) for edge in edges) + 1 # number of vertices
        self.adjlst = generate_adj_list()  #current state of the graph

        self.dsu = CheckJoined(self.n) #DSU to check connectivity
        self.disjoined = CheckDisjoined(self.n, self.edges) #to check disjointedness
        self.edges_flipped = 0 #number of edges flipped

    def __repr__(self):
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
        
    def get_edges(self) -> list[Edge]:
        return self.edges 

class GraphAlgorithm: 
    def __init__(self):
        pass
    
    def run(self, graph: Graph):
        print("Running base algorithm")
        pass #return the next edge to flip


class SimpleRingAlgorithm(GraphAlgorithm):
    def __init__(self, k: int):
        self.k = k #number of edges to be True 

    def run(self, graph: Graph):
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

    def run(self, graph: Graph):
        #sort the graph by the edges probabilty and always choose the highest likelihood 
        edges = graph.get_edges() 
        edges.sort()  #should automatically sort by edge probabilies 
        #sorted(edges, key = lambda x: x.probability)

        #find the first value that is not 1 
        #finds the first value that is greater than or equal to current value 
        from bisect import bisect_left 
        index = bisect_left(edges, Edge(0,0,1)) #search with an edge that has a 1 probability 
        answer = None 
        if (index - 1 >= 0): 
            answer = edges[index-1]
        return answer
    
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
    def __init__(self, simulations = 100):
        self.simulations = simulations
        self.connected = 0
        self.disconnected = 0 
        self.edges_flipped = 0 

    def update(self, graph: Graph):
        if graph.connected() == True:
            self.connected += 1
        else:
            self.disconnected += 1
        self.edges_flipped += graph.edges_flipped

    def __repr__(self):
        return f"Metrics(simulations={self.simulations}, connected={self.connected}, disconnected={self.disconnected}, runtime={self.edges_flipped/self.simulations})"
    
def monte_carlo(edge_lst: list[Edge], algorithm: GraphAlgorithm, iterations=100) -> float: 
    #monte carlo with take the graph at the point your insert it and then run the algorithm on it, reporting metrics 
    metrics = Metrics(simulations=iterations)
    #metrics track the number of total flips reported by the graph since its creation, not since its insertion into monte carlo function

    for _ in range(iterations):
        temp_graph = Graph(copy.deepcopy(edge_lst))
        test_algorithm(algorithm, temp_graph)
        metrics.update(temp_graph)
    return metrics



if __name__ == "__main__": 
    #------------------Example usage
    edge_lst = [Edge(0, 1, 0.1), Edge(1, 2, 0.99), Edge(2, 3, 0.8), Edge(3, 4, 0.99), Edge(4, 0, 0.99)]
    #test_algorithm(SimpleRingAlgorithm(len(edge_lst) - 1), Graph(edge_lst), False)

    edge_lst = [Edge(0, 1, 0.1), Edge(1, 2, 0.99), Edge(2, 0, 0.8)]
    #test_algorithm(AlwaysHighestAlgorithm(), Graph(edge_lst), True)

    #------------------Test Monte Carlo 
    edge_lst = [Edge(0, 1, 0.1), Edge(1, 2, 0.99), Edge(2, 3, 0.8), Edge(3, 4, 0.35), Edge(4, 0, 0.99)]
    #copies were not made, must create another instance of the edge list because a side effect of initialization is that 
    #the edge list is modified 
    metrics = monte_carlo(edge_lst, SimpleRingAlgorithm(len(edge_lst) - 1), 100)
    print(metrics)