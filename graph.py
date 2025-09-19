
class Edge:
    def __init__(self, node1, node2, prob):
        self.node1 = node1
        self.node2 = node2
        self.prob = prob

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
    def __init__(self, graph):
        self.graph = graph

    def update(self, graph): #in the future this will not be a graph that is passed in, but an edge removed from the graph 
        self.graph = graph

    def connected(self) -> bool: 
        #currently just a placeholder with a dfs to check connectivity in the future it will be dynamically updated 
        adjlst = self.graph.adjlst

        #for the state just check if all nodes are connected 
        visited = set()
        def dfs(node):
            visited.add(node)
            for neighbor in adjlst[node]:
                if neighbor not in visited:
                    for edge in adjlst[node][neighbor]:
                        if edge.prob != 0: #only traverse edges that are valid
                            dfs(neighbor)
                            break #exit 
        dfs(0)
        return len(visited) == self.graph.n


class Graph:
    def __init__(self, edges: list[Edge]): #assume edges are 0 indexed 
        self.edges = edges
        # adjacency list values interpretation 
        # 0-1 = probability of edge being valid 
        # once an edge is discovered, the probability collapses to either 0 or 1
        self.n = max(max(edge.node1, edge.node2) for edge in edges) + 1 # number of vertices
        self.adjlst = self.generate_adj_list()  #current state of the graph

        self.dsu = CheckJoined(self.n) #DSU to check connectivity
        self.disjoined = CheckDisjoined(self) #to check disjointedness

    def generate_adj_list(self) -> dict:
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

    def __repr__(self):
        return f"Graph({self.adjlst})"
    
    def flip(self, edge: Edge) -> bool:
        result = edge.flip() #flip the edge to be either 0 or 1 return true or false 
        if result: #if the edge is valid 
            self.dsu.union(edge.node1, edge.node2) #update the DSU 
        else:
            self.disjoined.update(self) #update the disjoined checker
        return result
    
    def connected(self):
        #check if the graph is connected dynamically 
        self.disjoined.update(self)
        # print("DSU CONNECTED IS :", self.dsu.connected())
        # print("DISJOINED CONNECTED IS :", self.disjoined.connected())

        if self.dsu.connected():
            return True
        elif self.disjoined.connected():
            return None
        else:
            return False
        
    def get_edges(self):
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
        num_true = sum(1 for edge in edges if edge.prob is 1)

        #3. find the correct value for k 
        target = self.k - num_true 
        sorted_edges = sorted(untested_edges, key=lambda x: x.prob, reverse=True)
        #print("Sorted edges:", sorted_edges)

        #4. return the edge 
        return sorted_edges[target-1] #return the edge with the kth highest probability
    

def test_algorithm(algo: GraphAlgorithm, graph: Graph):
    while graph.connected() == None:
        print(graph)
        test_edge = algo.run(graph) 
        print("Flipping edges:", test_edge)

        result = graph.flip(test_edge) 
        print("Result is", result)
    print(graph.connected())
    

#------------------Example usage
edge_lst = [Edge(0, 1, 0.1), Edge(1, 2, 0.99), Edge(2, 3, 0.8), Edge(3, 4, 0.99), Edge(4, 0, 0.99)]
test_algorithm(SimpleRingAlgorithm(len(edge_lst) - 1), Graph(edge_lst))