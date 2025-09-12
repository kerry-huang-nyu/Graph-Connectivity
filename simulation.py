import itertools


#create an interface that accepts booleans 
#create an interface that accepts graphs 
#this is limited to rings for now? 
#this algorithm should be able to handle more complex graphs in the future 

#ring graph where each edge has a probability of being valid 

class Edge:
    def __init__(self, node1, node2, prob):
        self.node1 = node1
        self.node2 = node2
        self.prob = prob

    def __repr__(self):
        return f"Edge({self.node1}, {self.node2}, {self.prob})"


class Graph:
    def __init__(self, edges: list[Edge]):
        self.edges = edges
        # adjacency list values interpretation 
        # 0-1 = probability of edge being valid 
        # once an edge is discovered, the probability collapses to either 0 or 1
        self.n = max(max(edge.node1, edge.node2) for edge in edges) + 1 # number of vertices
        self.adj_list = self.generate_adj_list() 

    def generate_adj_list(self) -> dict:
        adj_list = [{} for i in range(self.n)]

        for edge in self.edges: #assuming that this is an undirected graph
            if edge.node2 in adj_list[edge.node1]: 
                adj_list[edge.node1][edge.node2].append(edge.prob)
                adj_list[edge.node2][edge.node1].append(edge.prob)
            else:
                adj_list[edge.node1][edge.node2] = [edge.prob]
                adj_list[edge.node2][edge.node1] = [edge.prob]
        return adj_list

    def __repr__(self):
        return f"Graph({self.edges})"


#create boolean variables through the number of variables 
#then have the statement with and, or statementes 
#just parse through the entire statement and see if it is true or false definitively 
class Boolean:
    def __init__(self, probability: float):
        self.probability = probability
        self.value = None # None means unknown, True means true, False means false
        self.parent = None 
    # def set_value(self, value: bool):
    #     self.value = value

    def propagate(self):
        if self.parent is not None:
            for parent in self.parent:
                parent.update(self)
        else:
            raise ValueError("No parent to propagate to", self)

    def flip(self): #flip this coin to see what value this is 
        import random
        rand_val = random.random()
        if rand_val < self.probability:
            self.value = True
        else:
            self.value = False
        self.propagate()

        return self.value 
    
    def get_booleans(self) -> set:
        return {self}

    def __repr__(self):
        return f"Boolean({self.value, self.probability})"

class Gate:
    def __init__(self, literals: set, type: str): 
        self.literals = literals
        self.type = type #AND or OR
        self.value = None # None means unknown, True means true, False means false
        self.parent = None 
        self.count = 0 #number of literals that have been evaluated
        self.set_parent() 

    def set_parent(self):
        for literal in self.literals:
            if literal.parent is not None:
                literal.parent.add(self)
            else:
                literal.parent = {self}

    def __repr__(self):
        return f"{self.value} ({self.literals})"
    
    def eval(self):
        return self.value 
    
    def propagate(self):
        if self.parent is not None:
            for parent in self.parent:
                parent.update(self)
        else:
            print("No parent to propagate to", self)

    def get_booleans(self) -> set:
        answer = set() 
        for literal in self.literals:
            answer.update(literal.get_booleans())
        return answer 
    
    def update(self, item): #update the literals and that may update the the Gate value 
        if item in self.literals:
            if item.value is True:
                if self.type == "OR": 
                    self.value = True
                    self.propagate()
                else:
                    #self.literals.remove(item)
                    self.count += 1 
                    if self.count == len(self.literals):
                        self.value = True
                        self.propagate()
            else:
                if self.type == "AND":
                    self.value = False
                    self.propagate()
                else:
                    # self.literals.remove(item)
                    self.count += 1
                    if self.count == len(self.literals):
                        self.value = False
                        self.propagate()
        else:
            raise ValueError("Item not in literals", item)

class Algorithm:
    def __init__(self):
        pass

    def run(self, Gate: Gate):
        print("Running base algorithm")
        pass #return the next boolean to flip

class KofNAlgorithm(Algorithm):
    def __init__(self, k: int):
        self.k = k #number of booleans to be True 

    def run(self, gate: Gate):
        #return the boolean with the highest probability 
        literals = gate.get_booleans()
        literals = list(literals)
        print("Literals:", literals)
        #eliminate the ones that are already known 
        num_true = sum(1 for lit in literals if lit.value is True)
        literals = [lit for lit in literals if lit.value is None]

        target = self.k - num_true

        sorted_literals = sorted(literals, key=lambda x: x.probability, reverse=True)
        #print(Gate )
        #print("Sorted literals:", sorted_literals)
        return sorted_literals[target-1] #return the boolean with the kth highest probability


edge_lst = [Edge(0, 1, 0.1), Edge(1, 2, 0.99), Edge(2, 3, 0.5), Edge(3, 4, 0.2), Edge(4, 0, 0.99)]

def create_k_of_n_boolean(probabilities: list[float], k: int) -> Gate:
    booleans = [Gate({Boolean(prob)}, "AND") for prob in probabilities]
    gates = []

    for combo in itertools.combinations(booleans, k):
        gates.append(Gate(set(combo), "AND"))
    gate = Gate(set(gates), "OR") #at least one of these has to be true 
    return gate

def test_algorithm(algo: Algorithm, Gate: Gate): 
    while Gate.eval() is None:
        test_boolean = algo.run(Gate) 
        print("Flipping boolean:", test_boolean)

        #flip the next_val based on its probability 
        result = test_boolean.flip() 
        print("Result is", result)

#------------------Example Usage
k = 2
gate = create_k_of_n_boolean([0.1, 0.5, 0.9, 0.8, 0.3], k)
test_algorithm(KofNAlgorithm(k), gate)