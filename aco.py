import matplotlib.pyplot as plt
import networkx as nx
import operator
import numpy as np

class Map:
    """
    Useful class to represent the space domain of the problem
    """
    def __init__(self, nodes, distance_matrix):
        self.nodes = nodes
        self.distances = distance_matrix
        self.pheromones = np.ones((len(self.nodes), len(self.nodes)))
        self.local_pheromones = np.zeros((len(self.nodes), len(self.nodes)))
        for i in range(len(nodes)):
            self.pheromones[i][i] = 0

    def update_pheromone_local(self, fro, to, val):
        self.local_pheromones[fro][to] += val

    def update_pheromones_global(self):
        for i in range(0, len(self.nodes)):
            for j in range(0, len(self.nodes)):
                self.pheromones[i][j] = (1 - evaporation_factor) * self.pheromones[i][j] + self.local_pheromones[i][j]
        self.reset_locals()

    def get_pheromone(self, fro, to):
        return self.pheromones[fro][to]

    def get_distance(self, fro, to):
        return self.distances[fro][to]

    def reset_locals(self):
        self.local_pheromones = np.zeros((len(self.nodes), len(self.nodes)))


def draw_graph(graph, i, colorize_solution=False):
    """
    Helper function to draw networkx graphs
    """
    G = nx.Graph()
    G.add_edges_from(graph[0])
    pos = nx.spring_layout(G)
    plt.figure(i)
    if colorize_solution:
        edge_colors = ['purple' if graph[1][e]=="PATH" else 'orange' for e in G.edges]
    else:
        edge_colors ="orange"
    nx.draw(G, pos, edge_color=edge_colors, 
            node_size=100, node_color='lightgreen', alpha=0.59,
            labels={node: node for node in G.nodes()})

    nx.draw_networkx_edge_labels(G, pos, edge_labels=graph[1], font_color='black')
    plt.savefig(f"{save_at}graph{i}.png", dpi=600)
    return G

class Ant:
    """
    Ant class of the ACO algorithm
    """
    def __init__(self, current, unvisited):
        self.current = current
        self.unvisited = unvisited
        self.trail_length = 0

    def travel_next(self):
        prob = np.zeros(len(self.unvisited))
        prob_list = np.zeros(len(self.unvisited))
        fro = ord(self.current) - 65
        sum = 0
        for i in range(0, len(prob)):
            to = ord(self.unvisited[i]) - 65
            prob[i] = pow(country.get_pheromone(fro, to), alpha) * pow((1 / country.get_distance(fro, to)), beta)
            sum += prob[i]

        prob = [prob[i]/sum for i in range(0, len(prob))]
        prob_list = [(prob[i], self.unvisited[i]) for i in range(len(prob))]
        prob_list.sort(key=operator.itemgetter(0), reverse=True)
        probability = np.random.random()
        dest = -1
        for i in range(0, len(prob_list)):
            if probability < prob_list[i][0]:
                self.current = prob_list[i][1]
                dest = ord(self.current) - 65
                self.trail_length += country.get_distance(fro, dest)
                break
            else:
                probability -= prob_list[i][0]
        self.unvisited.remove(self.current)
        if fro != dest:
            country.update_pheromone_local(fro, dest,
                                           country.get_pheromone(fro, dest) + 1 / country.get_distance(
                                               fro, dest))

    def reset_ant(self):
        self.unvisited.clear()
        self.unvisited += country.nodes
        self.unvisited.remove(self.current)

def init():
    # making nodes
    nodes = [chr(i + 65) for i in range(node_count)]
    # creating distances
    for i in range(0, node_count):
        temp = []
        for j in range(0, i):
            temp.append(distances[j][i])
        for j in range(i, node_count):
            if i == j:
                temp.append(0)
            else:
                temp.append(np.random.uniform(min_distance_limit, max_distance_limit))
        distances.append(temp)
    return nodes

if __name__ == "__main__":
    node_count = 6
    min_distance_limit = 1
    max_distance_limit = 100
    iterations = 100
    evaporation_factor = 0.5
    alpha = 1
    beta = 5
    save_at = "./"
    nodes = []
    copy_nodes = []
    edges = []
    distances = []
    edgeLabels = {}
    nodes = init()
    country = Map(nodes, distances)
    ants = []
    copy_nodes += nodes
    np.random.shuffle(copy_nodes)

    # Generating Ants
    ants = [Ant(copy_nodes[i], list(set(copy_nodes) - set(copy_nodes[i]))) for i in range(len(copy_nodes))]

    # ACO
    cur = 0
    while cur < iterations:
        for i in range(len(nodes) - 1):
            for ant in ants:
                ant.travel_next()
        best = np.inf
        for ant in ants:
            if best > ant.trail_length:
                best = ant.trail_length
        for ant in ants:
            ant.reset_ant()
        country.update_pheromones_global()
        cur += 1

    # Finding shortest paths based on pheromone values
    shortest = []
    for i in range(node_count):
        max = 0
        for j in range(node_count):
            if country.pheromones[i][j] > country.pheromones[i][max]:
                max = j
        shortest.append((chr(i+65), chr(max+65), country.get_distance(i, max)))

    print(f"Best Traversal Distance: {best}")
    print(f"Shortest Paths A to B(A, B) are: {shortest}")

    # Paint graphs:

    # adding edges and edge labels to distance graph
    for i in range(node_count):
       for j in range(i+1, node_count):
            if i != j:
                edges.append([nodes[i], nodes[j]])
                edgeLabels[(nodes[i], nodes[j])] = distances[i][j]
    distance_graph = [edges, edgeLabels]
    dg = draw_graph(distance_graph, 0)

    # adding edges and edge labels to pheromone graph
    for i in range(node_count):
        for j in range(i+1, node_count):
            if i != j:
                edgeLabels[(nodes[i], nodes[j])] = f"{np.round(country.pheromones[i][j], 2):.4f}" if country.pheromones[i][j] < 100000 else "PATH"
    pheromone_graph = [edges, edgeLabels]
    pg = draw_graph(pheromone_graph, 1, True)
