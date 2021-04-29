import networkx as nx
import serialize

graph = nx.Graph()
n = 9
counter = 0
node_grid = []
for i in range(n):
    row = []
    for j in range(n):
        graph.add_node(counter)
        row.append(counter)
        for k in range(1, counter+1):
            if j - k >= 0:
                graph.add_edge(row[j-k], counter)
            if i - k >= 0:
                graph.add_edge(node_grid[i-k][j], counter)
            if i - k >= 0 and j - k >= 0:
                graph.add_edge(node_grid[i-k][j-k], counter)
            if i - k >= 0 and j + k < n:
                graph.add_edge(node_grid[i-k][j+k], counter)
        counter += 1
    node_grid.append(row)
print("Edges:", graph.number_of_edges())
print("Nodes:", graph.number_of_nodes())
    
serialize.encode(graph, "output.json")
