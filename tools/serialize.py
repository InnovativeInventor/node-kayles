import networkx as nx
from networkx.readwrite import json_graph
import json
import secrets

def encode(graph: nx.Graph, filename: str) -> dict:
    nodes = [None]*len(graph.nodes)
    edges = []
    for edge in graph.edges:
        edges.append([edge[0], edge[1], None])

    grid_tracker = []
    for node in graph.nodes:
        grid_tracker.append(((node, 0, secrets.randbelow(1<<64)), node))

    serialized_graph = (
        {
            "grid": {
                "nodes": nodes,
                "node_holes": [],
                "edge_property": "undirected",
                "edges": edges
            }
        }, 
        [grid_tracker]
    )

    with open(filename, "w") as f:
        json.dump(serialized_graph, f)
