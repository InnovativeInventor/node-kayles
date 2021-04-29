import networkx as nx
from networkx.readwrite import json_graph
import json

def encode(graph: nx.Graph, filename: str) -> dict:
    nodes = [None]*len(graph.nodes)
    edges = []
    for edge in graph.edges:
        edges.append([edge[0], edge[1], None])

    serialized_graph = {
        "grid": {
            "nodes": nodes,
            "node_holes": [],
            "edge_property": "undirected",
            "edges": edges
        }
    }

    with open(filename, "w") as f:
        json.dump(dict(serialized_graph), f)
