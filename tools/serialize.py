import networkx as nx
from networkx.readwrite import json_graph
import json
import secrets

def encode(graph: nx.Graph, filename: str) -> dict:
    """
    Takes in a networkx graph with integer node values and serializes it to a json file.
    Example usage with binary:
        ./target/release/node-kayles -r input.json
    Directed graphs are not supported
    """
    assert not isinstance(graph, nx.DiGraph)
    assert not isinstance(graph, nx.MultiDiGraph)

    nodes = [None]*len(graph.nodes)
    edges = []
    for edge in graph.edges:
        edges.append([edge[0], edge[1], None])

    grid_tracker = []
    for node, _ in enumerate(graph.nodes):
        grid_tracker.append(((node, 0, secrets.randbelow(1<<64)), node))
        # grid_tracker.append(((node, 0, abs(hash(node))), node))

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
