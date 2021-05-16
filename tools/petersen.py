import networkx as nx
import serialize
import typer

def draw_petersen(cycle: int) -> nx.Graph:
    """
    Credit: Riley S. Waechter, https://oeis.org/A316533/a316533.txt
    """
    graph = nx.Graph()
    for i in range(cycle):
        graph.add_node(i)
        graph.add_node(i+cycle)
    for i in range(cycle):
        graph.add_edge(cycle + i, (cycle + i + 1)%cycle + cycle )
        graph.add_edge(i, i+cycle)
    for i in range(cycle):
        graph.add_edge(i, (i+2)%cycle)
        graph.add_edge(i, (i+(cycle-2))%cycle)
    return graph

def main(n: int = 3, filename: str = "petersen.json"):
    graph = draw_petersen(n)
    serialize.encode(graph, filename)

if __name__ == "__main__":
    typer.run(main)

