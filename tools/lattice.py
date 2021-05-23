import networkx as nx
import serialize
import typer

"""
Usage: latice.py [OPTIONS]

Options:
      --n INTEGER           [default: 3]
      --m INTEGER           [default: 0]
      --filename TEXT       [default: input.json]
      --install-completion  Install completion for the current shell.
      --show-completion     Show completion for the current shell, to copy it or
                            customize the installation.

      --help                Show this message and exit

Example usage:
    python queen.py 9 input.json

    To calculate the board state, run:
        ./target/release/node-kayles -r lattice.json
"""

def main(n: int = 3, m: int = 0, filename: str = "lattice.json"):
    graph = nx.Graph()
    counter = 0
    node_grid = []
    for i in range(n):
        row = []
        for j in range(m):
            graph.add_node(counter)
            row.append(counter)

            if i > 0:
                graph.add_edge(node_grid[i-1][j], counter)
            if j > 0:
                graph.add_edge(row[j-1], counter)

            counter += 1
        node_grid.append(row)
        
    serialize.encode(graph, filename)

if __name__ == "__main__":
    typer.run(main)

