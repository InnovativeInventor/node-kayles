import networkx as nx
import serialize
import typer

"""
Usage: queen.py [OPTIONS]

Options:
      --n INTEGER           [default: 9]
      --filename TEXT       [default: input.json]
      --install-completion  Install completion for the current shell.
      --show-completion     Show completion for the current shell, to copy it or
                            customize the installation.

      --help                Show this message and exit

Example usage:
    python queen.py 9 input.json

    To calculate the board state, run:
        ./target/release/node-kayles -sr input.json
"""

def main(n: int = 9, filename: str = "queens.json"):
    graph = nx.Graph()
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
        
    serialize.encode(graph, filename)

if __name__ == "__main__":
    typer.run(main)

