use petgraph::graph::{NodeIndex, Graph};
use petgraph::stable_graph::StableGraph;
use petgraph::Undirected;
use petgraph::dot::{Dot};
use petgraph::algo::{is_isomorphic_matching};
use std::collections::HashMap;
use std::io::Write;
use structopt::StructOpt;
use itertools::Itertools;
use std::thread;

#[derive(Debug, StructOpt)]
#[structopt(name = "non-attacking-queens", about = "Queens shouldn't kill each other!")]
struct Opt {
    #[structopt(short, long)]
    size: usize,

    #[structopt(short = "t", long = "thread-level", default_value = "2")]
    thread_depth: usize
}

// The graph is represented by nodes that have a bool (alive or dead) and a edges with a byte
// 0: - , 1: |, 2: \, 3: /

fn instantiate_grid(n: usize) -> (StableGraph<bool, u8, Undirected>, Vec<Vec<((usize, usize), NodeIndex)>>) {
    let mut grid = StableGraph::<bool, u8, Undirected>::default();
    let mut grid_tracker: Vec<Vec<((usize, usize), NodeIndex)>> = vec![];
    for i in 0..n {
        let mut row = vec![];
        for j in 0..n {
            row.push(((i, j), grid.add_node(true)));
            if j > 0 {
                grid.update_edge(row[j-1].1, row[j].1, 0);
            }

            if i > 0 {
                grid.update_edge(grid_tracker[i-1][j].1, row[j].1, 1);
            }

            if i > 0 && j > 0 {
                grid.update_edge(grid_tracker[i-1][j-1].1, row[j].1, 2);
            }

            if i > 0 && j < n-1 {
                grid.update_edge(grid_tracker[i-1][j+1].1, row[j].1, 3);
            }

        }
        grid_tracker.push(row);
    }

    return (grid, grid_tracker)
}

fn main() {
    let opt = Opt::from_args();
    let (grid, grid_tracker) = instantiate_grid(opt.size);

    let mut state = BoardState::new(grid.clone(), grid_tracker);
    let value = state.calculate(0, opt.thread_depth-1);
    println!("{}", value);

    let mut fs = std::fs::File::create("out.dot").unwrap();
    fs.write_all(format!("{}", Dot::new(&grid)).as_bytes())
        .unwrap();
}

struct BoardState {
    grid: StableGraph<bool, u8, Undirected>,
    grid_tracker: Vec<Vec<((usize, usize), NodeIndex)>>, // todo: flatten
}

impl BoardState {
    fn new(grid: StableGraph<bool, u8, Undirected>, grid_tracker: Vec<Vec<((usize, usize), NodeIndex)>>) -> Self {
        Self { 
            grid: grid,
            grid_tracker: grid_tracker
        }
    }
    fn calculate(&mut self, level: usize, thread_depth: usize) -> usize {
        if self.grid.node_count() == 0 && self.grid.edge_count() == 0 {
            return 0
        } else if self.grid.node_count() == 1 {
            return 1
        }

        let mut graph_history: Vec<Graph<bool, u8, Undirected>> = vec![];
        let mut values: Vec<usize> = vec![];
        let mut diagram = HashMap::<(usize, usize), usize>::new();
        let mut handles: Vec<(usize, usize, std::thread::JoinHandle<usize>)> = vec![];

        for i in 0..self.grid_tracker.len() {
            'nodeloop: for ((x, y), mut node) in &self.grid_tracker[i] {
                if self.grid.contains_node(node) { // can make more efficient
                    let new_grid = remove_node(self.grid.clone(), &mut node);
                    assert!(new_grid.node_count() < self.grid.node_count()); // defensive

                    let grid_graph = &Graph::from(new_grid.clone());
                    for graph in &graph_history {
                        for perms in (0..5).permutations(4) {
                            if is_isomorphic_matching(grid_graph, graph, |_x, _y| true, |x, y| &perms[*x as usize]==y) { // todo: examine diff with/without matching
                                continue 'nodeloop;
                            }
                        }
                    }
                    graph_history.push(Graph::from(new_grid.clone()));


                    let grid_tracker = self.grid_tracker.clone();
                    if level < thread_depth {
                        handles.push((x.clone(), y.clone(), std::thread::spawn(move || {
                            let value = BoardState::new(new_grid, grid_tracker).calculate(level+1, thread_depth);
                            value
                        })));

                    } else {
                        let value = BoardState::new(new_grid, grid_tracker).calculate(level+1, thread_depth);
                        values.push(value);
                    }
                }
            }
        }


        if level < thread_depth {
            for (x, y, handle) in handles {
                let value = handle.join().unwrap();
                diagram.insert((x, y), value.clone());
                values.push(value);
            }

        }

        if level == 0 {
            println!("{:?}", diagram);
        }

        return mex(values);
    }
}

fn remove_node(mut grid: StableGraph<bool, u8, Undirected>, node: &mut NodeIndex) -> StableGraph<bool, u8, Undirected> {
    let mut followed_nodes: Vec<NodeIndex> = vec![];
    for weight in 0..5 {
        let mut directional_followed_nodes: Vec<NodeIndex> = vec![*node];
        let mut stop = false;
        while stop==false {
            stop = true;
            for j in 0..directional_followed_nodes.len() {
                let mut walker = grid.neighbors(directional_followed_nodes[j]).detach();
                while let Some((other_edge, other_node)) = walker.next(&grid) {
                    if grid.edge_weight(other_edge).unwrap().clone()==weight && ! directional_followed_nodes.contains(&other_node) { // can make more efficient
                        directional_followed_nodes.push(other_node);
                        stop = false;
                    }
                }
            }
        }

        // add back in, note: can make more efficient
        for index in directional_followed_nodes {
            if ! followed_nodes.contains(&index) {
                followed_nodes.push(index);
            }
        }
    }

    for curr_node_index in 0..followed_nodes.len() { // stitch
        let curr_node = followed_nodes[curr_node_index];

        let mut edge_map: HashMap<u8, Vec<NodeIndex>> = HashMap::new();
        let mut curr_walker = grid.neighbors(curr_node).detach();
        while let Some((edge, other_node)) = curr_walker.next(&grid) {
            /*
            if followed_nodes.contains(&other_node) {
                continue // skip
            }
            */

            let weight = grid.edge_weight(edge).unwrap().clone();

            if edge_map.contains_key(&weight) {
                let mut same_weight_neighbors = edge_map.get(&weight).unwrap().clone();
                for other_seen_node in &same_weight_neighbors {
                    grid.add_edge(other_node, *other_seen_node, weight.clone());
                }
                same_weight_neighbors.push(other_node);
                edge_map.entry(weight.clone()).or_insert(same_weight_neighbors.to_vec()); 
            } else {
                edge_map.insert(weight.clone(), vec![other_node]);
            }
        }
    }

    for each_node in followed_nodes { // contract
        grid.remove_node(each_node);
    }
    grid.remove_node(*node);

    grid
}

fn mex(values: Vec<usize>) -> usize {
    let mut min = 0;
    while values.contains(&min) {
        min += 1;
    }
    return min
}

#[cfg(test)]
mod tests {
    use crate::{instantiate_grid, mex, remove_node, BoardState};
    use petgraph::dot::{Dot};
    use std::io::Write;
    use std::collections::HashMap;

    #[test]
    fn test_grid_2() {
        let (grid, _grid_tracker) = instantiate_grid(2);
        assert!(grid.node_count() == 4);
        assert!(grid.edge_count() == 6);
    }

    #[test]
    fn test_grid_3() {
        let (grid, _grid_tracker) = instantiate_grid(3);
        assert!(grid.node_count() == 9);
        assert!(grid.edge_count() == 20);
    }

    #[test]
    fn test_mex() {
        assert!(mex(vec![0,1,2]) == 3);
        assert!(mex(vec![0,2]) == 1);
        assert!(mex(vec![1,0,2]) == 3);
        assert!(mex(vec![3,1,2,8]) == 0);
    }

    #[test]
    fn test_remove_node_2() {
        let (grid, grid_tracker) = instantiate_grid(2);
        let new_grid = remove_node(grid, &mut grid_tracker[0][0].1.clone());
        assert!(new_grid.node_count() == 0);
        assert!(new_grid.edge_count() == 0);
    }

    #[test]
    fn test_remove_node_3() {
        let (grid, grid_tracker) = instantiate_grid(3);

        let new_grid = remove_node(grid, &mut grid_tracker[0][0].1.clone());

        assert!(new_grid.node_count() == 2);
        assert!(new_grid.edge_count() == 1);
    }

    // end to end tests

    #[test]
    fn test_end() {
        for (size, sol) in &[(0, 0), (1, 1), (2, 1), (3, 2), (4, 1), (5, 3), (6, 1)] {
            let (grid, grid_tracker) = instantiate_grid(*size);
            let mut state = BoardState::new(grid.clone(), grid_tracker);
            assert!(state.calculate(0, 0)==*sol);
        }
    }

    #[test]
    fn test_end_multi() {
        for (size, sol) in &[(0, 0), (1, 1), (2, 1), (3, 2), (4, 1), (5, 3), (6, 1)] {
            let (grid, grid_tracker) = instantiate_grid(*size);
            let mut state = BoardState::new(grid.clone(), grid_tracker);
            assert!(state.calculate(0, 3)==*sol);
        }
    }
}
