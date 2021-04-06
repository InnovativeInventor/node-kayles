#![feature(test)]
#![feature(type_ascription)]
#![feature(const_fn)]

/*
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;
*/

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use petgraph::algo::is_isomorphic_matching;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::stable_graph::StableGraph;
use petgraph::Undirected;
use serde::{Deserialize, Serialize};
use fnv::{FnvHasher, FnvHashMap};
use hashbrown::raw::RawTable;
use std::fs::File;
use structopt::StructOpt;
use std::hash::{Hash, Hasher};

extern crate test;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "non-attacking-queens",
    about = "Queens shouldn't kill each other!"
)]
struct Opt {
    #[structopt(short = "n", long = "n", default_value = "9")]
    n: usize,

    #[structopt(short = "m", long = "m", default_value = "9")]
    m: usize,

    #[structopt(short = "t", long = "thread-level", default_value = "0")]
    thread_depth: usize,

    // Note: this will generate a lot of files
    #[structopt(short = "d", long = "distributed-level", default_value = "0")]
    dist_level: usize,

    // File to read from
    #[structopt(short = "r", long = "read")]
    read: Option<String>,
}

// The graph is represented by edges with a byte:
// 0: - , 1: |, 2: \, 3: /
fn instantiate_grid(
    n: usize,
    m: usize,
) -> (
    StableGraph<(), u8, Undirected>,
    Vec<Vec<((usize, usize, u64), NodeIndex)>>,
) {
    let mut grid = StableGraph::<(), u8, Undirected>::with_capacity(n*n, n*(n+1));
    let mut grid_tracker: Vec<Vec<((usize, usize, u64), NodeIndex)>> = Vec::with_capacity(n*n);
    for i in 0..n {
        let mut row = vec![];
        for j in 0..m {
            row.push(((i, j, hash(&(i, j))), grid.add_node(())));
            if j > 0 {
                grid.update_edge(row[j - 1].1, row[j].1, 0);
            }

            if i > 0 {
                grid.update_edge(grid_tracker[i - 1][j].1, row[j].1, 1);
            }

            if i > 0 && j > 0 {
                grid.update_edge(grid_tracker[i - 1][j - 1].1, row[j].1, 2);
            }

            if i > 0 && j < m - 1 {
                grid.update_edge(grid_tracker[i - 1][j + 1].1, row[j].1, 3);
            }
        }
        grid_tracker.push(row);
    }

    return (grid, grid_tracker);
}

fn main() {
    let opt = Opt::from_args();
    let (grid, grid_tracker) = instantiate_grid(opt.m, opt.n);
    let mut history: RawTable<usize> = RawTable::with_capacity(50000000);
    // let history: RwLock<HashMap<Vec<(usize, usize)>, usize>> = RwLock::new(HashMap::<Vec<(usize, usize)>, usize>::new());

    let mut state = match opt.read {
        Some(name) => BoardState::from(
            serde_cbor::from_reader(File::open(name).unwrap()).unwrap(): BoardStateRaw,
        ),
        None => BoardState::new(grid.clone())
    };

    let value = state.calculate(0, opt.dist_level, 0, & mut history, &grid_tracker);
    if value.is_some() {
        print!("Table: {{");
        for i in 0..opt.n {
            for j in 0..opt.m {
                if history.get(hash(&(i,j)), |_| true).is_some() {
                    print!("{:?}: {},", (i, j), history.get(hash(&(i,j)), |_| true).unwrap());
                }
            }
        }
        print!("}}\n");
        println!("Nimber: {}", value.unwrap());
        println!("Size of lookup table: {}", history.len());
    } else {
        println!("Progress saved to disk");
    }

    std::process::exit(0); // faster exit
}

struct BoardState {
    grid: StableGraph<(), u8, Undirected>,
}

impl From<BoardStateRaw> for BoardState {
    fn from(state: BoardStateRaw) -> Self {
        BoardState {
            grid: state.grid,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct BoardStateRaw {
    grid: StableGraph<(), u8, Undirected>,
}

impl From<BoardState> for BoardStateRaw {
    fn from(state: BoardState) -> Self {
        BoardStateRaw {
            grid: state.grid,
        }
    }
}

impl BoardState {
    fn new(
        grid: StableGraph<(), u8, Undirected>,
   //     history: &'static mut HashMap<Vec<(usize, usize)>, usize>
    ) -> Self {
        Self {
            grid: grid,
        }
    }
    fn calculate(&mut self, level: usize, dist_level: usize, stack: u64, history: &mut RawTable<usize>, grid_tracker: &Vec<Vec<((usize, usize, u64), NodeIndex)>>) -> Option<usize> {
        if self.grid.node_count() == 0 {
            return Some(0);
        } else if self.grid.node_count() == 1 {
            return Some(1);
        }

        let mut graph_history: Vec<Graph<(), u8, Undirected>> = Vec::with_capacity(self.grid.node_count());
        let mut values: Vec<usize> = Vec::with_capacity(self.grid.node_count());

        // let mut handles: Vec<(Vec<(usize, usize)>, std::thread::JoinHandle<Option<usize>>)> = Vec::new();
        /*
        let mut handles: Vec<(usize, usize, std::thread::JoinHandle<Option<usize>>)> = Vec::with_capacity(0);
        if level < thread_depth { // for performance
            handles.reserve_exact(self.grid.node_count());
        }
        */

        for i in 0..grid_tracker.len() {
            'nodeloop: for ((x, y, hash_value), mut node) in &grid_tracker[i] {
                if self.grid.contains_node(node) {
                    let mut curr_stack = stack.clone();
                    curr_stack ^= hash_value;

                    match history.get(curr_stack, |_| true) {
                        Some(value) =>  {
                            values.push(*value);
                            continue 'nodeloop
                        },
                        None => {}
                    }

                    // can make more efficient
                    let new_grid = remove_node(self.grid.clone(), &mut node);

                    if dist_level == level + 1 {
                        let name = format!("progress.{}.{}-{}.cbor", level, x, y);
                        let fs = File::create(name.clone()).unwrap();
                        serde_cbor::to_writer(
                            fs,
                            &BoardStateRaw::from(BoardState::new(new_grid)),
                        )
                        .expect(format!("Failed to serialize {}!", name).as_str());
                        continue 'nodeloop
                    }

                    let grid_graph = Graph::from(new_grid.clone());
                    for graph in &graph_history {
                        if graph.edge_count() == grid_graph.edge_count() {
                            // todo: measure if actually faster
                            for perms in &PERMUTATIONS_4 {
                                if is_isomorphic_matching(
                                    &grid_graph,
                                    graph,
                                    |_x, _y| true,
                                    |x, y| &perms[*x as usize] == y,
                                ) {
                                    // todo: examine diff with/without matching
                                    continue 'nodeloop;
                                }
                            }
                        }
                    }
                    graph_history.push(grid_graph);

                    let value = BoardState::new(new_grid).calculate(
                        level + 1,
                        dist_level,
                        curr_stack,
                        history,
                        grid_tracker
                    );
                    if value.is_some() {
                        let unwrapped_value = value.unwrap();
                        values.push(unwrapped_value);
                        {
                            history.insert(curr_stack, unwrapped_value, hash);
                        }
                    } else {
                        return None;
                    }
                }
            }
        }

        if dist_level == level + 1 {
            return None;
        }

        /*
        if level < thread_depth {
            for (stack, handle) in handles {
                let value = handle.join().unwrap();

                if value.is_some() {
                    let nimber = value.unwrap();

                    history.insert(stack, nimber);

                    values.push(nimber);
                } else {
                    return None;
                }
            }
        }
        */

        return Some(mex(values));
    }
}

fn remove_node(
    mut grid: StableGraph<(), u8, Undirected>,
    node: &mut NodeIndex,
) -> StableGraph<(), u8, Undirected> {
    let mut followed_nodes: Vec<(u8, NodeIndex)> = Vec::with_capacity(grid.node_count());
    let mut directional_followed_nodes: Vec<(u8, NodeIndex)> = vec![];

    for weight in 0..4 {
        directional_followed_nodes.clear();
        directional_followed_nodes.push((weight, *node));

        let mut stop = false;
        while stop == false {
            stop = true;
            for j in 0..directional_followed_nodes.len() {
                let mut walker = grid.neighbors(directional_followed_nodes[j].1).detach();
                while let Some((other_edge, other_node)) = walker.next(&grid) {
                    if grid.edge_weight(other_edge).unwrap() == &weight
                        && !directional_followed_nodes.contains(&(weight, other_node))
                    {
                        // can make more efficient
                        directional_followed_nodes.push((weight, other_node));
                        stop = false;
                    }
                }
            }
        }

        // add back in, note: can make more efficient
        for index in &directional_followed_nodes {
            if !followed_nodes.contains(index) {
                followed_nodes.push(*index);
            }
        }
    }

    let mut edge_map: FnvHashMap<u8, Vec<NodeIndex>> = FnvHashMap::default();
    for curr_node_index in 0..followed_nodes.len() {
        edge_map.clear();
        // stitch
        let (curr_weight, curr_node) = followed_nodes[curr_node_index];

        let mut curr_walker = grid.neighbors(curr_node).detach();
        while let Some((edge, other_node)) = curr_walker.next(&grid) {
            /*
            if followed_nodes.contains(&other_node) {
                continue // skip
            }
            */

            let weight = grid.edge_weight(edge).unwrap().clone();

            if weight == curr_weight {
                continue;
            }

            if edge_map.contains_key(&weight) {
                let same_weight_neighbors = edge_map.get_mut(&weight).unwrap();
                for k in 0..same_weight_neighbors.len() {
                    grid.add_edge(other_node, same_weight_neighbors[k], weight);
                }
                same_weight_neighbors.push(other_node);
            } else {
                edge_map.insert(weight, vec![other_node]);
            }
        }
    }

    for each_node in followed_nodes {
        // contract
        grid.remove_node(each_node.1);
    }
    // grid.remove_node(*node);

    grid
}

fn hash<T: Hash>(t: &T) -> u64 {
    let mut s: FnvHasher = Default::default();
    t.hash(&mut s);
    s.finish()
}

// fn mex(values: Vec<usize>) -> usize {
fn mex(values: Vec<usize>) -> usize {
    let mut min = 0;

    /*
    values.sort();
    for value in values {
        if value == min {
            min += 1;
        } else if value > min {
            return min
        }
    }
    */

    // faster
    while values.contains(&min) {
        min += 1;
    }

    return min;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{instantiate_grid, mex, remove_node, BoardState};
    use test::Bencher;

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
        assert!(mex(vec![0, 1, 2]) == 3);
        assert!(mex(vec![0, 2]) == 1);
        assert!(mex(vec![1, 0, 2]) == 3);
        assert!(mex(vec![3, 1, 2, 8]) == 0);
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
            let mut state = BoardState::new(grid.clone(), grid_tracker, Arc::new(RwLock::new(HashMap::<Vec<(usize, usize)>, usize>::new())));
            assert!(state.calculate(0, 0, 0, vec![]) == Some(*sol));
        }
    }

    #[test]
    fn test_end_multi() {
        for (size, sol) in &[(0, 0), (1, 1), (2, 1), (3, 2), (4, 1), (5, 3), (6, 1)] {
            let (grid, grid_tracker) = instantiate_grid(*size);
            let mut state = BoardState::new(grid.clone(), grid_tracker, Arc::new(RwLock::new(HashMap::<Vec<(usize, usize)>, usize>::new())));
            assert!(state.calculate(0, 3, 0, vec![]) == Some(*sol));
        }
    }

    #[bench]
    fn bench_5(b: &mut Bencher) {
        let (grid, grid_tracker) = instantiate_grid(5);
        let mut state = BoardState::new(grid.clone(), grid_tracker, Arc::new(RwLock::new(HashMap::<Vec<(usize, usize)>, usize>::new())));
        b.iter(|| {
            state.calculate(0, 0, 0, vec![]);
        });
    }

    #[bench]
    fn bench_6(b: &mut Bencher) {
        let (grid, grid_tracker) = instantiate_grid(6);
        let mut state = BoardState::new(grid.clone(), grid_tracker, Arc::new(RwLock::new(HashMap::<Vec<(usize, usize)>, usize>::new())));
        b.iter(|| {
            state.calculate(0, 0, 0, vec![]);
        });
    }

    #[bench]
    fn bench_mex(b: &mut Bencher) {
        let test_vec = vec![3, 3, 9, 2, 0, 9, 8, 4, 2, 7];
        b.iter(|| {
            mex(test_vec.clone());
        });
    }
}

// Vec::from_iter((0..5).permutations(4))
// This is only for performance :)
const PERMUTATIONS_4: [[u8; 4]; 24] = [
    [0, 1, 2, 3],
    [0, 1, 3, 2],
    [0, 2, 1, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
    [0, 3, 2, 1],
    [1, 0, 2, 3],
    [1, 0, 3, 2],
    [1, 2, 0, 3],
    [1, 2, 3, 0],
    [1, 3, 0, 2],
    [1, 3, 2, 0],
    [2, 0, 1, 3],
    [2, 0, 3, 1],
    [2, 1, 0, 3],
    [2, 1, 3, 0],
    [2, 3, 0, 1],
    [2, 3, 1, 0],
    [3, 0, 1, 2],
    [3, 0, 2, 1],
    [3, 1, 0, 2],
    [3, 1, 2, 0],
    [3, 2, 0, 1],
    [3, 2, 1, 0],
];
