#![feature(test)]
#![feature(type_ascription)]
#![feature(destructuring_assignment)]

/*
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;
*/

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use fnv::FnvHasher;
use petgraph::algo::is_isomorphic;
use petgraph::graph::{Graph};
use petgraph::stable_graph::{NodeIndex};
use petgraph::stable_graph::StableGraph;
use petgraph::Undirected;
use serde::{Deserialize, Serialize};
use std::cmp;
use std::collections::HashMap;
use std::fs::File;
use std::hash::{BuildHasher, Hash, Hasher};
use petgraph::visit::NodeIndexable;
use petgraph::visit::IntoEdgeReferences; 
use petgraph::visit::EdgeRef; 
use petgraph::unionfind::UnionFind;
use std::io;
use structopt::StructOpt;

extern crate test;

#[derive(Debug, StructOpt, Clone)]
#[structopt(
    name = "non-attacking-queens",
    about = "Queens shouldn't kill each other!"
)]
struct Opt {
    #[structopt(short = "n", long = "n", default_value = "10")]
    n: usize,

    #[structopt(short = "m", long = "m", default_value = "10")]
    m: usize,

    #[structopt(long = "seq")]
    seq: bool,

    #[structopt(short = "v", long = "verbose")]
    verbose: bool,

    #[structopt(short = "t", long = "thread-level", default_value = "0")]
    thread_depth: usize,

    // Note: this will generate a lot of files
    #[structopt(short = "d", long = "distributed-level", default_value = "0")]
    dist_level: usize,

    // File to read from
    #[structopt(short = "r", long = "read")]
    read: Option<String>,

    // Non-interactive, for use in scripts and CI/CD
    #[structopt(short = "s", long = "non-interactive")]
    noninteractive: bool,

    #[structopt(short = "o", long = "output")]
    output: bool,
}

// The graph is represented by edges with a byte:
// 0: - , 1: |, 2: \, 3: /
fn instantiate_grid(
    n: usize,
    m: usize,
) -> (
    StableGraph<u64, (), Undirected>,
    Vec<Vec<((usize, usize), NodeIndex)>>,
) {
    println!("Calculating {} by {} board", n, m);
    let mut grid = StableGraph::<u64, (), Undirected>::with_capacity(n * n, n * (n + 1));
    let mut grid_tracker: Vec<Vec<((usize, usize), NodeIndex)>> = Vec::with_capacity(n * n);
    for i in 0..n as i64 {
        let mut row = vec![];
        for j in 0..m as i64 {
            row.push((
                (i as usize, j as usize),
                grid.add_node(hash(&(i as usize, j as usize))),
            ));
            for k in 1..(cmp::max(i, j) + 1) {
                if j - k >= 0 {
                    grid.update_edge(row[(j - k) as usize].1, row[j as usize].1, ());
                }

                if i - k >= 0 {
                    grid.update_edge(
                        grid_tracker[(i - k) as usize][j as usize].1,
                        row[j as usize].1,
                        (),
                    );
                }

                if i - k >= 0 && j - k >= 0 {
                    grid.update_edge(
                        grid_tracker[(i - k) as usize][(j - k) as usize].1,
                        row[j as usize].1,
                        (),
                    );
                }

                if i - k >= 0 && j + k < m as i64 {
                    grid.update_edge(
                        grid_tracker[(i - k) as usize][(j + k) as usize].1,
                        row[j as usize].1,
                        (),
                    );
                }
            }
        }
        grid_tracker.push(row);
    }

    (grid, grid_tracker)
}

fn main() {
    let mut opt = Opt::from_args();

    if opt.seq {
        opt.n = 0;
        opt.m = 0;
    }

    let (mut state, mut grid_tracker) = match opt.read.clone() {
        Some(name) => {
            println!("Warning: table coords may not be correct");
            (
                BoardState::from(
                    (serde_json::from_reader(File::open(name.clone()).unwrap()).unwrap():
                        (BoardStateRaw, Vec<Vec<((usize, usize), NodeIndex)>>))
                        .0,
                ),
                (serde_json::from_reader(File::open(name).unwrap()).unwrap():
                    (BoardStateRaw, Vec<Vec<((usize, usize), NodeIndex)>>))
                    .1,
            )
        }
        None => {
            let (grid, grid_tracker) = instantiate_grid(opt.m, opt.n);
            (BoardState::new(grid), grid_tracker)
        }
    };

    let mut grid = state.grid.clone();

    loop {
        run(&mut state.clone(), opt.clone(), &grid_tracker);

        if opt.noninteractive {
            break;
        }

        if !opt.seq {
            println!("X coord:");
            let mut x = String::new();
            io::stdin()
                .read_line(&mut x)
                .expect("failed to read from stdin");

            println!("Y coord:");
            let mut y = String::new();
            io::stdin()
                .read_line(&mut y)
                .expect("failed to read from stdin");

            state = match y.trim().parse::<usize>() {
                // TODO: switch back to if/let syntax
                Ok(y_coord) => match x.trim().parse::<usize>() {
                    Ok(x_coord) => {
                        println!("Move to play: ({}, {})", x_coord, y_coord);

                        assert!(grid_tracker[x_coord][y_coord].0 .0 == x_coord); // defensive
                        assert!(grid_tracker[x_coord][y_coord].0 .1 == y_coord);

                        (_, grid) = remove_node(grid, &mut grid_tracker[x_coord][y_coord].1.clone(), 0);
                        BoardState::new(grid.clone())
                    }
                    Err(_err) => state.clone(),
                },
                Err(_err) => state.clone(),
            };
        } else {
            opt.n += 1;
            opt.m += 1;
            (grid, grid_tracker) = instantiate_grid(opt.m, opt.n);
            state = BoardState::new(grid.clone());
        }
    }

    if opt.output {
        let fs = File::create("output.json").unwrap();
        serde_json::to_writer(fs, &(&BoardStateRaw::from(state), grid_tracker)).unwrap();
    }

    std::process::exit(0); // faster exit
}

fn run(
    state: &mut BoardState,
    opt: Opt,
    grid_tracker: &Vec<Vec<((usize, usize), NodeIndex)>>,
) {
    let mut history: HashMap<u64, usize, U64Hasher> =
        HashMap::with_capacity_and_hasher(50000000, U64Hasher::new());
    // let original_grid = state.grid.clone(); 
    
    let value = state.calculate(0, opt.dist_level, 0, &mut history);
    if value.is_some() {
        println!("Nimber: {}", value.unwrap());
        if opt.verbose {
            print!("Table (symmetries removed): {{");
            for i in 0..grid_tracker.len() {
                for ((x, y), mut node) in &grid_tracker[i] {
                    let (hash_value, _graph) = remove_node(state.grid.clone(), &mut node, 0);
                    if history.get(&hash_value).is_some() {
                        print!("{:?}: {},", (x, y), history.get(&hash_value).unwrap());
                    }
                }
            }
            println!("}}");
            println!("Size of lookup table: {}", history.len());
        }
    } else {
        if opt.verbose {
            println!("Progress saved to disk");
        }
    }
}

pub struct U64Hasher(u64);

impl U64Hasher {
    fn new() -> Self {
        Self { 0: 0 }
    }
}

impl Hasher for U64Hasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.0 = u64::from_ne_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);

        /*
        self.0 = unsafe {
            std::mem::transmute([bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]])
        };
        */
    }
}

impl BuildHasher for U64Hasher {
    type Hasher = U64Hasher;
    #[inline]
    fn build_hasher(&self) -> U64Hasher {
        U64Hasher::new()
    }
}

#[derive(Debug, Clone)]
struct BoardState {
    grid: StableGraph<u64, (), Undirected>,
}

impl From<BoardStateRaw> for BoardState {
    fn from(state: BoardStateRaw) -> Self {
        BoardState { grid: state.grid }
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct BoardStateRaw {
    grid: StableGraph<u64, (), Undirected>,
}

impl From<BoardState> for BoardStateRaw {
    fn from(state: BoardState) -> Self {
        BoardStateRaw { grid: state.grid }
    }
}

impl BoardState {
    fn new(
        grid: StableGraph<u64, (), Undirected>,
        //     history: &'static mut HashMap<Vec<u64>, usize>
    ) -> Self {
        Self { grid }
    }
    fn calculate(
        &mut self,
        level: usize,
        dist_level: usize,
        stack: u64,
        history: &mut HashMap<u64, usize, U64Hasher>,
    ) -> Option<usize> {
        if self.grid.node_count() == 0 {
            return Some(0);
        } else if self.grid.node_count() == 1 {
            return Some(1);
        }

        let subgraphs = connected_subgraphs(self.grid.clone(), stack);
        // let subgraphs_len = subgraphs.len();

        let mut sum_value = 0;
        'subgraphloop: for (subgraph, subgraph_stack) in &subgraphs {
            match history.get(&subgraph_stack) { // we check the lookup table twice
                Some(value) => {
                    sum_value ^= *value;
                    continue 'subgraphloop;
                }
                None => {}
            }

            let mut graph_history: Vec<Graph<u64, (), Undirected>> =
                Vec::with_capacity(self.grid.node_count());
            let mut values: Vec<usize> = Vec::with_capacity(self.grid.node_count());

            'nodeloop: for mut node in subgraph.node_indices() {
                if self.grid.contains_node(node) {
                    // can make more efficient
                    let (curr_stack, new_grid) = remove_node(subgraph.clone(), &mut node, *subgraph_stack);
                    if new_grid.node_count() == 0 {
                        values.push(0);
                        continue 'nodeloop;
                    } else if new_grid.node_count() == 1 {
                        values.push(1);
                        continue 'nodeloop;
                    }

                    match history.get(&curr_stack) {
                        Some(value) => {
                            values.push(*value);
                            continue 'nodeloop;
                        }
                        None => {}
                    }

                    let grid_graph = Graph::from(new_grid.clone());
                    for graph in &graph_history {
                        // todo: measure if actually faster
                        if is_isomorphic(&grid_graph, graph) {
                            // todo: examine diff with/without matching
                            continue 'nodeloop;
                        }
                    }
                    graph_history.push(grid_graph);

                    if dist_level == level + 1 {
                        let name = format!("progress.{}.{}.json", level, curr_stack);
                        let fs = File::create(name.clone()).unwrap();
                        serde_json::to_writer(
                            fs,
                            &(
                                &BoardStateRaw::from(BoardState::new(new_grid)),
                                self.grid.clone(),
                            ),
                        )
                        .expect(format!("Failed to serialize {}!", name).as_str());
                        continue 'nodeloop;
                    }

                    let subgraph_value = BoardState::new(new_grid).calculate(
                        level + 1,
                        dist_level,
                        curr_stack,
                        history,
                    );

                    if let Some(unwrapped_value) = subgraph_value {
                        history.insert(curr_stack, unwrapped_value);
                        values.push(sum_value);
                    }
                }
            }

            sum_value ^= mex(values);
        }


        if dist_level == level + 1 {
            return None;
        }

        Some(sum_value)
    }
}

fn remove_node(
    mut grid: StableGraph<u64, (), Undirected>,
    node: &mut NodeIndex,
    stack: u64
) -> (u64, StableGraph<u64, (), Undirected>) {
    let mut walker = grid.neighbors(*node).detach();
    let mut curr_stack = stack.clone();
    while let Some((_edge, neighbor)) = walker.next(&grid) {
        // todo: check if duplicate node
        // contract
        match grid.remove_node(neighbor) {
            Some(hash) => {curr_stack ^= hash},
            None => {}
        };
    }

    match grid.remove_node(*node) {
        Some(hash) => {curr_stack ^= hash},
        None => {}
    };

    (curr_stack, grid)
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

    min
}

fn connected_subgraphs(graph: StableGraph<u64, (), Undirected>, stack: u64) -> Vec<(StableGraph<u64, (), Undirected>, u64)> {
    let mut vertex_sets = UnionFind::new(graph.node_bound());
    for edge in graph.edge_indices() {
        if let Some((source, target)) = graph.edge_endpoints(edge) {
            vertex_sets.union(graph.to_index(source), graph.to_index(target));
        }
        // union the two vertices of the edge
    }

    // Finding size/number of connected subgraphs
    let labels = vertex_sets.into_labeling();
    let mut deduped_labels = labels.clone();
    deduped_labels.dedup();
    let size = deduped_labels.len();

    if size > 1 {
        let mut subgraphs = Vec::<(StableGraph<u64, (), Undirected>, u64)>::with_capacity(size);
        for label in deduped_labels {
            // println!("{}, {:?}", size, labels);
            let mut dead = 0;
            let mut result_g = StableGraph::<u64, (), Undirected>::default();

            for node in graph.node_indices() {
                if let Some(node_weight) = graph.node_weight(node) {
                    if labels[node.index()] == label {
                        result_g.add_node(*node_weight);
                    } else {
                        // println!("weight: {}, node: {:?}, label: {}", node_weight, node, label);
                        dead ^= node_weight;
                    }
                }
            }
            // println!("done, dead: {}, size:{}", dead, size);
            for edge in graph.edge_indices() {
                if let Some((source, target)) = graph.edge_endpoints(edge) {
                    if result_g.contains_node(source) && result_g.contains_node(target) {
                        result_g.add_edge(source, target, ());
                    }
                }
            }
            subgraphs.push((result_g, dead^stack)) // alive^total should be all the dead nodes (aka the stack)
        }
        subgraphs
    } else if size == 1 {
        vec![(graph, stack)]
    } else {
        println!("size: {}, labels: {:?}, graph: {:?}", size, labels, graph);
        panic!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{instantiate_grid, mex, remove_node, BoardState};
    use std::iter::FromIterator;
    use test::Bencher;

    #[test]
    fn test_grid_2() {
        let (grid, _grid_tracker) = instantiate_grid(2, 2);
        assert!(grid.node_count() == 4);
        println!("{}", grid.edge_count());
        assert!(grid.edge_count() == 6);
    }

    #[test]
    fn test_grid_3() {
        let (grid, _grid_tracker) = instantiate_grid(3, 3);
        assert!(grid.node_count() == 9);
        assert!(grid.edge_count() == 28);
    }

    #[test]
    fn test_grid_many() {
        for i in 3..10 {
            for j in 3..10 {
                let (grid, _grid_tracker) = instantiate_grid(i, j);
                assert!(grid.node_count() > 0 && grid.edge_count() > 0);
            }
        }
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
        let (grid, grid_tracker) = instantiate_grid(2, 2);
        let (_, new_grid) = remove_node(grid, &mut grid_tracker[0][0].1.clone(), 0);
        assert!(new_grid.node_count() == 0);
        assert!(new_grid.edge_count() == 0);
    }

    #[test]
    fn test_remove_node_3() {
        let (grid, grid_tracker) = instantiate_grid(3, 3);

        let (_, new_grid) = remove_node(grid, &mut grid_tracker[0][0].1.clone(), 0);

        assert!(new_grid.node_count() == 2);
        assert!(new_grid.edge_count() == 1);
    }

    // end to end tests

    #[test]
    fn test_end() {
        for (size, sol) in &[(0, 0), (1, 1), (2, 1), (3, 2), (4, 1), (5, 3), (6, 1)] {
            let mut history: HashMap<u64, usize, U64Hasher> =
                HashMap::with_capacity_and_hasher(50000000, U64Hasher::new());
            let (grid, grid_tracker) = instantiate_grid(*size, *size);
            let mut state = BoardState::new(grid.clone());
            assert!(state.calculate(0, 0, 0, &mut history) == Some(*sol));
        }
    }

    #[test]
    fn test_end_multi() {
        for (size, sol) in &[(0, 0), (1, 1), (2, 1), (3, 2), (4, 1), (5, 3), (6, 1)] {
            let mut history: HashMap<u64, usize, U64Hasher> =
                HashMap::with_capacity_and_hasher(50000000, U64Hasher::new());
            let (grid, grid_tracker) = instantiate_grid(*size, *size);
            let mut state = BoardState::new(grid.clone());
            assert!(state.calculate(0, 0, 0, &mut history) == Some(*sol));
        }
    }

    #[bench]
    fn bench_5(b: &mut Bencher) {
        let mut history: HashMap<u64, usize, U64Hasher> =
            HashMap::with_capacity_and_hasher(50000000, U64Hasher::new());
        let (grid, grid_tracker) = instantiate_grid(5, 5);
        let mut state = BoardState::new(grid.clone());
        b.iter(|| state.calculate(0, 0, 0, &mut history));
    }

    #[bench]
    fn bench_6(b: &mut Bencher) {
        let (grid, grid_tracker) = instantiate_grid(6, 6);
        let mut state = BoardState::new(grid.clone());
        let mut history: HashMap<u64, usize, U64Hasher> =
            HashMap::with_capacity_and_hasher(50000000, U64Hasher::new());
        b.iter(|| state.calculate(0, 0, 0, &mut history));
    }

    #[bench]
    fn bench_mex(b: &mut Bencher) {
        let test_vec = vec![3, 3, 9, 2, 0, 9, 8, 4, 2, 7];
        b.iter(|| {
            mex(test_vec.clone());
        });
    }

    #[bench]
    fn bench_hash(b: &mut Bencher) {
        let test_vec = Vec::from_iter(0..1000);
        let mut total = 0;
        b.iter(|| {
            for int in &test_vec {
                total ^= u64hash(int);
            }
        });
    }

    fn u64hash(t: &u64) -> u64 {
        let mut s: U64Hasher = U64Hasher::new();
        t.hash(&mut s);
        s.finish()
    }

    #[test]
    fn test_u64hash() {
        println!("{}", u64hash(&1));
        assert!(u64hash(&1) == u64hash(&1));
        assert!(u64hash(&1) == 1);
        assert!(u64hash(&100) == u64hash(&100));
        assert!(u64hash(&100) == 100);
    }

    #[test]
    fn test_subgraph_detection() {
        let (grid, grid_tracker) = instantiate_grid(6, 6);
        let subgraphs = connected_subgraphs(grid, 0);
        assert!(subgraphs.len() == 1);
        // assert!(stack != 0);
    }

    #[test]
    fn test_subgraph_detection_2() { // from docs
        let mut graph = StableGraph::<u64, (), Undirected>::default();
        let a = graph.add_node(1); // node with no weight
        let b = graph.add_node(0);
        let c = graph.add_node(0);
        let d = graph.add_node(0);
        let e = graph.add_node(0);
        let f = graph.add_node(0);
        let g = graph.add_node(0);
        let h = graph.add_node(1);

        graph.extend_with_edges(&[
            (a, b),
            (b, c),
            (c, d),
            (d, a),
            (e, f),
            (f, g),
            (g, h),
            (h, e)
        ]);
        // a ----> b       e ----> f
        // ^       |       ^       |
        // |       v       |       v
        // d <---- c       h <---- g

        assert_eq!(connected_subgraphs(graph.clone(), 0).len(),2);
        assert_eq!(connected_subgraphs(graph.clone(), 0)[0].1,1);
        assert_eq!(connected_subgraphs(graph.clone(), 0)[1].1,1);
        graph.add_edge(b,e,());
        assert_eq!(connected_subgraphs(graph.clone(), 0).len(),1);
        assert_eq!(connected_subgraphs(graph.clone(), 0)[0].1,0);

    }
}
