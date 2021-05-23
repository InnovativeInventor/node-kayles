## Node-Kayles Sprague-Grundy Value Calculator
[![Rust tests](https://github.com/InnovativeInventor/node-kayles/actions/workflows/test.yaml/badge.svg)](https://github.com/InnovativeInventor/node-kayles/actions/workflows/test.yaml)

Some calculations for the [non attacking queens](https://www.maa.org/sites/default/files/may_2006_-_noon55524.pdf) problem using a graph-theoretic node-contraction algorithm (aka Node-Kayles).

To run and build (with release):
```
bash release.sh
./target/release/node-kayles [args]
```

## Results
The new terms for the following [OEIS](https://oeis.org/) sequences are shown below, along with steps to reproduce.
All ***bolded and italicized sequence terms*** are new/novel results that were previously unknown.

### Non-Attacking Queens Sequence (OEIS [A344227](https://oeis.org/draft/A344227))
```
bash release.sh
bash scripts/queens.sh
```
Sequence (where the first term is a 0x0 chessboard): 0, 1, 1, 2, 1, 3, 1, 2, 3, 1, 0, ***1, 0, 1***

In addition to the optimized Rust implementation, a naive Haskell implementation can be found at `haskell-impls/queens.hs`).

### Generalized Petersen Sequence (OEIS [A316533](https://oeis.org/A316533))
```
bash release.sh
bash scripts/petersen_n_2.sh
```
*Note: this script will skip every other term since it has been proven that for all even values, the Sprague-Grundy value is 0.*
Sequence (where the first term is P(5,2)): 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, ***0***

### 3xN Lattice Sequence (OEIS [A316632](https://oeis.org/A316632))
```
bash release.sh
bash scripts/lattice_3_n.sh
```
Sequence (where the first term is the 3x1 lattice): 2, 1, 1, 0, 3, 3, 2, 2, 2, 3, 3, 5, 2, ***4, 1, 3***

In addition to the optimized Rust implementation, a naive Haskell implementation can be found at `haskell-impls/lattice.hs`).

## License
This project is licensed under `GPL-3.0-only`.
Pull requests, feedback, and contributions are always welcome.
