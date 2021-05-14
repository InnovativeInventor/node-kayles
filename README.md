## non-attacking-queens
[![Rust tests](https://github.com/InnovativeInventor/non-attacking-queens/actions/workflows/test.yaml/badge.svg)](https://github.com/InnovativeInventor/non-attacking-queens/actions/workflows/test.yaml)

Some calculations for the [non attacking queens](https://www.maa.org/sites/default/files/may_2006_-_noon55524.pdf) problem using a graph-theoretic node-contraction algorithim.

## Building and Running (with release)
To run and build:
```
bash release.sh
./target/release/non-attacking-queens -n $M -m $M
```
where `$N` and `$M` represent the `$Nx$M` grid to calculate.

To plot the graph, run:
```
python format.py
```
then paste the dictionary that you want to plot into stdin.

## Calculating Sequence (for reproducing A344227)
```
./target/release/non-attacking-queens --seq
```
