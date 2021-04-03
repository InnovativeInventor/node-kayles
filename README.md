## non-attacking-queens
[![Rust tests](https://github.com/InnovativeInventor/non-attacking-queens/actions/workflows/test.yaml/badge.svg)](https://github.com/InnovativeInventor/non-attacking-queens/actions/workflows/test.yaml)

Some calculations for the [non attacking queens](https://www.maa.org/sites/default/files/may_2006_-_noon55524.pdf) problem.

## Building and Running (with release)
To run and build:
```
bash release.sh
/target/release/non-attacking-queens --size $SIZE
```
where `$SIZE` is the nxn grid to calculate.

To plot the graph, run:
```
python format.py
```
then paste the dictionary that you want to plot into stdin.

It is recommended that you increase the stack size before running on large grids. Eg:
```
ulimit -s 65532
```

