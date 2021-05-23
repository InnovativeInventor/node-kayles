#!/bin/bash
for i in {0..20}
do
	echo "Calculating 3 X $i"
	python tools/lattice.py --n 3 --m $i
	./target/release/node-kayles -sr lattice.json
done
