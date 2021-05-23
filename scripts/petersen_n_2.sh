#!/bin/bash
for i in {2..100}
do
	n=$(((i * 2) + 1))
	echo "Calculating Petersen P($n,2)"
	python tools/petersen.py --n $n
	./target/release/node-kayles -sr petersen.json
done
