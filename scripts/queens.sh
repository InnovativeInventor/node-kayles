#!/bin/bash
for i in {0..100}
do
	echo "Calculating value for queens (n=$i)"
	python tools/queens.py --n $i
	./target/release/node-kayles -sr queens.json
done
