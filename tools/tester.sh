#!/bin/bash

for i in $(seq 1000 600 11000); do
    python $1 "$i" | $2 >> "test10000" 2>&1
done
