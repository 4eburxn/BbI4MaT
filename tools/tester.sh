#!/bin/bash
mkdir -p /tmp/ROFLS/
for i in $(seq 1000 600 11000); do
    python $1 "$i" > /tmp/ROFLS
	cat /tmp/ROFLS | $2 >> "test10000" 2>&1
	cat /tmp/ROFLS | $3 >> "test10000" 2>&1
done
