#!/bin/bash
mkdir -p /tmp/ROFLS
for i in $(seq 1000 500 10200); do
    python $1 "$i" > /tmp/ROFLS/matrix
	cat /tmp/ROFLS/matrix | $2 >> "test10000" 2>&1
	cat /tmp/ROFLS/matrix | $3 >> "test10000" 2>&1
done
for i in $(seq 3 1 1020); do
    python $1 "$i" > /tmp/ROFLS/matrix
	cat /tmp/ROFLS/matrix | $2 >> "test1000" 2>&1
	cat /tmp/ROFLS/matrix | $3 >> "test1000" 2>&1
done
