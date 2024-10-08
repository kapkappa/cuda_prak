#!/bin/bash

rm -f logs/log.err log.out

#GPU
if [ 1 -eq 0 ]; then
    bsub -J "jacobi3d" -n 1 -m polus-c3-ib -o logs/log.out -eo logs/log.err -x -gpu "num=1:mode=exclusive_process" ./prog -size 150 -driver GPU -iters 200
    bsub -J "jacobi3d" -n 1 -m polus-c3-ib -o logs/log.out -eo logs/log.err -x -gpu "num=1:mode=exclusive_process" ./prog -size 300 -driver GPU -iters 200
    bsub -J "jacobi3d" -n 1 -m polus-c3-ib -o logs/log.out -eo logs/log.err -x -gpu "num=1:mode=exclusive_process" ./prog -size 500 -driver GPU -iters 200
    bsub -J "jacobi3d" -n 1 -m polus-c3-ib -o logs/log.out -eo logs/log.err -x -gpu "num=1:mode=exclusive_process" ./prog -size 750 -driver GPU -iters 200
    bsub -J "jacobi3d" -n 1 -m polus-c3-ib -o logs/log.out -eo logs/log.err -x -gpu "num=1:mode=exclusive_process" ./prog -size 1000 -driver GPU -iters 200
fi

#CPU
if [ 1 -eq 0 ]; then
    bsub -J "jacobi3d" -n 1 -m polus-c3-ib -o logs/log.out -eo logs/log.err -x -gpu "num=1:mode=exclusive_process" ./prog -size 150 -driver CPU -iters 200
    bsub -J "jacobi3d" -n 1 -m polus-c3-ib -o logs/log.out -eo logs/log.err -x -gpu "num=1:mode=exclusive_process" ./prog -size 300 -driver CPU -iters 200
    bsub -J "jacobi3d" -n 1 -m polus-c3-ib -o logs/log.out -eo logs/log.err -x -gpu "num=1:mode=exclusive_process" ./prog -size 500 -driver CPU -iters 200
    bsub -J "jacobi3d" -n 1 -m polus-c3-ib -o logs/log.out -eo logs/log.err -x -gpu "num=1:mode=exclusive_process" ./prog -size 750 -driver CPU -iters 200
    bsub -J "jacobi3d" -n 1 -m polus-c3-ib -o logs/log.out -eo logs/log.err -x -gpu "num=1:mode=exclusive_process" ./prog -size 1000 -driver CPU -iters 200
fi

