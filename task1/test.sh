#!/bin/bash

comb="0 0 0"
min="999"

for x in 4 8 16 32; do
    for y in 1 2 4 8; do
        for z in 1 2 4; do

            make BS="-DX_BLOCKSIZE=${x} -DY_BLOCKSIZE=${y} -DZ_BLOCKSIZE=${z}" --silent

            time1=$(./prog -driver GPU -size 500 -iters 200 | grep "Jacobi Time" | tr -s '  ' ' ' | cut -d ' ' -f5)
#            echo $x $y $z
#            echo $time1

            if [ 1 -eq "$(echo "$min > $time1" | bc)" ]
            then
                comb=$(echo $x $y $z)
                min=$time1
            fi

        done
    done
done

echo $comb $min
