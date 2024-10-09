#!/bin/bash

comb="0 0 0"
min="999"

for x in 4 8 16 32; do
    for y in 1 2 4 8; do
        for z in 1 2 4; do

            make BS="-DX_BLOCKSIZE=${x} -DY_BLOCKSIZE=${y} -DZ_BLOCKSIZE=${z}" --silent

#            echo $x $y $z
            time1=$(./prog -driver GPU -size 750 -iters 200 | grep "Jacobi Time" | tr -s '  ' ' ' | cut -d ' ' -f5)

            if [ 1 -eq "$(echo "$min > $time1" | bc)" ]
            then
                comb=$(echo $x $y $z)
                min=$time1
            fi

        done
    done
done

echo $comb $min
