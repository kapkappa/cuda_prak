#!/bin/bash

comb="0 0 0"
min="999"

for x in 1 2 4 8 16 32; do
    for y in 1 2 4 8 16 32; do

        make BS="-DX_BLOCKSIZE=${x} -DY_BLOCKSIZE=${y}" --silent

        time1=$(./prog -driver GPU -size 384 | grep "Time" | cut -d ' ' -f18)

        echo $x $y
        echo $time1

        if [ 1 -eq "$(echo "$min > $time1" | bc)" ]
        then
            comb=$(echo $x $y)
            min=$time1
        fi

    done
done

echo $comb $min
