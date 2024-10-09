#!/bin/bash

function get_minimum {
min=$1
for i in $@; do
    if [ 1 -eq "$(echo "$min > $i" | bc)" ]
    then
        min=$i
    fi
done
echo $min
}


make;

for size in 150 300 500 750; do

    time1=$(./prog -size $size -iters 200 -driver CPU | grep "Jacobi Time" | tr -s '  ' ' ' | cut -d ' ' -f5)
    time2=$(./prog -size $size -iters 200 -driver CPU | grep "Jacobi Time" | tr -s '  ' ' ' | cut -d ' ' -f5)
    time3=$(./prog -size $size -iters 200 -driver CPU | grep "Jacobi Time" | tr -s '  ' ' ' | cut -d ' ' -f5)

    min=$(get_minimum $time1 $time2 $time3)

    echo $min >> log.cpu
done


for size in 150 300 500 750; do

    time1=$(./prog -size $size -iters 200 -driver GPU | grep "Jacobi Time" | tr -s '  ' ' ' | cut -d ' ' -f5)
    time2=$(./prog -size $size -iters 200 -driver GPU | grep "Jacobi Time" | tr -s '  ' ' ' | cut -d ' ' -f5)
    time3=$(./prog -size $size -iters 200 -driver GPU | grep "Jacobi Time" | tr -s '  ' ' ' | cut -d ' ' -f5)

    min=$(get_minimum $time1 $time2 $time3)

    echo $min >> log.gpu
done



