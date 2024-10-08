# Отчёт по 1 заданию (jacobi3d)

## Local workstation
8-cores Intel Core i7-11700, 2-channel DDR4 2933GHz, NVIDIA GeForce RTX 3060ti (8 Gb)

The actual memory bus bandwidth for the CPU is 43 GB/s, and for the GPU is 417 GB/s.

1. size = 150 (50 Mb)
> CPU to GPU execition time: 0.953228 to 0.141007 per 200 iterations

> Speedup is 6.76 times
2. size = 300 (400 Mb)
> CPU to GPU execition time: 7.848086 to 1.023979 per 200 iterations

> Speedup is 7.6 times
3. size = 500 (1.86 Gb) 
> CPU to GPU execution time: 36.874870 to 4.592491 per 200 iterations

> Speedup is 8 times
4. size = 750 (6.29 Gb)
> CPU to GPU execution time: 141.973732 to 15.452016 per 200 iterations

> Speedup is 9.2 times

## Polus node
2x 10-cores IBM POWER8s, NVIDIA Tesla P100 GPU (16 Gb)

1. size = 150 (50 Mb)
> CPU to GPU execition time: 1.594155 to 0.083022 per 200 iterations

> Speedup is 19.2 times
2. size = 300 (400 Mb)
> CPU to GPU execition time: 11.558987 to 0.578239 per 200 iterations

> Speedup is 20 times
3. size = 500 (1.86 Gb) 
> CPU to GPU execution time: 59.464701 to 2.840264 per 200 iterations

> Speedup is 21 times
4. size = 750 (6.29 Gb)
> CPU to GPU execution time: 221.982260 to 9.543416 per 200 iterations

> Speedup is 23.2 times
5. size = 1000 (14.9 Gb)
> CPU to GPU execution time: 552.025841 to 22.211708 per 200 iterations

> Speedup is 10 times
