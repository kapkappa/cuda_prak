# Отчёт по 1 заданию (jacobi3d)

## Local workstation
8-cores Intel Core i7-11700, 2-channel DDR4 2933GHz, NVIDIA GeForce RTX 3060ti (8 Gb)

The actual memory bus bandwidth for the CPU is 43 GB/s, and for the GPU is 417 GB/s.

| Size (side/Mb) | CPU      | GPU       | speedup   |
|     :----:     | :----:   | :----:    | :----:    |
|  150 / 0.05 Gb | 0.841    | 0.106     | 7.95      |
|  300 / 0.40 Gb | 7.233    | 0.778     | 9.29      |
|  500 / 1.86 Gb | 35.091   | 3.551     | 9.88      |
|  750 / 6.29 Gb | 131.230  | 11.769    | 11.15     |

The times are given for 200 iterations without taking into account the calculation of the residual.

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

> Speedup is 25 times
