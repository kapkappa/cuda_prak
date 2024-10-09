# Отчёт по 1 заданию (jacobi3d)

All times are given for 200 iterations without taking into account the calculation of the residual.

## Local workstation
8-cores Intel Core i7-11700, 2-channel DDR4 2933GHz, NVIDIA GeForce RTX 3060ti (8 Gb)

The actual memory bus bandwidth for the CPU is 43 GB/s, and for the GPU is 417 GB/s.

| Size (side/Mb) | CPU      | GPU       | speedup   |
|     :----:     | :----:   | :----:    | :----:    |
|  150 / 0.05 Gb | 0.841    | 0.106     | 7.95      |
|  300 / 0.40 Gb | 7.233    | 0.778     | 9.29      |
|  500 / 1.86 Gb | 35.091   | 3.551     | 9.88      |
|  750 / 6.29 Gb | 131.230  | 11.769    | 11.15     |

## Polus node
2x 10-cores IBM POWER8s, NVIDIA Tesla P100 GPU (16 Gb)

| Size (side/Mb)  | CPU      | GPU       | speedup   |
|     :----:      | :----:   | :----:    | :----:    |
|  150 / 0.05 Gb  | 1.247    | 0.060     | 20.78     |
|  300 / 0.40 Gb  | 41.045   | 0.456     | 90.01     |
|  500 / 1.86 Gb  | 52.878   | 2.132     | 24.80     |
|  750 / 6.29 Gb  | 184.280  | 7.110     | 25.92     |
|  1000 / 14.9 Gb | 472.891  | 16.781    | 28.18     |

