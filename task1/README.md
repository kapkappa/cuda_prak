# Отчёт по 1 заданию (jacobi3d)

All times are given for 200 iterations without taking into account the calculation of the residual.

## Local workstation
8-cores Intel Core i7-11700, 2-channel DDR4 2933GHz, NVIDIA GeForce RTX 3060ti (8 Gb)

The actual memory bus bandwidth for the CPU is 43 GB/s, and for the GPU is 417 GB/s.

| Size (side/Mb) | CPU      | GPU       | speedup   |
|     :----:     | :----:   | :----:    | :----:    |
|  150 / 0.05 Gb | 0.566    | 0.078     | 7.26      |
|  300 / 0.40 Gb | 4.363    | 0.581     | 7.50      |
|  500 / 1.86 Gb | 20.687   | 2.578     | 8.02      |
|  750 / 6.29 Gb | 84.767   | 8.689     | 9.76      |

## Polus node
2x 10-cores IBM POWER8s, NVIDIA Tesla P100 GPU (16 Gb)

| Size (side/Mb)  | CPU      | GPU      | speedup   |
|     :----:      | :----:   | :----:   | :----:    |
|  150 / 0.05 Gb  | 0.858    | 0.025    | 34.43     |
|  300 / 0.40 Gb  | 36.783   | 0.233    | 157.7     |
|  500 / 1.86 Gb  | 35.735   | 1.085    | 32.94     |
|  750 / 6.29 Gb  | 140.664  | 3.651    | 38.53     |
|  1000 / 14.9 Gb | 344.255  | 8.786    | 39.18     |

