# Отчёт по 2 заданию (adi3d)

All times are given for 100 iterations including the residual calculations

## Local workstation
8-cores Intel Core i7-11700, 2-channel DDR4 2933GHz, NVIDIA GeForce RTX 3060ti (8 Gb)

The actual memory bus bandwidth for the CPU is 43 GB/s, and for the GPU is 417 GB/s.

| Size (side/Mb)  | CPU      | GPU       | speedup   |
|     :-----:     | :----:   | :----:    | :----:    |
|  384 / 391.5 Mb | 15.27    | 5.787     | 2.64      |

## Polus node
2x 10-cores IBM POWER8s, NVIDIA Tesla P100 GPU (16 Gb)

| Size (side/Mb)  | CPU      | GPU      | speedup   |
|     :----:      | :----:   | :----:   | :----:    |
|  384 / 391.5 Mb | 25.79    | 8.285    | 3.11      |

