[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/eoBxOH0K)

# Q2 Image Colorization CNN - Analytical Solutions

### Architecture Parameters
- **Input spatial size**: 32×32  
- **NIC** = Input channels (typically 1 for grayscale)  
- **NF** = Number of filters in first conv layer  
- **NC** = Output channels (24 color centroids)  
---

## Number of Weights (Ignoring Biases and BN)

Each layer’s weights =  
`Input_Channels × Output_Channels × Kernel_Height × Kernel_Width`

| Layer | Formula | Weights |
|:------|:---------|:--------|
| Conv1 | NIC × NF × 3 × 3 | **9 NIC NF** |
| Conv2 | NF × 2NF × 3 × 3 | **18 NF²** |
| Conv3 | 2NF × 4NF × 3 × 3 | **72 NF²** |
| Deconv1 | 4NF × 2NF × 2 × 2 | **32 NF²** |
| Deconv2 | 2NF × NF × 2 × 2 | **8 NF²** |
| Deconv3 | NF × NC × 2 × 2 | **4 NF NC** |
| Conv4 | NC × NC × 1 × 1 | **NC²** |

### Total Weights
`W_total = 9*NIC*NF + 130*NF^2 + 4*NF*NC + NC^2`

Weights depend only on channels and kernels — **unchanged for 64×64 input**.

---

## Number of Outputs (Activations)

Each layer’s activations =  
`Output_Channels × Height × Width`

| Layer | Output Shape | Elements |
|:------|:--------------|:----------|
| Conv1 | [NF, 16, 16] | 256 NF |
| Conv2 | [2NF, 8, 8] | 128 NF |
| Conv3 | [4NF, 4, 4] | 64 NF |
| Deconv1 | [2NF, 8, 8] | 128 NF |
| Deconv2 | [NF, 16, 16] | 256 NF |
| Deconv3 | [NC, 32, 32] | 1024 NC |
| Conv4 | [NC, 32, 32] | 1024 NC |

### Total Activations (32×32)
`A_total = 832*NF + 2048*NC`

### For 64×64 Input  
(all spatial maps ×4 larger)

`A'_total = 4(832*NF + 2048*NC) = 3328*NF + 8192*NC`

---

## Number of Connections

Each kernel is applied over all spatial positions →  
`Connections = Weights × (#output spatial positions)`

| Layer | Weights | Output Spatial Size | Connections |
|:------|:---------|:------------------|:-------------|
| Conv1 | 9 NIC NF | 32×32 / 2 / 2 = 16×16 → 256 → 1024 total | **9216 NIC NF** |
| Conv2 | 18 NF² | 8×8 → 64 → 256 total | **4608 NF²** |
| Conv3 | 72 NF² | 4×4 → 16 → 64 total | **4608 NF²** |
| Deconv1 | 32 NF² | 4×4 → 64 → 64 total | **2048 NF²** |
| Deconv2 | 8 NF² | 8×8 → 256 | **2048 NF²** |
| Deconv3 | 4 NF NC | 32×32 → 1024 | **4096 NF NC** |
| Conv4 | NC² | 32×32 → 1024 | **1024 NC²** |

### Total Connections (32×32)
`C_total = 9216*NIC*NF + 13312*NF^2 + 4096*NF*NC + 1024*NC^2`

### For 64×64 Input  
(spatial size doubled → connections ×4)

`C'_total = 36864*NIC*NF + 53248*NF^2 + 16384*NF*NC + 4096*NC^2`

---

## Final Summary

| Quantity | 32×32 Input | 64×64 Input | Scaling |
|:----------|:-------------|:-------------|:---------|
| **Weights** | 9NICNF + 130NF² + 4NFNC + NC² | same | 1× |
| **Activations** | 832NF + 2048NC | 3328NF + 8192NC | 4× |
| **Connections** | 9216NICNF + 13312NF² + 4096NFNC + 1024NC² | 36864NICNF + 53248NF² + 16384NFNC + 4096NC² | 4× |

---

## Assumptions

- BatchNorm and biases are **ignored** (per assignment).  
- `padding = K//2`, `stride=2` for pooling/upsampling.  

---

