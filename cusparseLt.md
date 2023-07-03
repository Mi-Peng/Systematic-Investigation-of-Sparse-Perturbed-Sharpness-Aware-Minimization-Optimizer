

# Introduction

NVIDIA cuSPARSELt is a high-performance CUDA library dedicated to general matrix-matrix operations in which at least one operand is a sparse matrix, which is used in our paper. 

[[Official Docs]](https://docs.nvidia.com/cuda/cusparselt/index.html) &nbsp;&nbsp;&nbsp;&nbsp;
[[Official Download]](https://developer.nvidia.com/cusparselt-downloads) &nbsp;&nbsp;&nbsp;&nbsp;
[[Official Github]](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSPARSELt)

# Installation

> This file provides instructions for installing the [cusparseLt 0.2.0](https://docs.nvidia.com/cuda/cusparselt/index.html) library, which worked on my device, and setup it for using in Python.

1. Firstly, install cusparselt from anaconda
```bash
conda install cusparselt -c conda-forge -y
```

Or you can click [Official Download cusparseLt 0.2.0](https://developer.nvidia.com/cusparselt/downloads/v0.2.0) and choose the target platform.

3. Install `spmm`
```bash
python setup.py install
```

4. Check the hardware(only support cusparseLt on NVIDIA Ampere, *e.g.*, A100, H100.)
```bash
python cspmm/test.py
```
and normally, it should return the result "HARDWARE PASSED"



**Please note that the library `cusparselt` is updated frequently, but this guide is still valid for now (Last Update: 2023 July 3th ).**


# Instructions
We calculate the matrxi multiplication like:
$$
C = A * B 
$$
where $A$ is a sparse matrix.

Always import the spmm repo after torch.

Always use spmm for the tensor which is on CUDA and contiguous.

- `initSpmmNum(int num)`: Allocate a series of memory for subsequent sparse-matrix-multiplication.

```python
import torch
import spmm
spmm.initSpmmNum(4)
```

- `checkCusparseLt()`: Check if hardware supports cusparselt.
```python
import torch
import spmm
spmm.checkCusparseLt() # it should return 0
```

- `initSpmmDescriptor(int index, int num_batches, int num_A_rows, int num_A_cols, int lda, int num_B_rows, int num_B_cols, int ldb, int num_C_rows, int num_C_cols, int ldc)`: Init the Sparse Matrix Descriptor. Function locates the memory by `index`, and normally `lda` is equals to `num_A_cols`.

```python
import torch
import spmm
spmm.initSpmmDescriptor(0, 128, 64, 64, 64, 64, 64, 64, 64, 64, 64)
```

- `pruneMatrix(int index, float* original_matrix, float* prunned_matrix)`: Prune the Dense Matrix to a Sparse Matrix automatically.

```python
import torch
import spmm
A = torch.rand(64, 64).cuda().contiguous()
A_prunned = torch.rand(64, 64).cuda().contiguous()
spmm.pruneMatrix(0, A, A_prunned)
```

- `checkPrunned(int index, float* A_prunned)`: Check if Sparse Matrix meets Structural Requirements. (See more in https://docs.nvidia.com/cuda/cusparselt/types.html#cusparseltprunealg-t. Note that 2:4 structure only works for data whose type is `half`, `bfloat16` or `int8`, so you need to change the type of input in [cspmm/cspmm_imple.cpp](cspmm/cspmm_imple.cpp) from `float` to `half` or use template function.)

```python
import torch
import spmm
mask = ... # 1:2 or 2:4
A = ...
A_prunned = (A * mask).contiguous()
spmm.checkPrunned(0, A_prunned)
```

- `compressMatrix(int index, float* A_prunned)`: Compress sparse matrix.
```python
import torch
import spmm
...
spmm.compressMatrix(0, A_prunned)
```


- `spmm(int index, float* dB, float* dC)`: Perform sparse matrix multiplication, where dB is another matrix and dC is the results.

```python
import torch
import spmm
...
spmm.compressMatrix(0, B, C)
```

Here is a pseudo code for using spmm:
```python
import torch
import spmm
spmm.checkCusparseLt()
spmm.initSpmmNum(n) # suppose you have n nn.Linear
for i in range(n):
    spmm.initSpmmDescriptor(i, ...) 
    mask = cal_mask(...) # suppose you get the mask from your own function manually 
    A_prunned = (A * mask).cuda().contiguous()
    spmm.checkPrunned(i, A_prunned)
    spmm.compressMatrix(i, A_prunned)

while "(receive a input)":
    spmm.spmm(...)
```
