

# Introduction

NVIDIA cuSPARSELt is a high-performance CUDA library dedicated to general matrix-matrix operations in which at least one operand is a sparse matrix, which is used in our paper. 

[[Official Docs]](https://docs.nvidia.com/cuda/cusparselt/index.html) &nbsp;&nbsp;&nbsp;&nbsp;
[[Official Download]](https://developer.nvidia.com/cusparselt-downloads) &nbsp;&nbsp;&nbsp;&nbsp;
[[Official Github]](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSPARSELt)

# Installation

> This file provides instructions for installing the [cusparseLt 0.2.0](https://docs.nvidia.com/cuda/cusparselt/index.html) library, which worked on my device.

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
