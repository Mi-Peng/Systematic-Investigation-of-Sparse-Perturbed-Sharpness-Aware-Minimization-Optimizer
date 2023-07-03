import torch
import spmm

if spmm.checkCusparseLt() == 0:
    print('HARDWARE PASSED')
