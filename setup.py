from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name = 'spmm',
    version='0.0.1',
    ext_modules=[
        CppExtension(
            name='spmm',
            sources=['cspmm/cspmm.cpp', 'cspmm/cspmm_imple.cpp'],
            libraries=['cusparseLt', 'cusparse']
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    # packages=['spmm']
)