from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Detect compute capability
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability()
    arch_flag = f'-arch=sm_{cc[0]}{cc[1]}'
else:
    arch_flag = '-arch=sm_86'  # default to Ampere

setup(
    name='craftax_cuda',
    ext_modules=[
        CUDAExtension(
            'craftax_cuda',
            ['craftax_ext.cu'],
            extra_compile_args={
                'nvcc': ['-O3', arch_flag, '--expt-relaxed-constexpr',
                         '-lineinfo', '--use_fast_math'],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
