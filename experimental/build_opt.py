"""JIT-load the optimized craftax extension."""
import os
import torch
from torch.utils.cpp_extension import load

_here = os.path.dirname(os.path.abspath(__file__))

def _arch():
    cc = torch.cuda.get_device_capability()
    return f'-arch=sm_{cc[0]}{cc[1]}'

def load_opt():
    return load(
        name='craftax_cuda_opt',
        sources=[os.path.join(_here, 'craftax_opt_ext.cu')],
        extra_cuda_cflags=['-O3', _arch(), '--expt-relaxed-constexpr', '-lineinfo', '--use_fast_math'],
        extra_include_paths=[_here],
        verbose=False,
    )

def load_opt4():
    return load(
        name='craftax_cuda_opt4',
        sources=[os.path.join(_here, 'craftax_opt4_ext.cu')],
        extra_cuda_cflags=['-O3', _arch(), '--expt-relaxed-constexpr', '-lineinfo', '--use_fast_math'],
        extra_include_paths=[_here],
        verbose=False,
    )

def load_opt3():
    return load(
        name='craftax_cuda_opt3',
        sources=[os.path.join(_here, 'craftax_opt3_ext.cu')],
        extra_cuda_cflags=['-O3', _arch(), '--expt-relaxed-constexpr', '-lineinfo', '--use_fast_math'],
        extra_include_paths=[_here],
        verbose=False,
    )

def load_opt2():
    return load(
        name='craftax_cuda_opt2',
        sources=[os.path.join(_here, 'craftax_opt2_ext.cu')],
        extra_cuda_cflags=['-O3', _arch(), '--expt-relaxed-constexpr', '-lineinfo', '--use_fast_math'],
        extra_include_paths=[_here],
        verbose=False,
    )
