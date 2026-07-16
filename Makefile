# Grid of Craftax implementations: {classic, full} x {C, CUDA}.
#   make classic  -> ./craftax_classic  (unified CUDA+CPU binary; CPU-only without nvcc)
#   make full     -> ./craftax_full     (CPU; CUDA port pending)
#   make          -> everything buildable with the toolchains present
CC        := gcc
NVCC      := $(shell command -v nvcc 2>/dev/null)
NVCCFLAGS ?= -O3 -arch=native --expt-relaxed-constexpr --use_fast_math
CPUFLAGS  ?= -O3 -march=native -mtune=native -ffast-math -fno-math-errno \
             -funroll-loops -fopenmp

all: classic full
classic: craftax_classic
full: craftax_full

ifneq ($(NVCC),)
craftax_classic: main_classic.cu craftax_classic.cu craftax_classic_cpu.o
	$(NVCC) $(NVCCFLAGS) main_classic.cu craftax_classic_cpu.o -o $@ -Xcompiler -fopenmp -lpthread -lcublas

craftax_classic_cpu.o: craftax_classic.c
	$(CC) $(CPUFLAGS) -c craftax_classic.c -o $@

# Full-game CUDA port (M2: split step/reset/encode kernels, lazy floors,
# warp-cooperative worldgen, coalesced obs encode). -fmad=false is required:
# exact IEEE per-op float semantics make the GPU trajectory hash bit-identical
# to the gcc -ffast-math C build (verified 64x2000 and 4x20000 seed 42
# anchors). fmad-on measured no faster, so the exact build is the only one.
NVCCFLAGS_FULL ?= -O3 -arch=native --expt-relaxed-constexpr -fmad=false
full: craftax_full_cuda
craftax_full_cuda: craftax_full.cu
	$(NVCC) $(NVCCFLAGS_FULL) craftax_full.cu -o $@

# Compact byte observations (996B vs 3372B per env): same trajectories, its
# own hash universe -- matches the C -DCRAFTAX_COMPACT_OBS build bit-exactly
# (64x2000 seed 42 => 0x4fb32c98731b75e8, 4x20000 => 0x7d6c02f20a72e8f9).
full-compact: craftax_full_cuda_compact
craftax_full_cuda_compact: craftax_full.cu
	$(NVCC) $(NVCCFLAGS_FULL) -DCRAFTAX_COMPACT_OBS craftax_full.cu -o $@

# Hidden-128 policy build (default hidden size is 32). Env trajectories
# (hash/statehash anchors) are hidden-independent; runhash/runverify/train
# numbers live in their own universe (see README).
full-h128: craftax_full_cuda_h128
craftax_full_cuda_h128: craftax_full.cu
	$(NVCC) $(NVCCFLAGS_FULL) -DCRAFTAX_HIDDEN=128 craftax_full.cu -o $@
else
craftax_classic: craftax_classic.c
	$(CC) $(CPUFLAGS) -DCRAFTAX_STANDALONE craftax_classic.c -o $@ -lpthread -lm
endif

craftax_full: craftax_full.c
	$(CC) $(CPUFLAGS) craftax_full.c -o $@ -lpthread -lm

clean:
	rm -f craftax_classic craftax_full craftax_full_cuda craftax_classic_cpu.o \
	      craftax_full_cuda_compact craftax_full_cuda_h128

.PHONY: all classic full full-compact full-h128 clean
