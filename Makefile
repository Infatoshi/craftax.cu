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
	$(NVCC) $(NVCCFLAGS) main_classic.cu craftax_classic_cpu.o -o $@ -Xcompiler -fopenmp -lpthread

craftax_classic_cpu.o: craftax_classic.c
	$(CC) $(CPUFLAGS) -c craftax_classic.c -o $@
else
craftax_classic: craftax_classic.c
	$(CC) $(CPUFLAGS) -DCRAFTAX_STANDALONE craftax_classic.c -o $@ -lpthread -lm
endif

craftax_full: craftax_full.c
	$(CC) $(CPUFLAGS) craftax_full.c -o $@ -lpthread -lm

clean:
	rm -f craftax_classic craftax_full craftax_classic_cpu.o

.PHONY: all classic full clean
