# Builds ./craftax with whatever toolchains are available:
#   nvcc found   -> unified binary, both backends (--backend cuda|cpu)
#   nvcc missing -> CPU-only binary, same CLI (bench mode only)
CC        := gcc
NVCC      := $(shell command -v nvcc 2>/dev/null)
NVCCFLAGS ?= -O3 -arch=native --expt-relaxed-constexpr --use_fast_math
CPUFLAGS  ?= -O3 -march=native -mtune=native -ffast-math -fno-math-errno \
             -funroll-loops -fopenmp

ifneq ($(NVCC),)
craftax: main.cu craftax.cu craftax_cpu.o
	$(NVCC) $(NVCCFLAGS) main.cu craftax_cpu.o -o $@ -Xcompiler -fopenmp -lpthread

craftax_cpu.o: craftax.c
	$(CC) $(CPUFLAGS) -c craftax.c -o $@
else
craftax: craftax.c
	$(CC) $(CPUFLAGS) -DCRAFTAX_STANDALONE craftax.c -o $@ -lpthread -lm
endif

clean:
	rm -f craftax craftax_c craftax_cpu.o

.PHONY: clean
