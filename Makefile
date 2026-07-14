NVCC      ?= nvcc
CC        ?= gcc
NVCCFLAGS ?= -O3 -arch=native --expt-relaxed-constexpr --use_fast_math
CFLAGS    ?= -O3 -march=native -mtune=native -ffast-math -fno-math-errno \
             -funroll-loops -flto -fopenmp

all: cuda cpu

cuda: craftax
craftax: main.cu craftax.cu
	$(NVCC) $(NVCCFLAGS) main.cu -o $@

cpu: craftax_c
craftax_c: craftax.c
	$(CC) $(CFLAGS) craftax.c -o $@ -lpthread -lm

clean:
	rm -f craftax craftax_c

.PHONY: all cuda cpu clean
