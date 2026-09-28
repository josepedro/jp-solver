# JP Solver: build the 3D Taylor-Green vortex solvers (uniform reference and 3-level AMR).
# Requires the CUDA toolkit (nvcc). Tested with CUDA 12.0 on an RTX 2060 (sm_75).

NVCC    ?= nvcc
ARCH    ?= sm_75
MB      ?= 8
NVFLAGS  = -O3 -arch=$(ARCH) -DMB=$(MB)

# The AMR solver needs a higher register cap: 128 avoids spills in the prep and
# prolongation kernels, which otherwise halve throughput.
AMRFLAGS = $(NVFLAGS) -maxrregcount=128

all: amr uniform

amr: src/tgv3d_amr.cu
	$(NVCC) $(AMRFLAGS) $< -o $@

uniform: src/tgv3d_uniform.cu
	$(NVCC) $(NVFLAGS) $< -o $@

clean:
	rm -f amr uniform

.PHONY: all clean
