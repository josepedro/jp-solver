# Performance

Measured on an RTX 2060 6 GB, sm_75, CUDA 12.0, default clocks, FP32,
`nvcc -O3 -arch=sm_75 -DMB=8 -maxrregcount=128`. Work is counted as actual lattice updates per
coarse step, `(cnb0 + 2*nf1 + 4*nf2) * MB^3`, so the fine levels are counted with their subcycle
multiplier and MLUPS is normalised by real work. The register cap of 128 matters: at 96 the prep
and prolongation kernels spill and throughput halves.

## Solver step throughput (bench, no diagnostics or output)

| version | MLUPS (work) |
|---|---|
| baseline (scattered per-face, correctness first) | 31.6 |
| ghost caching (fill once per frozen parent state) | 59.8 |
| register cap 128 (was 96, spilled) | 104.7 |
| amortised regularisation (prep per parent cell) | 437.6 |
| current (skin-prep, default buffer 0) | ~450 |

The uniform 192^3 reference reaches 1342.8 MLUPS, 95 percent of the D3Q19 DRAM ceiling on this
card. The optimised AMR runs at about 33 percent of that ceiling. Differential timing shows the
coarse-fine interface (prep plus ghost fill) is 65 percent of the step, restriction 10 percent, and
the actual update 35 percent. The step itself is efficient; the interface dominates and is largely
intrinsic to scattered refinement.

## AMR vs uniform, same physical time

Dynamic case: coarse 48, fine 192, D3Q19, sensor on both levels, at matched non-dimensional time.

| metric | AMR 3-level | uniform 192^3 | ratio |
|---|---|---|---|
| wall-clock | 3.3 s | 12.7 s | AMR 3.8x faster |
| pool memory | 0.47 GB | 1.08 GB | AMR 2.3x less |
| work (lattice updates) | 1.46e9 | 1.70e10 | AMR 11.7x less |

A refinement buffer or dilation raises per-kernel MLUPS but explodes the refined volume on the
space-filling Taylor-Green vortex, making the wall-clock slower overall, so the default buffer is
zero. It pays only for localized features.

## GPU-native adaptation

The adaptation does the threshold (a sort in VRAM), marking, scan, scatter, deep-interior mask and
skin lists on the device; only scalar counts return to the host. The result is bit-identical to a
host reference path. The wall-clock is unchanged because the regrid is not the bottleneck; the value
is the zero host round-trip, which is what makes frequent regrids and multi-GPU cheap.

## Device-side diagnostics

The per-sample full-state copy to the host plus two host-side loops were replaced by a device
composite kernel and an atomic reduction that returns three doubles. For a dynamic run with
diagnostics (about 175 samples): wall-clock 106.96 s to 13.22 s, an 8.1x speedup, and host RAM
289 MB to 109 MB. The solver-only bench is unchanged; the win is the diagnostic overhead, which the
host version dominated.
