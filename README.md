# wgpu-mm

How many flops can we squeeze out of wgpu?

This is basically a direct port of Bram Wasti's blog post [here](https://jott.live/markdown/webgpu_safari).
All credits to him.

## Stats

The M1 8 core GPU can supposedly hit 2.6 TFLOPS of FP32.

Our baseline is 400GFLOPS currently or ~15% of peak theoretical. This kernel is `gemm.wgsl`.

In [this post](https://jott.live/markdown/m1_webgpu_perf), Bram his ~900GFLOPS or ~35% of theoretical peak without using SIMD
group magic, so this seems like the one to beat.

Implementing the code from Bram's blogpost, we can see it holds up with approx ~860GFLOPs (using `create_shader_module_unchecked` which removes all bounds checking). With bounds checking we get 580GFLOPS.

A custom metal shader from [Tinygrad](https://github.com/geohot/tinygrad) can
hit 1800 GFLOPS or ~70% of theoretical peak. This shader uses SIMD groups which
WebGPU doesn't support yet - but it's been proposed a few times e.g [here](https://github.com/gpuweb/gpuweb/issues/3950).

## Read More 

[NVIDIA Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#gpu-perf)
