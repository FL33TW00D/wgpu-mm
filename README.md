# wgpu-mm

How many flops can we squeeze out of wgpu?

This is basically a direct port of Bram Wasti's blog post [here](https://jott.live/markdown/webgpu_safari).
All credits to him.

## Stats

The M1 8 core GPU can supposedly hit 2.6 TFLOPS of FP32.

Our baseline from [WONNX](https://github.com/webonnx/wonnx) hits 400GFLOPS currently or ~15% of peak (`gemm.wgsl`).

In [this post](https://jott.live/markdown/m1_webgpu_perf), Bram hits ~900GFLOPS or ~35% of theoretical peak without using SIMD
group magic, so this seems like the one to beat.

Implementing the code from Bram's blogpost, we can see it holds up with approx ~860GFLOPs (using `create_shader_module_unchecked` which removes all bounds checking). With bounds checking we get 580GFLOPS.

The excellent [Webgpu-BLAS](https://github.com/milhidaka/webgpu-blas) repo gives an example of a shader that hits ~900 GFLOPs. Without bounds checking we get ~680GFLOPs.

A custom metal shader from [Tinygrad](https://github.com/geohot/tinygrad) can
hit 2000 GFLOPS or ~75% of theoretical peak. This shader uses SIMD groups which
WebGPU doesn't support yet - but it's been proposed a few times e.g [here](https://github.com/gpuweb/gpuweb/issues/3950).

## Read More 

[NVIDIA Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#gpu-perf)
