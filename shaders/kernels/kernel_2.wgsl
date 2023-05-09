//Kernel 2: Global Memory Coalescing
//https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/2_kernel_global_mem_coalesce.cuh
@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> B: array<f32>;

@group(0) @binding(2)
var<storage, read_write> C: array<f32>;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let M = {{ M }}u;
    let N = {{ N }}u;
    let K = {{ K }}u;
    let cRow = group_id.x * 16u + (local_id.x / 16u);
    let cCol = group_id.y * 16u + (local_id.x % 16u);
    if (cRow < M && cCol < N) {
        var tmp = 0f;
        for (var i = 0u; i < K; i = i + 1u) {
          tmp += A[cRow * K + i] * B[i * N + cCol];
        }
        C[cRow * N + cCol] = tmp;
    }
}
