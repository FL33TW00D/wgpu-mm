//Naive matrix multiplication
//https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/1_naive.cuh
@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> B: array<f32>;

@group(0) @binding(2)
var<storage, read_write> C: array<f32>;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let M = {{ M }}u;
    let N = {{ N }}u;
    let K = {{ K }}u;
    let x = global_id.x;
    let y = global_id.y;
    if (x < M && y < N) {
        var tmp = 0f;
        for (var i = 0u; i < K; i = i + 1u) {
          tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = tmp; 
    }
}
