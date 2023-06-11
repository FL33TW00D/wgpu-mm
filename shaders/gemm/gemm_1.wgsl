//Naive matrix multiplication
//https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/1_naive.cuh
@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> C: array<vec4<f32>>;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let M = {{ M }}u;
    let N = {{ N }}u;
    let K = {{ K }}u;
    let cRow = global_id.x;
    let cCol = global_id.y;  
    if (cRow < M && cCol < N / 4u) {
        var tmp = vec4<f32>();
        for (var k = 0u; k < K / 4u; k = k + 1u) {
          let a = A[cRow * K / 4u + k];
          tmp += vec4<f32>(a.x) * B[k * N + cCol]; 
          tmp += vec4<f32>(a.y) * B[k * N + cCol + (1u * N/4u)]; 
          tmp += vec4<f32>(a.z) * B[k * N + cCol + (2u * N/4u)];
          tmp += vec4<f32>(a.w) * B[k * N + cCol + (3u * N/4u)];
        }
        C[cRow * N / 4u + cCol] = tmp; 
    }
}
