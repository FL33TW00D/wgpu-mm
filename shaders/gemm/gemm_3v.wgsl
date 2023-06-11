//Kernel 3: Shared Memory Cache-Blocking
//https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/3_kernel_shared_mem_blocking.cuh
@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> C: array<vec4<f32>>;

var<workgroup> As: array<vec4<f32>, {{ * BLOCKSIZE }}u>;
var<workgroup> Bs: array<vec4<f32>, {{ BLOCKSIZE * BLOCKSIZE }}u>;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let M = {{ M }}u;
    let N = {{ N }}u;
    let K = {{ K }}u;
    let cRow = group_id.x; 
    let cCol = group_id.y;

    let threadCol = local_id.x % {{ BLOCKSIZE / 4 }}u;
    let threadRow = local_id.x / {{ BLOCKSIZE }}u;

    var aIdx = cRow * {{ BLOCKSIZE }}u * (K / 4u); 
    var bIdx = cCol * {{ BLOCKSIZE / 4 }}u;
    var cIdx = cRow * {{ BLOCKSIZE }}u * N + cCol * {{ BLOCKSIZE / 4 }}u; 

    var tmp = vec4<f32>();
    for (var bkIdx = 0u; bkIdx < K; bkIdx += {{ BLOCKSIZE }}u) {
        As[threadRow * {{ BLOCKSIZE / 4 }}u + threadCol] = A[aIdx + (threadRow * (K / 4u) + threadCol)];
        Bs[threadRow * {{ BLOCKSIZE }}u + threadCol] = B[bIdx + (threadRow * N + threadCol)];
        workgroupBarrier();

        aIdx += {{ BLOCKSIZE / 4 }}u;
        bIdx += {{ BLOCKSIZE }}u * (N / 4u);

        for (var dotIdx = 0u; dotIdx < {{ BLOCKSIZE }}u; dotIdx += 1u) {
            let aCached = As[threadRow * {{ BLOCKSIZE / 4 }}u + dotIdx];
            let b0 = Bs[dotIdx * {{ BLOCKSIZE }}u + threadCol];
            let b1 = Bs[dotIdx + 1u * {{ BLOCKSIZE }}u + threadCol];
            let b2 = Bs[dotIdx + 2u * {{ BLOCKSIZE }}u + threadCol];
            let b3 = Bs[dotIdx + 3u * {{ BLOCKSIZE }}u + threadCol];
            tmp = fma(vec4<f32>(aCached.x), b0, tmp);
            tmp = fma(vec4<f32>(aCached.y), b1, tmp);
            tmp = fma(vec4<f32>(aCached.z), b2, tmp);
            tmp = fma(vec4<f32>(aCached.w), b3, tmp);
        }
        workgroupBarrier();
    }
    C[cIdx + (threadRow * N + threadCol)] = tmp;
}
