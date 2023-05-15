//Kernel 3: Shared Memory Cache-Blocking
//https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/3_kernel_shared_mem_blocking.cuh
@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> B: array<f32>;

@group(0) @binding(2)
var<storage, read_write> C: array<f32>;

var<workgroup> As: array<f32, {{ BLOCKSIZE * BLOCKSIZE }}u>;
var<workgroup> Bs: array<f32, {{ BLOCKSIZE * BLOCKSIZE }}u>;

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

    let threadCol = local_id.x % {{ BLOCKSIZE }}u;
    let threadRow = local_id.x / {{ BLOCKSIZE }}u;

    var aIdx = cRow * {{ BLOCKSIZE }}u * K;                    // row=cRow, col=0
    var bIdx = cCol * {{ BLOCKSIZE }}u;                        // row=0, col=cCol
    var cIdx = cRow * {{ BLOCKSIZE }}u * N + cCol * {{ BLOCKSIZE }}u; // row=cRow, col=cCol

    var tmp = 0f;
    for (var bkIdx = 0u; bkIdx < K; bkIdx += {{ BLOCKSIZE }}u) {
        As[threadRow * {{ BLOCKSIZE }}u + threadCol] = A[aIdx + (threadRow * K + threadCol)];
        Bs[threadRow * {{ BLOCKSIZE }}u + threadCol] = B[bIdx + (threadRow * N + threadCol)];
        workgroupBarrier();

        aIdx += {{ BLOCKSIZE }}u;
        bIdx += {{ BLOCKSIZE }}u * N;

        for (var dotIdx = 0u; dotIdx < {{ BLOCKSIZE }}u; dotIdx += 1u) {
            tmp = fma(As[threadRow * {{ BLOCKSIZE }}u + dotIdx], Bs[dotIdx * {{ BLOCKSIZE }}u + threadCol], tmp);
        }
        workgroupBarrier();
    }
    C[cIdx + (threadRow * N + threadCol)] = tmp;
}
