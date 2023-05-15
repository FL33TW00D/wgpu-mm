//Kernel 3: Shared Memory Cache-Blocking
//https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/3_kernel_shared_mem_blocking.cuh
@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> C: array<vec4<f32>>;

var<workgroup> As: array<vec4<f32>, {{ BM * BK / 4 }}u>;
var<workgroup> Bs: array<vec4<f32>, {{ BK * BN }}u>;

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

    let threadCol = local_id.x % {{ BN }}u;
    let threadRow = local_id.x / {{ BN }}u;

    var aIdx = cRow * {{ BM }}u * K;                    // row=cRow, col=0
    var bIdx = cCol * {{ BN }}u;                        // row=0, col=cCol
    var cIdx = cRow * {{ BM }}u * N + cCol * {{ BN }}u; // row=cRow, col=cCol

    var tmp = vec4<f32>();
    for (var bkIdx = 0u; bkIdx < K; bkIdx += {{ BK }}u) {
        As[threadRow * {{ BK }}u + threadCol] = A[aIdx + (threadRow * K + threadCol)];
        Bs[threadRow * {{ BN }}u + threadCol] = B[bIdx + (threadRow * N + threadCol)];
        workgroupBarrier();

        aIdx += {{ BK }}u;
        bIdx += {{ BK }}u * N;

        for (var dotIdx = 0u; dotIdx < {{ BK }}u; dotIdx += 1u) {
            let b0 = Bs[dotIdx * {{ BN }}u + threadCol];
            let b1 = Bs[dotIdx + 1u * {{ BN }}u + threadCol];
            let b2 = Bs[dotIdx + 2u * {{ BN }}u + threadCol];
            let b3 = Bs[dotIdx + 3u * {{ BN }}u + threadCol];
            tmp = fma(vec4<f32>(As[threadRow * {{ BK }}u + dotIdx].x), b0, tmp);
            tmp = fma(vec4<f32>(As[threadRow * {{ BK }}u + dotIdx].y), b1, tmp);
            tmp = fma(vec4<f32>(As[threadRow * {{ BK }}u + dotIdx].z), b2, tmp);
            tmp = fma(vec4<f32>(As[threadRow * {{ BK }}u + dotIdx].w), b3, tmp);
        }
        workgroupBarrier();
    }
    C[cIdx + (threadRow * N + threadCol)] = tmp;
}
