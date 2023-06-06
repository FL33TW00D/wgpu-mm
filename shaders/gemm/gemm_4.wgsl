//Kernel 4: 1D Blocktiling for Calculating Multiple Results per Thread
//https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/4_kernel_1D_blocktiling.cuh
@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> B: array<f32>;

@group(0) @binding(2)
var<storage, read_write> C: array<f32>;

var<workgroup> As: array<f32, {{ BM * BK }}u>;
var<workgroup> Bs: array<f32, {{ BK * BN }}u>;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let M = {{ M }}u;
    let N = {{ N }}u;
    let K = {{ K }}u;
    let cRow = group_id.y; 
    let cCol = group_id.x;

    let threadCol = local_id.x % {{ BN }}u;
    let threadRow = local_id.x / {{ BN }}u;

    var aIdx = cRow * {{ BM }}u * K;                    
    var bIdx = cCol * {{ BN }}u;                        
    var cIdx = cRow * {{ BM }}u * N + cCol * {{ BN }}u; 

    let innerColA = local_id.x % {{ BK }}u; // warp-level GMEM coalescing
    let innerRowA = local_id.x / {{ BK }}u;
    let innerColB = local_id.x % {{ BN }}u; // warp-level GMEM coalescing
    let innerRowB = local_id.x / {{ BN }}u;

    var threadResults = array<f32, {{ TM }}u>(
        {% for i in range(end = TM) -%}
            0.0,
        {% endfor -%}
    );
    for (var bkIdx = 0u; bkIdx < K; bkIdx += {{ BK }}u) {
        As[innerRowA * {{ BK }}u + innerColA] = A[aIdx + (innerRowA * K + innerColA)];
        Bs[innerRowB * {{ BN }}u + innerColB] = B[bIdx + (innerRowB * N + innerColB)];
        workgroupBarrier();

        aIdx += {{ BK }}u;
        bIdx += {{ BK }}u * N;

        for (var dotIdx = 0u; dotIdx < {{ BK }}u; dotIdx++) {
            var tmpB = Bs[dotIdx * {{ BN }}u + threadCol];
            for (var resIdx = 0u; resIdx < {{ TM }}u; resIdx++) {
                threadResults[resIdx] = fma(As[(threadRow * {{ TM }}u + resIdx) * {{ BK }}u + dotIdx], tmpB, threadResults[resIdx]);
            }
        }
        workgroupBarrier();
    }
    for (var resIdx = 0u; resIdx < {{ TM }}u; resIdx++) {
        C[cIdx + (threadRow * {{ TM }}u + resIdx) * N + threadCol] = threadResults[resIdx];
    }
}
