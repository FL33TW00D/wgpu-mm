//Kernel 5: 2D Blocktiling for Calculating Multiple Results per Thread
//https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/5_kernel_2D_blocktiling.cuh
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

    let totalResultsBlocktile = {{ BM * BN }}u;
    let numThreadsBlocktile = totalResultsBlocktile / {{ TM * TN }}u;

    let threadCol = local_id.x % {{ BN / TN }}u;
    let threadRow = local_id.x / {{ BN / TN }}u;

    var aIdx = cRow * {{ BM }}u * K;                    
    var bIdx = cCol * {{ BN }}u;                        
    var cIdx = cRow * {{ BM }}u * N + cCol * {{ BN }}u; 

    let innerColA = local_id.x % {{ BK }}u; // warp-level GMEM coalescing
    let innerRowA = local_id.x / {{ BK }}u;

    // calculates the number of rows of As that are being loaded in a single step
    // by a single block
    let strideA = numThreadsBlocktile / {{ BK }}u;

    let innerColB = local_id.x % {{ BN }}u; // warp-level GMEM coalescing
    let innerRowB = local_id.x / {{ BN }}u;
    // for both As and Bs we want each load to span the full column-width, for
    // better GMEM coalescing (as opposed to spanning full row-width and iterating
    // across columns)
    let strideB = numThreadsBlocktile / {{ BN }}u;

    var threadResults = array<f32, {{ TM * TN }}u>();
    var regM = array<f32, {{ TM }}u>();
    var regN = array<f32, {{ TN }}u>();
    for (var bkIdx = 0u; bkIdx < K; bkIdx += {{ BK }}u) {
        for (var loadOffset = 0u; loadOffset < {{ BM }}u; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * {{ BK }}u + innerColA] = A[aIdx + (innerRowA + loadOffset) * K + innerColA];
        }
        for (var loadOffset = 0u; loadOffset < {{ BK }}u; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * {{ BN }}u + innerColB] = B[bIdx + (innerRowB + loadOffset) * N + innerColB];
        }
        workgroupBarrier();

        aIdx += {{ BK }}u;
        bIdx += {{ BK }}u * N;

        for (var dotIdx = 0u; dotIdx < {{ BK }}u; dotIdx++) {
            for (var i = 0u; i < {{ TM }}u; i++) {
                regM[i] = As[(threadRow * {{ TM }}u + i) * {{ BK }}u + dotIdx];
            }
            for (var i = 0u; i < {{ TN }}u; i++) {
                regN[i] = Bs[dotIdx * {{ BN }}u + threadCol * {{ TN }}u + i];
            }
            for (var resIdxM = 0u; resIdxM < {{ TM }}u; resIdxM++) {
                for (var resIdxN = 0u; resIdxN < {{ TN }}u; resIdxN++) {
                    threadResults[resIdxM * {{ TN }}u + resIdxN] = fma(regM[resIdxM], regN[resIdxN], threadResults[resIdxM * {{ TN }}u + resIdxN]);
                }
            }
        }
        workgroupBarrier();
    }
    for (var resIdxM = 0u; resIdxM < {{ TM }}u; resIdxM++) {
        for (var resIdxN = 0u; resIdxN < {{ TN }}u; resIdxN++) {
            C[cIdx + (threadRow * {{ TM }}u + resIdxM) * N + threadCol * {{ TN }}u + resIdxN] = threadResults[resIdxM * {{ TN }}u + resIdxN];
        }
    }
}
