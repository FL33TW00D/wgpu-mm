//Kernel 6: Vectorize 
//https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/6_kernel_vectorize.cuh
@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> C: array<vec4<f32>>;

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

    // BN/TN are the number of threads to span a column
    let threadCol = local_id.x % {{ BN / TN }}u;
    let threadRow = local_id.x / {{ BN / TN }}u;

    var aIdx = cRow * {{ BM }}u * K;                    
    var bIdx = cCol * {{ BN }}u;                        
    var cIdx = cRow * {{ BM }}u * N + cCol * {{ BN }}u; 

    let tileRowA = local_id.x / {{ BK / 4 }}u;
    let tileColA = local_id.x % {{ BK / 4 }}u; 

    let tileRowB = local_id.x / {{ BN / 4 }}u;
    let tileColB = local_id.x % {{ BN / 4 }}u; 

    var threadResults = array<f32, {{ TM * TN }}u>();

    var regM = array<f32, {{ TM }}u>();
    var regN = array<f32, {{ TN }}u>();

    for (var bkIdx = 0u; bkIdx < K; bkIdx += {{ BK }}u) {

        var tmp = A[aIdx + (tileRowA * K + tileColA * 4u)];
        As[(tileColA * 4u + 0u) * {{ BM }}u + tileRowA] = tmp.x;
        As[(tileColA * 4u + 1u) * {{ BM }}u + tileRowA] = tmp.y;
        As[(tileColA * 4u + 2u) * {{ BM }}u + tileRowA] = tmp.z;
        As[(tileColA * 4u + 3u) * {{ BM }}u + tileRowA] = tmp.w;
        
        tmp = B[bIdx + (tileRowB * N + tileColB * 4u)];
        Bs[tileRowB * {{ BN }}u + tileColB * 4u] = tmp.x;
        Bs[tileRowB * {{ BN }}u + tileColB * 4u + 1u] = tmp.y;
        Bs[tileRowB * {{ BN }}u + tileColB * 4u + 2u] = tmp.z;
        Bs[tileRowB * {{ BN }}u + tileColB * 4u + 3u] = tmp.w;
        workgroupBarrier();

        aIdx += {{ BK }}u;
        bIdx += {{ BK }}u * N;

        for (var dotIdx = 0u; dotIdx < {{ BK }}u; dotIdx++) {
            for (var i = 0u; i < {{ TM }}u; i++) {
                regM[i] = As[dotIdx * {{ BM }}u + threadRow * {{ TM }}u + i];
            }
            for (var i = 0u; i < {{ TN }}u; i++) {
                regN[i] = Bs[dotIdx * {{ BN }}u + threadCol * {{ TN }}u + i];
            }
            for (var resIdxM = 0u; resIdxM < {{ TM }}u; resIdxM++) {
                for (var resIdxN = 0u; resIdxN < {{ TN }}u; resIdxN++) {
                    threadResults[resIdxM * {{ TN }}u + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        workgroupBarrier();
    }
    for (var resIdxM = 0u; resIdxM < {{ TM }}u; resIdxM++) {
        for (var resIdxN = 0u; resIdxN < {{ TN }}u; resIdxN += 4u) {
            var tmp = C[cIdx + (threadRow * {{ TM }}u + resIdxM) * N + threadCol * {{ TN }}u + resIdxN]; 
            tmp.x += threadResults[resIdxM * {{ TN }}u + resIdxN];
            tmp.y += threadResults[resIdxM * {{ TN }}u + resIdxN + 1u];
            tmp.z += threadResults[resIdxM * {{ TN }}u + resIdxN + 2u];
            tmp.w += threadResults[resIdxM * {{ TN }}u + resIdxN + 3u];
            C[cIdx + (threadRow * {{ TM }}u + resIdxM) * N + threadCol * {{ TN }}u + resIdxN] = tmp;
        }
    }
}
