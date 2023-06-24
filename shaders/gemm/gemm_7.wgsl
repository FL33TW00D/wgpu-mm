//Kernel 6: Vectorize 
//https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/6_kernel_vectorize.cuh
@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> C: array<vec4<f32>>;

// Maximum workgroup memory size is 16KB
var<workgroup> As: array<vec4<f32>, {{ (BM * BK) / 4 }}u>;
var<workgroup> Bs: array<vec4<f32>, {{ (BK * BN) / 4 }}u>;

//workgroup_size_x = 256 
@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let M = {{ M }}u;
    let N = {{ N }}u;
    let K = {{ K }}u;
    let cRow = group_id.y; // 0..16 
    let cCol = group_id.x; // 0..16

    let threadCol = local_id.x % {{ BN / TN }}u; // 256 % (64 / 1) = 0..63
    let threadRow = local_id.x / {{ BN / TN }}u; // 256 / (64 / 1) = 0..3

    //top left corner of the matrix
    var aIdx = cRow * {{ BM }}u * K / 4u);                    
    var bIdx = cCol * {{ BN / 4 }}u;                        
    var cIdx = cRow * {{ BM }}u * (N / 4u) + cCol * {{ BN / 4 }}u; 

    let tileRowA = local_id.x / {{ BK / 4 }}u; // 256 / 4 = 0..63
    let tileColA = local_id.x % {{ BK / 4 }}u; // 256 % 4 = 0..3 

    let tileRowB = local_id.x / {{ BN / 4 }}u; // 256 / 16 = 0..15
    let tileColB = local_id.x % {{ BN / 4 }}u; // 256 % 16 = 0..15

    var threadResults = array<vec4<f32>, {{ (TM * TN) / 4 }}u>();

    var regM = array<f32, {{ TM }}u>();
    var regN = array<f32, {{ TN }}u>();

    for (var bkIdx = 0u; bkIdx < K; bkIdx += {{ BK }}u) {

        var tmp = A[aIdx + (tileRowA * K + tileColA * 4u)];
        As[(tileColA * 4u + 0u) * {{ BM }}u + tileRowA] = tmp.x;
        As[(tileColA * 4u + 1u) * {{ BM }}u + tileRowA] = tmp.y;
        As[(tileColA * 4u + 2u) * {{ BM }}u + tileRowA] = tmp.z;
        As[(tileColA * 4u + 3u) * {{ BM }}u + tileRowA] = tmp.w;
        
        Bs[tileRowB * {{ BN }}u + tileColB * 4u] = B[bIdx + (tileRowB * N + tileColB * 4u)];
        workgroupBarrier();

        aIdx += {{ BK }}u;
        bIdx += {{ BK }}u * (N / 4u);

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
