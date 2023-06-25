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
    let cTileRow = group_id.y; // 0..16 
    let cTileCol = group_id.x; // 0..16

    let threadCol = local_id.x % {{ BN }}u; // 256 % 64 = 0..63
    let threadRow = local_id.x / {{ BN }}u; // 256 / 64 = 0..3

    //top left corner of the matrix
    //Where the march starts
    var aIdx = cTileRow * {{ BM }}u * (K / 4u);                    
    var bIdx = cTileCol * {{ BN / 4 }}u;                        
    var cIdx = cTileRow * {{ BM }}u * (N / 4u) + cTileCol * {{ BN / 4 }}u; 

    let tileRowA = local_id.x / {{ BK / 4 }}u; // 256 / 4 = 0..63
    let tileColA = local_id.x % {{ BK / 4 }}u; // 256 % 4 = 0..3 

    let tileRowB = local_id.x / {{ BN / 4 }}u; // 256 / 16 = 0..15
    let tileColB = local_id.x % {{ BN / 4 }}u; // 256 % 16 = 0..15

    // 16 * 4 = 64 results per thread
    var threadResults = array<vec4<f32>, 4u>();

    for (var bkIdx = 0u; bkIdx < K; bkIdx += {{ BK }}u) {
        As[tileRowA * {{ BK }}u + tileColA * 4u] = A[aIdx + (tileRowA * K + tileColA * 4u)];
        Bs[tileRowB * {{ BN }}u + tileColB * 4u] = B[bIdx + (tileRowB * N + tileColB * 4u)];
        workgroupBarrier();

        aIdx += {{ BK }}u;
        bIdx += {{ BK }}u * (N / 4u);

        for (var dotIdx = 0u; dotIdx < {{ BK }}u; dotIdx++) {
            let tmpA = As[(tileRowA * {{ BK }}u + dotIdx) / 4u];
            for (var resIdx = 0u; resIdx < 4u; resIdx++) {
                let tmpB = Bs[(tileColB * {{ BN }}u + resIdx) / 4u];
                threadResults[ ] += vec4<f32>(tmpA[resIdx]) * tmpB; 
            }
        }
        workgroupBarrier();
    }

}
