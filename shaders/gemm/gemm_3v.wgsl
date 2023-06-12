//Kernel 3: Shared Memory Cache-Blocking
//https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/3_kernel_shared_mem_blocking.cuh
@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> C: array<vec4<f32>>;

var<workgroup> As: array<vec4<f32>, {{ 256 }}u>;
var<workgroup> Bs: array<vec4<f32>, {{ 256 }}u>;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let M = {{ M }}u;
    let N = {{ N }}u;
    let K = {{ K }}u;
    let cRow = group_id.x; //0-31
    let cCol = group_id.y; //0-31

    let threadCol = local_id.x % {{ BSD4 }}u; //0-7
    let threadRow = local_id.x / {{ BS }}u; //0-31

    var aStart = cRow * {{ BS }}u * (K / 4u); 
    var bStart = cCol * {{ BSD4 }}u;
    var cIdx = cRow * {{ BS }}u * (N / 4u) + cCol * {{ BSD4 }}u; 
    //I think the above is correct

    let numTiles = K / {{ BSD4 }}u;

    var tmp = vec4<f32>();
    for (var t = 0u; t < numTiles; t += {{ BSD4 }}u) {
        As[threadRow * {{ BSD4 }}u + threadCol] = A[aStart + (threadRow * (K / 4u) + threadCol)];
        Bs[threadRow * {{ BSD4 }}u + threadCol] = B[bStart + (threadRow * (N / 4u) + threadCol)];
        workgroupBarrier();

        aStart += {{ BS }}u;
        bStart += {{ BS }}u * (N / 4u);

        for (var dotIdx = 0u; dotIdx < {{ BSD4 }}u; dotIdx += 1u) {
            let aCached = As[threadRow * {{ BSD4 }}u + dotIdx];
            let b0 = Bs[dotIdx * {{ BSD4 }}u + threadCol + {{ BSD4 }}u * 0u];
            let b1 = Bs[dotIdx * {{ BSD4 }}u + threadCol + {{ BSD4 }}u * 1u];
            let b2 = Bs[dotIdx * {{ BSD4 }}u + threadCol + {{ BSD4 }}u * 2u];
            let b3 = Bs[dotIdx * {{ BSD4 }}u + threadCol + {{ BSD4 }}u * 3u];
            tmp = fma(vec4<f32>(aCached.x), b0, tmp);
            tmp = fma(vec4<f32>(aCached.y), b1, tmp);
            tmp = fma(vec4<f32>(aCached.z), b2, tmp);
            tmp = fma(vec4<f32>(aCached.w), b3, tmp);
        }
        workgroupBarrier();
    }
    C[cIdx + (threadRow * (N/4u) + threadCol)] = tmp;
}
