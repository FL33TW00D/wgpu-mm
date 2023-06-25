//Kernel 3: Shared Memory Cache-Blocking
//https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/3_kernel_shared_mem_blocking.cuh
@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> C: array<vec4<f32>>;

//64 vec4 in As and 64 vec4 in Bs
var<workgroup> As: array<vec4<f32>, {{ (BLOCKSIZE * BLOCKSIZE) / 4 }}u>;
var<workgroup> Bs: array<vec4<f32>, {{ (BLOCKSIZE * BLOCKSIZE) / 4 }}u>;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let M = {{ M }}u;
    let N = {{ N }}u;
    let K = {{ K }}u;
    let cRow = group_id.x; // 0..64
    let cCol = group_id.y; // 0..64

    let threadCol = local_id.x % {{ BLOCKSIZE / 4 }}u; // 256 % 4 = 0..3
    let threadRow = local_id.x / {{ BLOCKSIZE }}u; // 256 / 16 = 0..15

    var aIdx = cRow * {{ BLOCKSIZE / 4 }}u * (K / 4u);                    // row=cRow, col=0
    var bIdx = cCol * {{ BLOCKSIZE / 4 }}u;                        // row=0, col=cCol
    var cIdx = cRow * {{ BLOCKSIZE / 4 }}u * (N / 4u) + cCol * {{ BLOCKSIZE / 4 }}u; // row=cRow, col=cCol

    var acc = vec4<f32>(); 
    for (var bkIdx = 0u; bkIdx < (K / 4u); bkIdx += {{ BLOCKSIZE / 4 }}u) {
        As[threadRow * {{ BLOCKSIZE / 4 }}u + threadCol] = A[aIdx + (threadRow * (K / 4u) + threadCol)];
        Bs[threadRow * {{ BLOCKSIZE / 4 }}u + threadCol] = B[bIdx + (threadRow * (N / 4u) + threadCol)];
        workgroupBarrier();

        aIdx += {{ BLOCKSIZE / 4 }}u;
        bIdx += {{ BLOCKSIZE }}u * (N / 4u);

        for(var dotIdx = 0u; dotIdx < {{ BLOCKSIZE / 4 }}u; dotIdx += 1u) {
        let BCached0 = Bs[dotIdx * {{ BLOCKSIZE / 4 }}u + threadCol]; 
        let BCached1 = Bs[(dotIdx + 1u) * {{ BLOCKSIZE / 4 }}u + threadCol];
        let BCached2 = Bs[(dotIdx + 2u) * {{ BLOCKSIZE / 4 }}u + threadCol];
        let BCached3 = Bs[(dotIdx + 3u) * {{ BLOCKSIZE / 4 }}u + threadCol];
            
            let ACached = As[threadRow * {{ BLOCKSIZE / 4 }}u + dotIdx];
            acc = fma(BCached0, vec4<f32>(ACached.x), acc);
            acc = fma(BCached1, vec4<f32>(ACached.y), acc);
            acc = fma(BCached2, vec4<f32>(ACached.z), acc);
            acc = fma(BCached3, vec4<f32>(ACached.w), acc);
        }
        workgroupBarrier();
    }
    //C[cIdx + (threadRow * (N / 4u))] = acc;
    if(global_id.x == 0u && global_id.y == 0u && global_id.z == 0u) {
        C[0] = acc; 
    }
}
