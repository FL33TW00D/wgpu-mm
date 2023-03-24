//Original: https://github.com/milhidaka/webgpu-blas/blob/master/src/shader_sgemm_block.ts
@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> C: array<vec4<f32>>;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  let x = global_id.x;
  let y = global_id.y;

  let KD4 = 256u;
  let ND4 = 256u;

  var result_0_0: vec4<f32> = vec4<f32>();
  var result_0_1: vec4<f32> = vec4<f32>();
  var result_0_2: vec4<f32> = vec4<f32>();
  var result_0_3: vec4<f32> = vec4<f32>();
  var result_1_0: vec4<f32> = vec4<f32>();
  var result_1_1: vec4<f32> = vec4<f32>();
  var result_1_2: vec4<f32> = vec4<f32>();
  var result_1_3: vec4<f32> = vec4<f32>();
  for(var k: u32 = 0u; k < KD4; k = k + 1u) {
    var arow0: vec4<f32> = A[(y * 4u + 0u) * KD4 + k];
    var arow1: vec4<f32> = A[(y * 4u + 1u) * KD4 + k];
    var arow2: vec4<f32> = A[(y * 4u + 2u) * KD4 + k];
    var arow3: vec4<f32> = A[(y * 4u + 3u) * KD4 + k];
    var brow: vec4<f32>;

    brow = B[(k * 4u + 0u) * ND4 + x * 2u + 0u];
    result_0_0 = fma(vec4<f32>(arow0.x), brow, result_0_0);
    result_0_1 = fma(vec4<f32>(arow1.x), brow, result_0_1);
    result_0_2 = fma(vec4<f32>(arow2.x), brow, result_0_2);
    result_0_3 = fma(vec4<f32>(arow3.x), brow, result_0_3);

    brow = B[(k * 4u + 0u) * ND4 + x * 2u + 1u];
    result_1_0 = fma(vec4<f32>(arow0.x), brow, result_1_0);
    result_1_1 = fma(vec4<f32>(arow1.x), brow, result_1_1);
    result_1_2 = fma(vec4<f32>(arow2.x), brow, result_1_2);
    result_1_3 = fma(vec4<f32>(arow3.x), brow, result_1_3);
    
    brow = B[(k * 4u + 1u) * ND4 + x * 2u + 0u];
    result_0_0 = fma(vec4<f32>(arow0.y), brow, result_0_0);
    result_0_1 = fma(vec4<f32>(arow1.y), brow, result_0_1);
    result_0_2 = fma(vec4<f32>(arow2.y), brow, result_0_2);
    result_0_3 = fma(vec4<f32>(arow3.y), brow, result_0_3);

    brow = B[(k * 4u + 1u) * ND4 + x * 2u + 1u];
    result_1_0 = fma(vec4<f32>(arow0.y), brow, result_1_0);
    result_1_1 = fma(vec4<f32>(arow1.y), brow, result_1_1);
    result_1_2 = fma(vec4<f32>(arow2.y), brow, result_1_2);
    result_1_3 = fma(vec4<f32>(arow3.y), brow, result_1_3);
    
    brow = B[(k * 4u + 2u) * ND4 + x * 2u + 0u];
    result_0_0 = fma(vec4<f32>(arow0.z), brow, result_0_0);
    result_0_1 = fma(vec4<f32>(arow1.z), brow, result_0_1);
    result_0_2 = fma(vec4<f32>(arow2.z), brow, result_0_2);
    result_0_3 = fma(vec4<f32>(arow3.z), brow, result_0_3);

    brow = B[(k * 4u + 2u) * ND4 + x * 2u + 1u];
    result_1_0 = fma(vec4<f32>(arow0.z), brow, result_1_0);
    result_1_1 = fma(vec4<f32>(arow1.z), brow, result_1_1);
    result_1_2 = fma(vec4<f32>(arow2.z), brow, result_1_2);
    result_1_3 = fma(vec4<f32>(arow3.z), brow, result_1_3);
    
    brow = B[(k * 4u + 3u) * ND4 + x * 2u + 0u];
    result_0_0 = fma(vec4<f32>(arow0.w), brow, result_0_0);
    result_0_1 = fma(vec4<f32>(arow1.w), brow, result_0_1);
    result_0_2 = fma(vec4<f32>(arow2.w), brow, result_0_2);
    result_0_3 = fma(vec4<f32>(arow3.w), brow, result_0_3);

    brow = B[(k * 4u + 3u) * ND4 + x * 2u + 1u];
    result_1_0 = fma(vec4<f32>(arow0.w),  brow , result_1_0);
    result_1_1 = fma(vec4<f32>(arow1.w),  brow , result_1_1);
    result_1_2 = fma(vec4<f32>(arow2.w),  brow , result_1_2);
    result_1_3 = fma(vec4<f32>(arow3.w),  brow , result_1_3);
  }
  C[x * 2u + 0u + (y * 4u + 0u) * ND4] = result_0_0;
  C[x * 2u + 0u + (y * 4u + 1u) * ND4] = result_0_1;
  C[x * 2u + 0u + (y * 4u + 2u) * ND4] = result_0_2;
  C[x * 2u + 0u + (y * 4u + 3u) * ND4] = result_0_3;
  C[x * 2u + 1u + (y * 4u + 0u) * ND4] = result_1_0;
  C[x * 2u + 1u + (y * 4u + 1u) * ND4] = result_1_1;
  C[x * 2u + 1u + (y * 4u + 2u) * ND4] = result_1_2;
  C[x * 2u + 1u + (y * 4u + 3u) * ND4] = result_1_3;
}
