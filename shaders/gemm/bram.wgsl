//Original: https://jott.live/markdown/m1_webgpu_perf
@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> C: array<vec4<f32>>;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{workgroup_size_z }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let m = global_id.x;
  let n = global_id.y;
    
  var result_0_0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var result_1_0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var result_2_0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var result_3_0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  for (var k: u32 = 0u; k < 256u; k = k + 1u) {
    let a_0_0 = A[(m * 4u ) * 256u + (k * 1u )];
    let a_1_0 = A[(m * 4u + 1u) * 256u + (k * 1u )];
    let a_2_0 = A[(m * 4u + 2u) * 256u + (k * 1u )];
    let a_3_0 = A[(m * 4u + 3u) * 256u + (k * 1u )];
    let b_0_0 = B[(k * 4u ) * 256u + (n * 1u )];
    let b_0_1 = B[(k * 4u + 1u) * 256u + (n * 1u )];
    let b_0_2 = B[(k * 4u + 2u) * 256u + (n * 1u )];
    let b_0_3 = B[(k * 4u + 3u) * 256u + (n * 1u )];
    result_0_0 += a_0_0.x * b_0_0;
    result_1_0 += a_1_0.x * b_0_0;
    result_2_0 += a_2_0.x * b_0_0;
    result_3_0 += a_3_0.x * b_0_0;
    result_0_0 += a_0_0.y * b_0_1;
    result_1_0 += a_1_0.y * b_0_1;
    result_2_0 += a_2_0.y * b_0_1;
    result_3_0 += a_3_0.y * b_0_1;
    result_0_0 += a_0_0.z * b_0_2;
    result_1_0 += a_1_0.z * b_0_2;
    result_2_0 += a_2_0.z * b_0_2;
    result_3_0 += a_3_0.z * b_0_2;
    result_0_0 += a_0_0.w * b_0_3;
    result_1_0 += a_1_0.w * b_0_3;
    result_2_0 += a_2_0.w * b_0_3;
    result_3_0 += a_3_0.w * b_0_3;
  }
  C[(m * 4u ) * 256u + (n * 1u )] = result_0_0;
  C[(m * 4u + 1u) * 256u + (n * 1u )] = result_1_0;
  C[(m * 4u + 2u) * 256u + (n * 1u )] = result_2_0;
  C[(m * 4u + 3u) * 256u + (n * 1u )] = result_3_0;
}
