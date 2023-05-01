{% import "gemm_macro.wgsl" as gemm %}
@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> C: array<vec4<f32>>;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    {{ gemm::gemm(A="A", B="B", C="C") }}
}

