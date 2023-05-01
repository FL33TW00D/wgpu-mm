//Original: https://jott.live/markdown/m1_webgpu_perf
@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<f32>;

@group(0) @binding(2)
var<storage, read_write> C: array<f32>;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{workgroup_size_z }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let gidx = global_id.x;

    {# Calculate stacking offsets #}
    let left_offset = global_id.y * {{ (M * K) / kernel_size | int }}u; 
    let right_offset = global_id.y * {{ (K * N) }}u;
    let output_offset = global_id.y * {{ (M * N) }}u;

    var result = 0f;
    var product = 0f;

    for(var k: u32 = 0u; k < {{ K / kernel_size | int }}u; k = k + 1u) {
        workgroupBarrier();
        let index_left = left_offset + k; 
        let index_right = right_offset + (k * {{ N * kernel_size }}u) + gidx; 

        let vec_left = A[index_left];

        let vec_right = vec4<f32>(
            {% for i in range(end = kernel_size) %}
                B[index_right + {{ i * N }}u] {%-if not loop.last -%},{%- endif -%}
            {% endfor %}
        );

        product = dot(vec_left, vec_right);
        result += product; 
    }
}
