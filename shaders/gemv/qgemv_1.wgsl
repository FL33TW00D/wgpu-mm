@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<u32>;

@group(0) @binding(2)
var<storage, read_write> C: array<vec4<f32>>;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let left_offset = global_id.y * {{ (K / 4) }}u; 
    let right_offset = global_id.y * {{ (K * N) / 4 }}u;
    let output_offset = global_id.y * {{ (N / 4) }}u;
    
    var result = vec4<f32>(0.0);

    for(var k: u32 = 0u; k < {{ K / 4 }}u; k = k + 1u) {
        let index_left = left_offset + k; 
        let index_right = right_offset + global_id.x + (k * {{ N }}u);

        let left = A[index_left];

        {% for i in range(end = 4) %}
            let right_{{ i }} = unpack4x8snorm(B[index_right + ({{ i }}u * {{ (N / 4)}}u)]) * {{ absmax }}f;
        {% endfor %}

        {% for i in range(end = 4) %}
            result[{{ i }}] += dot(left, vec4<f32>(
            {%- for j in range(end = 4) -%}
                right_{{ j }}[{{ i }}],
            {%- endfor -%}
            ));
        {% endfor %}

    }
    
    C[output_offset + global_id.x] = result; 
}


