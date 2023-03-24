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

    let KD = {{ K / 4 }}u;
    let ND = {{ N / 4 }}u;
        
    {% for x_coord in range(end=B_TILE_N) -%}
        {% for y_coord in range(end=B_TILE_K) -%}
            var result{{ x_coord }}_{{ y_coord }}: vec4<f32> = vec4<f32>();
        {% endfor %}
    {% endfor %}

    for(var k: u32 = 0u; k < KD; k = k + 1u){
        {% for x_coord in range(end=A_TILE_K) -%} //don't support A_TILE_K > 1
            {% for y_coord in range(end=A_TILE_M) -%}
                var a{{ x_coord }}_{{ y_coord }} = A[(y * 4u + {{ y_coord }}u) * KD + k];
            {% endfor %}    
        {% endfor %}
        var brow: vec4<f32>;

        {% for component_idx in range(end=4) -%}
            {% for x_coord in range(end=B_TILE_N) -%}
                brow = B[(k * 4u + {{ component_idx }}u) * ND + x * {{ B_TILE_N }}u + {{ x_coord }}u];
                {% for y_coord in range(end=B_TILE_K) -%}
                    result{{ x_coord }}_{{ y_coord }} = fma(vec4<f32>(a0_{{ y_coord }}.{{ components[component_idx] }}), brow, result{{ x_coord }}_{{ y_coord }});
                {% endfor %}
            {% endfor %}
        {% endfor %}
    }

    {% for x_coord in range(end=B_TILE_N) -%}
        {% for y_coord in range(end=B_TILE_K) -%}
            C[x * {{ B_TILE_N }}u + {{ x_coord }}u + (y * 4u + {{ y_coord }}u) * ND] = result{{ x_coord }}_{{ y_coord }};
        {% endfor %}
    {% endfor %}
}
