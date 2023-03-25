@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> C: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read_write> debug: array<mat4x4<f32>>;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let x = global_id.x;
    let y = global_id.y;

    let KD = {{ K / 4 }}u;
    let ND = {{ N / 4 }}u;
        
    {% for bx in range(end=B_TILE_X) -%}
        {% for by in range(end=B_TILE_Y) -%}
            var result{{ bx }}_{{ by }}: vec4<f32> = vec4<f32>();
        {% endfor %}
    {% endfor %}

    for(var k: u32 = 0u; k < KD; k = k + 1u){
        //Load A tile
        {% for ax in range(end=A_TILE_X) -%}
            {% for ay in range(end=A_TILE_Y) -%}
                var a{{ ax }}_{{ ay }} = A[(y * 4u + {{ ay }}u) * KD + k * {{ A_TILE_X }}u + {{ ax }}u];
            {% endfor %}    
        {% endfor %}
        var brow: vec4<f32>;

        {% for component_idx in range(end=4) -%}
            {% for bx in range(end=B_TILE_X) -%}
                brow = B[(k * 4u + {{ component_idx }}u) * ND + x * {{ B_TILE_X }}u + {{ bx }}u];
                {% for by in range(end=B_TILE_Y) -%}
                    result{{ bx }}_{{ by }} = fma(vec4<f32>(a0_{{ by }}.{{ components[component_idx] }}), brow, result{{ bx }}_{{ by }});
                {% endfor %}
            {% endfor %}
        {% endfor %}
    }

    {% for bx in range(end=B_TILE_X) -%}
        {% for by in range(end=B_TILE_Y) -%}
            C[x * {{ B_TILE_X }}u + {{ bx }}u + (y * 4u + {{ by }}u) * ND] = result{{ bx }}_{{ by }};
        {% endfor %}
    {% endfor %}
}
