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

    let KD = {{ K / B_TILE_Y }}u;
    let ND = {{ N / (B_TILE_X * 4) }}u;
        
    {% for x_coord in range(end=B_TILE_X) %}
        {% for y_coord in range(end=B_TILE_Y) %}
            var result{{ x_coord }}_{{ y_coord }}: vec4<f32> = vec4<f32>();
        {% endfor %}
    {% endfor %}

    for(var k: u32 = 0u; k < KD; k = k + 1u){
        {% for x_coord in range(end=A_TILE_X) %}
            {% for y_coord in range(end=A_TILE_Y) %}
                var a{{ x_coord }}_{{ y_coord }} = A[(y * 4u + {{ y_coord }}u) * KD + k * {{ A_TILE_Y }}u + {{ x_coord }}u];
            {% endfor %}    
        {% endfor %}
        var brow: vec4<f32>;

        {% for x_coord in range(end=B_TILE_X) %}
            {% for y_coord in range(end=B_TILE_Y) %}
                brow = B[(k * 4u + {{ y_coord }}u) * ND + x * {{ B_TILE_X }}u + {{ x_coord }}u];
                {% for y_coord in range(end=A_TILE_Y) %}
                    result{{ x_coord }}_{{ y_coord }} = result{{ x_coord }}_{{ y_coord }} + a{{ x_coord }}_{{ y_coord }} * brow;
                {% endfor %}
            {% endfor %}
        {% endfor %}
    }

    {% for x_coord in range(end=B_TILE_X) %}
        {% for y_coord in range(end=B_TILE_Y) %}
            C[(y * 4u + {{ y_coord }}u) * ND + x * {{ B_TILE_X }}u + {{ x_coord }}u] = result{{ x_coord }}_{{ y_coord }};
        {% endfor %}
    {% endfor %}
}
