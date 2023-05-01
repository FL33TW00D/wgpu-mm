//Original: https://github.com/webonnx/wonnx/blob/master/wonnx/templates/matrix/gemm.wgsl
{% macro gemm_core(A, B) %}
    let y = global_id.x % {{ N / 4 }}u;
    let x = global_id.x / {{ N / 4 }}u;

    let index = (x * {{ N }}u) + y;

    let zero_vec = vec4<f32>(
        {% for i in range(end = 4) %}
            0. {%-if not loop.last -%},{%- endif -%}
        {% endfor %}
    );

    var zero_matrix = mat4x4<f32>(
        {% for i in range(end = 4) %}
            zero_vec {%-if not loop.last -%},{%- endif -%}
        {% endfor %}
    );

    var result = zero_matrix;
    var product = zero_matrix;

    for(var k: u32 = 0u; k < {{ K / 4 }}u; k = k + 1u) {
        workgroupBarrier();
        let index_left =  (x * {{ K }}u) + k;
        let index_right =  (k * {{ N }}u) + y;

        let mat_left = mat4x4<f32>(
            {% for i in range(end = 4) %}
                {{ A }}[index_left + {{ i * K/4 }}u] {%-if not loop.last -%},{%- endif -%}
            {% endfor %}
        );
        
        var mat_right = mat4x4<f32>(
            {% for i in range(end = 4) %}
                {{ B }}[index_right + ({{ i * N/4 }}u)] {%-if not loop.last -%},{%- endif -%}
            {% endfor %}
        );

        product = mat_right * mat_left;

        {% for i in range(end = 4) %}
            result[{{ i }}u] += product[{{ i }}u];
        {% endfor %}
    }
{% endmacro gemm_core %}

{% macro gemm(A, B, C) %}
    {{ self::gemm_core(A=A, B=B) }}
    {% for i in range(end = 4) %}
        {{ C }}[index + {{ i * N / 4 }}u] = result[{{ i }}u];
    {% endfor %}
{% endmacro gemm %}

