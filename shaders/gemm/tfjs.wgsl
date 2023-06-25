  fn mm_readA(batch: i32, row: i32, colIn: i32) -> f32 {
    var value = f32(0.0);
    let col = colIn * 1;

    if(row < uniforms.aShape[1] && col < uniforms.aShape[2])
    {

      value = getA(batch, row, col);


    }

    return value;
  }

  fn mm_readB(batch: i32, row: i32, colIn: i32) -> f32 {
    let col = colIn * 1;
    var value = f32(0.0);
    value = getB(batch, row, col);
    return value
  }

  fn mm_write(batch: i32, row: i32, colIn: i32, valueIn: f32) {
      let index = batch * 1024 * (1024/4) + row * 1024 + colIn;
  }

var<workgroup> mm_Asub : array<array<vec4<f32>, 8>, 32>;
var<workgroup> mm_Bsub : array<array<vec4<f32>, 8>, 32>;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let localRow = i32(local_id.y);
    let tileRow = localRow * 4;
    let tileCol = i32(local_id.x);

    let globalRow = i32(global_id.y) * 4;
    let globalCol = i32(global_id.x);
    let batch = i32(global_id.z);
    let batchA = batch;
    let batchB = batch;
    let globalRowStart = i32(group_id.y) * 32;

    let numTiles = (1024 - 1) / 32 + 1;
    var kStart = 0;

    var acc: array<vec4<f32>, 4>;

    // Loop over shared dimension.
    let tileRowB = localRow * 4;
    for (var t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (var innerRow = 0; innerRow < 4; innerRow++) {
            let inputRow = tileRow + innerRow;
            let inputCol = tileCol;

            mm_Asub[inputRow][inputCol] = mm_readA(
              batchA,
              globalRow + innerRow,
              kStart / 4 + inputCol
            );
        }

        // Load one tile of B into local memory.
        for (var innerRow = 0; innerRow < 4; innerRow++) {
            let inputRow = tileRowB + innerRow;
            let inputCol = tileCol;
            mm_Bsub[inputRow][inputCol] = mm_readB(batchB, kStart + inputRow, globalCol);
        }
        kStart = kStart + 32;
        workgroupBarrier();

        // Compute acc values for a single thread.
        for (var k = 0; k < 8; k++) {
            let BCached0 = mm_Bsub[k * 4][tileCol];
            let BCached1 = mm_Bsub[k * 4 + 1][tileCol];
            let BCached2 = mm_Bsub[k * 4 + 2][tileCol];
            let BCached3 = mm_Bsub[k * 4 + 3][tileCol];


            for (var i = 0; i < 4; i++) {
              let ACached = mm_Asub[tileRow + i][k];
              acc[i] = fma(BCached0, vec4<f32>(ACached.x), acc[i]);
              acc[i] = fma(BCached1, vec4<f32>(ACached.y), acc[i]);
              acc[i] = fma(BCached2, vec4<f32>(ACached.z), acc[i]);
              acc[i] = fma(BCached3, vec4<f32>(ACached.w), acc[i]);
            }
        }

        workgroupBarrier();
    }

    for (var innerRow = 0; innerRow < 4; innerRow++) {
        mm_write(batch, globalRow + innerRow, globalCol, acc[innerRow]);
    }
  }
