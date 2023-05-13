use num_traits::{AsPrimitive, Float};
use std::fmt::Debug;

pub fn sint8_quantize<F: Float + AsPrimitive<u32> + Debug>(
    matrix: &[F],
    K: usize,
    N: usize,
) -> (Vec<u32>, F) {
    let block_size = 4;
    let mut quantized_matrix = vec![0u32; K * (N / block_size)];

    let absmax = matrix.iter().fold(F::zero(), |acc, &x| acc.max(x.abs()));
    println!("absmax: {:?}", absmax);
    let sf = F::from(127.).unwrap();

    for i in 0..K {
        for j in (0..N).step_by(block_size) {
            let packed_value = ((matrix[i * N + j] / absmax * sf).round().as_() & 0xFF)
                | (((matrix[i * N + j + 1] / absmax * sf).round().as_() & 0xFF) << 8)
                | (((matrix[i * N + j + 2] / absmax * sf).round().as_() & 0xFF) << 16)
                | (((matrix[i * N + j + 3] / absmax * sf).round().as_() & 0xFF) << 24);
            quantized_matrix[i * (N / block_size) + (j / block_size)] = packed_value;
        }
    }
    (quantized_matrix, absmax)
}

pub fn sint8_dequantize(quantized_matrix: &[u32], absmax: f32, M: usize, N: usize) -> Vec<f32> {
    let block_size = 4;
    let mut matrix = vec![0.0; M * N];

    for (i, &packed_value) in quantized_matrix.iter().enumerate() {
        let i = i * block_size;
        matrix[i] = ((packed_value << 24) >> 24) as f32 / 127.0 * absmax;
        matrix[i + 1] = ((packed_value << 16) >> 24) as f32 / 127.0 * absmax;
        matrix[i + 2] = ((packed_value << 8) >> 24) as f32 / 127.0 * absmax;
        matrix[i + 3] = (packed_value >> 24) as f32 / 127.0 * absmax;
    }

    matrix
}
