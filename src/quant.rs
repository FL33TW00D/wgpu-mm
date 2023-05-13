use num_traits::{AsPrimitive, Float};
use std::fmt::Debug;

/// Quantize a matrix of floats to 8-bit signed integers.
/// The AsPrimitive<i32> may seem confusing, we be need to do the bit masking
/// using signed integers, then cast to unsigned, to avoid losing negative values
pub fn sint8_quantize<F: Float + AsPrimitive<i32> + Debug>(
    matrix: &[F],
    K: usize,
    N: usize,
) -> (Vec<u32>, F) {
    assert!(matrix.len() == K * N);
    assert!(matrix.len() % 4 == 0);
    let block_size = 4;
    let mut quantized_matrix = vec![0u32; K * N / block_size];

    let absmax = matrix.iter().fold(F::zero(), |acc, &x| acc.max(x.abs()));
    let sf = F::from(127.).unwrap();

    for i in (0..(K * N)).step_by(block_size) {
        let packed_value: i32 = ((matrix[i] / absmax * sf).round().as_() & 0xFF)
            | (((matrix[i + 1] / absmax * sf).round().as_() & 0xFF) << 8)
            | (((matrix[i + 2] / absmax * sf).round().as_() & 0xFF) << 16)
            | (((matrix[i + 3] / absmax * sf).round().as_() & 0xFF) << 24);
        quantized_matrix[i / block_size] = packed_value as u32
    }
    (quantized_matrix, absmax)
}

pub fn sint8_dequantize(quantized_matrix: &[u32], absmax: f32, K: usize, N: usize) -> Vec<f32> {
    let block_size = 4;
    let mut matrix = vec![0.0; K * N];

    for i in (0..(K * N)).step_by(block_size) {
        let packed_value = quantized_matrix[i.div_floor(block_size)] as i32;
        matrix[i] = ((packed_value << 24) >> 24) as f32 / 127.0 * absmax;
        matrix[i + 1] = ((packed_value << 16) >> 24) as f32 / 127.0 * absmax;
        matrix[i + 2] = ((packed_value << 8) >> 24) as f32 / 127.0 * absmax;
        matrix[i + 3] = (packed_value >> 24) as f32 / 127.0 * absmax;
    }

    matrix
}

#[cfg(test)]
mod tests {
    #[test]
    pub fn test_qdq() {
        let matrix = vec![
            0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 1.2, -1.2, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 1.2, -1.2,
        ];
        println!("unquant: {:>12?}", matrix);
        let (quantized_matrix, absmax) = super::sint8_quantize(&matrix, 4, 4);
        assert_eq!(quantized_matrix.len(), 4);
        assert_eq!(
            quantized_matrix,
            vec![3409310987, 2172622442, 3409310987, 2172622442]
        );
        let dequantized_matrix = super::sint8_dequantize(&quantized_matrix, absmax, 4, 4);
        println!("dequant: {:>12?}", dequantized_matrix);
        for i in 0..matrix.len() {
            assert!((matrix[i] - dequantized_matrix[i]).abs() < 0.01);
        }
    }
}
