#![allow(non_snake_case)]
use std::{borrow::Cow, fmt::Debug, time::Instant};

use num_traits::{AsPrimitive, Float};
use rand::{
    distributions::{uniform::SampleUniform, Standard, Uniform},
    prelude::Distribution,
    Rng,
};
use wgpu::{util::DeviceExt, InstanceDescriptor};

use crate::{WorkgroupCount, Workload};

fn mm_ref(A: &[f32], B: &[f32], C: &mut [f32], dims: (usize, usize, usize)) {
    let (M, N, K) = dims;
    for m in 0..M {
        for n in 0..N {
            let mut res = 0.;
            for k in 0..K {
                res += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = res;
        }
    }
}

fn quant_mm_ref(A: &[f32], B: &[u32], C: &mut [f32], dims: (usize, usize, usize)) {
    let (M, N, K) = dims;
    for m in 0..M {
        for n in 0..N {
            let mut res = 0;
            for k in 0..K {
                res += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = res;
        }
    }
}

async fn check(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    workgroup_count: &WorkgroupCount,
    dims: (usize, usize, usize),
) {
    let (M, N, K) = dims;
    let BufferResult {
        gpu_buf: A,
        cpu_buf: A_cpu,
        ..
    } = rand_gpu_buffer::<f32>(&device, (M, N), true);
    let BufferResult {
        gpu_buf: B,
        cpu_buf: B_cpu,
        ..
    } = rand_gpu_buffer::<f32>(&device, (K, N), true);
    let BufferResult {
        gpu_buf: C,
        cpu_buf: C_cpu,
        ..
    } = rand_gpu_buffer::<f32>(&device, (M, N), true);
    let mut C_cpu = C_cpu.unwrap();

    mm_ref(&A_cpu.unwrap(), &B_cpu.unwrap(), &mut C_cpu, dims);

    let mm = mm(&device, &pipeline, &A, &B, &C, &workgroup_count);
    queue.submit(vec![mm]);
    let gpu_out = to_cpu(&C, &device, &queue).await;

    let mut mae = 0.0;
    for i in 0..M * N {
        let diff = (gpu_out[i] - C_cpu[i]).abs();
        if diff > mae {
            mae = diff;
        }
    }
    println!(
        "GPU\n{:?}\n...\n{:?}",
        &gpu_out[..16],
        &gpu_out[M * N - 16..]
    );
    println!("CPU\n{:?}\n...\n{:?}", &C_cpu[..16], &C_cpu[M * N - 16..]);
    println!("Max Absolute Error: {}", mae);
}

async fn gpu_handle() -> (wgpu::Device, wgpu::Queue) {
    let backends = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY);

    let instance = wgpu::Instance::new(InstanceDescriptor {
        backends,
        ..Default::default()
    });
    let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, backends, None)
        .await
        .expect("No GPU found given preference");
    adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .expect("Could not create adapter for GPU device")
}

#[derive(derive_new::new)]
struct BufferResult<F: Float + bytemuck::Pod> {
    gpu_buf: wgpu::Buffer,
    cpu_buf: Option<Vec<F>>,
}

fn sint8_quantize<F: Float + AsPrimitive<u32> + Debug>(
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

fn generate_weight_data<F: Float + bytemuck::Pod + AsPrimitive<u32> + Debug>(
    M: usize,
    N: usize,
) -> Vec<F>
where
    Standard: Distribution<F>,
    F: SampleUniform,
{
    let mut rng = rand::thread_rng();
    let dist = Uniform::from(F::from(-10.0).unwrap()..F::from(10.0).unwrap());
    let mut data = vec![F::zero(); M * N];
    for i in 0..M {
        for j in 0..N {
            data[i * N + j] = rng.sample(dist) / F::from(50).unwrap();
        }
    }
    data
}

fn rand_quantized_gpu_buffer<F: Float + bytemuck::Pod + AsPrimitive<u32> + Debug>(
    device: &wgpu::Device,
    dims: (usize, usize),
    return_cpu: bool,
) -> BufferResult<F>
where
    Standard: Distribution<F>,
    F: SampleUniform,
{
    let (M, N) = dims;
    let data = generate_weight_data::<F>(M, N);
    let (quantized, _absmax) = sint8_quantize(&data, M, N);
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&quantized),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    BufferResult::new(buffer, if return_cpu { Some(data) } else { None })
}

fn rand_gpu_buffer<F: Float + bytemuck::Pod + AsPrimitive<u32> + Debug>(
    device: &wgpu::Device,
    dims: (usize, usize),
    return_cpu: bool,
) -> BufferResult<F>
where
    Standard: Distribution<F>,
    F: SampleUniform,
{
    let (M, N) = dims;
    let data = generate_weight_data::<F>(M, N);
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    BufferResult::new(buffer, if return_cpu { Some(data) } else { None })
}

pub async fn test_harness(
    workload: Workload,
    shader: String,
    dims: (usize, usize, usize),
    quantize_b: bool,
) {
    let (device, queue) = gpu_handle().await;
    let (M, N, K) = dims;

    let shader_module = unsafe {
        device.create_shader_module_unchecked(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader)),
        })
    };

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader_module,
        entry_point: "main",
    });

    check(&device, &queue, &pipeline, &workload.count(), (M, N, K)).await;

    let BufferResult { gpu_buf: A, .. } = rand_gpu_buffer::<f32>(&device, (M, K), false);
    let B = if quantize_b {
        let buffer_result = rand_quantized_gpu_buffer::<f32>(&device, (K, N), false);
        let quantized_buffer = buffer_result.gpu_buf;
        quantized_buffer
    } else {
        rand_gpu_buffer::<f32>(&device, (K, N), false).gpu_buf
    };
    let BufferResult { gpu_buf: C, .. } = rand_gpu_buffer::<f32>(&device, (M, N), false);

    //warmup
    queue.submit(vec![
        mm(&device, &pipeline, &A, &B, &C, &workload.count()),
        mm(&device, &pipeline, &C, &B, &A, &workload.count()),
        mm(&device, &pipeline, &A, &C, &B, &workload.count()),
        mm(&device, &pipeline, &B, &A, &C, &workload.count()),
        mm(&device, &pipeline, &A, &B, &C, &workload.count()),
        mm(&device, &pipeline, &C, &B, &A, &workload.count()),
        mm(&device, &pipeline, &A, &C, &B, &workload.count()),
        mm(&device, &pipeline, &B, &A, &C, &workload.count()),
    ]);

    let _warmup_res = to_cpu(&C, &device, &queue).await;

    let start = Instant::now();
    queue.submit(vec![
        mm(&device, &pipeline, &A, &B, &C, &workload.count()),
        mm(&device, &pipeline, &C, &B, &A, &workload.count()),
        mm(&device, &pipeline, &A, &C, &B, &workload.count()),
        mm(&device, &pipeline, &B, &A, &C, &workload.count()),
        mm(&device, &pipeline, &A, &B, &C, &workload.count()),
        mm(&device, &pipeline, &C, &B, &A, &workload.count()),
        mm(&device, &pipeline, &A, &C, &B, &workload.count()),
        mm(&device, &pipeline, &B, &A, &C, &workload.count()),
        mm(&device, &pipeline, &A, &B, &C, &workload.count()),
        mm(&device, &pipeline, &B, &A, &C, &workload.count()),
    ]);

    let _result = to_cpu(&C, &device, &queue).await;

    let elapsed = start.elapsed();

    let nanos = elapsed.as_nanos();
    println!("{} ns", nanos);
    let flops = M * N * K * 2 * 10;
    let gflops = (flops as f64 / 1e9) / (nanos as f64 / 1e9);
    println!("{} GFLOPS", gflops);
}

fn mm(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    A: &wgpu::Buffer,
    B: &wgpu::Buffer,
    C: &wgpu::Buffer,
    workgroup_count: &WorkgroupCount,
) -> wgpu::CommandBuffer {
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: A.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: B.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: C.as_entire_binding(),
            },
        ],
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);

        cpass.dispatch_workgroups(workgroup_count.0, workgroup_count.1, workgroup_count.2);
    }
    encoder.finish()
}

async fn to_cpu(buffer: &wgpu::Buffer, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
    let buffer_slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);

    wgpu::util::DownloadBuffer::read_buffer(device, queue, &buffer_slice, move |buffer| {
        tx.send(match buffer {
            Ok(bytes) => bytemuck::cast_slice(&bytes)[..].to_vec(),
            _ => panic!("Error reading buffer"),
        })
        .unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap()
}
