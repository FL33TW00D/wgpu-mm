#![allow(non_snake_case)]
use std::{borrow::Cow, time::Instant};

use num_traits::Float;
use rand::{distributions::Standard, prelude::Distribution, Rng};
use tera::{Context, Tera};
use wgpu::{util::DeviceExt, InstanceDescriptor};
use wgpu_mm::{WorkgroupCount, Workload, WorkloadDim};

const M: usize = 1024;
const N: usize = 1024;
const K: usize = 1024;

pub fn mm_ref(A: &[f32], B: &[f32], C: &mut [f32]) {
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

pub async fn check(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    workgroup_count: &WorkgroupCount,
) -> bool {
    let (A, A_cpu) = rand_gpu_buffer::<f32>(&device, M * K, true);
    let (B, B_cpu) = rand_gpu_buffer::<f32>(&device, K * N, true);
    let (C, C_cpu) = rand_gpu_buffer::<f32>(&device, M * N, true);
    let mut C_cpu = C_cpu.unwrap();

    mm_ref(&A_cpu.unwrap(), &B_cpu.unwrap(), &mut C_cpu);

    let mm = mm(&device, &pipeline, &A, &B, &C, &workgroup_count);
    queue.submit(vec![mm]);
    let gpu_out = to_cpu(&C, &device, &queue).await;

    let mut max_diff = 0.0;
    for i in 0..M * N {
        let diff = (gpu_out[i] - C_cpu[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    if max_diff < 0.0001 {
        true
    } else {
        println!("fail! max diff: {}", max_diff);
        println!("GPU: {:?}", &gpu_out[..32]);
        println!("CPU: {:?}", &C_cpu[..32]);
        false
    }
}

pub async fn gpu_handle() -> (wgpu::Device, wgpu::Queue) {
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

pub fn rand_gpu_buffer<F: Float + bytemuck::Pod>(
    device: &wgpu::Device,
    numel: usize,
    return_cpu: bool,
) -> (wgpu::Buffer, Option<Vec<F>>)
where
    Standard: Distribution<F>,
{
    let mut rng = rand::thread_rng();
    let mut data = vec![F::zero(); numel];
    for i in 0..numel {
        data[i] = F::from(rng.gen::<F>()).unwrap() / F::from(511.91).unwrap();
    }
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    if return_cpu {
        (buffer, Some(data))
    } else {
        (buffer, None)
    }
}

#[tokio::main]
async fn main() {
    let _ = env_logger::builder().try_init();

    let (device, queue) = gpu_handle().await;
    let mut tera = Tera::default();
    tera.add_raw_template(
        "gemm_macro.wgsl",
        include_str!("../shaders/gemm_macro.wgsl"),
    )
    .unwrap();
    tera.add_raw_template("gemm.wgsl", include_str!("../shaders/gemm.wgsl"))
        .unwrap();
    tera.add_raw_template("gemm2.wgsl", include_str!("../shaders/bram.wgsl"))
        .unwrap();
    tera.add_raw_template("gemm3.wgsl", include_str!("../shaders/gemm3.wgsl"))
        .unwrap();
    tera.add_raw_template("gemm3_fma.wgsl", include_str!("../shaders/gemm3_fma.wgsl"))
        .unwrap();
    tera.add_raw_template("chonk.wgsl", include_str!("../shaders/chonk.wgsl"))
        .unwrap();
    tera.add_raw_template("chonk2.wgsl", include_str!("../shaders/chonk2.wgsl"))
        .unwrap();

    let mut context = Context::new();
    context.insert("M", &M);
    context.insert("N", &N);
    context.insert("K", &K);

    let B_TILE_N = 2;
    let B_TILE_K = 4;
    let A_TILE_K = 1;
    let A_TILE_M = 4;
    context.insert("B_TILE_N", &B_TILE_N);
    context.insert("B_TILE_K", &B_TILE_K);
    context.insert("A_TILE_K", &A_TILE_K);
    context.insert("A_TILE_M", &A_TILE_M);
    context.insert("components", &['x', 'y', 'z', 'w']);

    let n_blocks = Workload::ceil(M * N, 4 * 4);
    let (x_count, x_size) = Workload::compute_dim(n_blocks, WorkloadDim::X);

    context.insert("workgroup_size_x", &4);
    context.insert("workgroup_size_y", &8);
    context.insert("workgroup_size_z", &1);

    //gemm3
    let workgroup_count = WorkgroupCount((N / 32) as u32, (M / 32) as u32, 1);
    //bram
    //let workgroup_count = WorkgroupCount(128, 32, 1);

    let shader = tera.render("chonk2.wgsl", &context).unwrap();
    println!("{}", shader);

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

    if !check(&device, &queue, &pipeline, &workgroup_count).await {
        //panic!("Matrix multiplication does not match reference implementation");
    } else {
        println!("Matrix multiplication matches reference implementation");
    }

    let (A, _) = rand_gpu_buffer::<f32>(&device, M * K, false);
    let (B, _) = rand_gpu_buffer::<f32>(&device, K * N, false);
    let (C, _) = rand_gpu_buffer::<f32>(&device, M * N, false);

    //warmup
    queue.submit(vec![
        mm(&device, &pipeline, &A, &B, &C, &workgroup_count),
        mm(&device, &pipeline, &C, &B, &A, &workgroup_count),
        mm(&device, &pipeline, &A, &C, &B, &workgroup_count),
        mm(&device, &pipeline, &B, &A, &C, &workgroup_count),
        mm(&device, &pipeline, &A, &B, &C, &workgroup_count),
        mm(&device, &pipeline, &C, &B, &A, &workgroup_count),
        mm(&device, &pipeline, &A, &C, &B, &workgroup_count),
        mm(&device, &pipeline, &B, &A, &C, &workgroup_count),
    ]);

    let _warmup_res = to_cpu(&C, &device, &queue).await;

    let start = Instant::now();
    queue.submit(vec![
        mm(&device, &pipeline, &A, &B, &C, &workgroup_count),
        mm(&device, &pipeline, &C, &B, &A, &workgroup_count),
        mm(&device, &pipeline, &A, &C, &B, &workgroup_count),
        mm(&device, &pipeline, &B, &A, &C, &workgroup_count),
        mm(&device, &pipeline, &A, &B, &C, &workgroup_count),
        mm(&device, &pipeline, &C, &B, &A, &workgroup_count),
        mm(&device, &pipeline, &A, &C, &B, &workgroup_count),
        mm(&device, &pipeline, &B, &A, &C, &workgroup_count),
        mm(&device, &pipeline, &A, &B, &C, &workgroup_count),
        mm(&device, &pipeline, &B, &A, &C, &workgroup_count),
        mm(&device, &pipeline, &C, &B, &A, &workgroup_count),
    ]);

    let _result = to_cpu(&C, &device, &queue).await;

    let elapsed = start.elapsed();

    let nanos = elapsed.as_nanos();
    println!("{} ns", nanos);
    let flops = M * N * K * 2 * 10;
    let gflops = (flops as f64 / 1e9) / (nanos as f64 / 1e9);
    println!("{} GFLOPS", gflops);
}

pub fn mm(
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

pub async fn to_cpu(buffer: &wgpu::Buffer, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
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
