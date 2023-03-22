#![allow(non_snake_case)]
use std::{borrow::Cow, time::Instant};

use num_traits::Float;
use rand::{distributions::Standard, prelude::Distribution, Rng};
use tera::{Context, Tera};
use wgpu::{util::DeviceExt, InstanceDescriptor};

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
) -> wgpu::Buffer
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
    buffer
}

#[tokio::main]
async fn main() {
    let _ = env_logger::builder().try_init();
    const M: usize = 1024;
    const N: usize = 1024;
    const K: usize = 1024;

    let (device, queue) = gpu_handle().await;
    let mut tera = Tera::default();
    tera.add_raw_template(
        "gemm_macro.wgsl",
        include_str!("../shaders/gemm_macro.wgsl"),
    )
    .unwrap();
    tera.add_raw_template("gemm.wgsl", include_str!("../shaders/gemm.wgsl"))
        .unwrap();

    let mut context = Context::new();
    context.insert("M", &M);
    context.insert("N", &N);
    context.insert("K", &K);
    context.insert("workgroup_size_x", &2);
    context.insert("workgroup_size_y", &1);
    context.insert("workgroup_size_z", &1);

    let shader = tera.render("gemm.wgsl", &context).unwrap();

    let shader_module = unsafe {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
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

    let A = rand_gpu_buffer::<f32>(&device, M * K);
    let B = rand_gpu_buffer::<f32>(&device, N * K);
    let C = rand_gpu_buffer::<f32>(&device, M * N);

    //warmup
    queue.submit(vec![
        mm(&device, &pipeline, &A, &B, &C),
        mm(&device, &pipeline, &C, &B, &A),
        mm(&device, &pipeline, &A, &C, &B),
        mm(&device, &pipeline, &B, &A, &C),
        mm(&device, &pipeline, &A, &B, &C),
        mm(&device, &pipeline, &C, &B, &A),
        mm(&device, &pipeline, &A, &C, &B),
        mm(&device, &pipeline, &B, &A, &C),
    ]);

    let warmup_res = to_cpu(&C, &device, &queue).await;
    println!("{:?}", warmup_res[0]);

    let start = Instant::now();
    queue.submit(vec![
        mm(&device, &pipeline, &A, &B, &C),
        mm(&device, &pipeline, &C, &B, &A),
        mm(&device, &pipeline, &A, &C, &B),
        mm(&device, &pipeline, &B, &A, &C),
        mm(&device, &pipeline, &A, &B, &C),
        mm(&device, &pipeline, &C, &B, &A),
        mm(&device, &pipeline, &A, &C, &B),
        mm(&device, &pipeline, &B, &A, &C),
        mm(&device, &pipeline, &A, &B, &C),
        mm(&device, &pipeline, &B, &A, &C),
        mm(&device, &pipeline, &C, &B, &A),
    ]);

    let result = to_cpu(&C, &device, &queue).await;
    println!("{:?}", result[0]);

    let elapsed = start.elapsed();

    let millis = elapsed.as_millis();
    println!("{} ms", millis);
    let flops = M * N * K * 2 * 10;
    println!("{} FLOPS", flops);
    let gflops = flops as f64 / (millis as f64 * 1e6);
    println!("{} GFLOPS", gflops);
}

pub fn mm(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    A: &wgpu::Buffer,
    B: &wgpu::Buffer,
    C: &wgpu::Buffer,
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

        cpass.dispatch_workgroups(32768, 1, 1);
    }
    encoder.finish()
}

pub async fn to_cpu(buffer: &wgpu::Buffer, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
    let buffer_slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    println!("reading buffer");
    println!("buffer size: {:?}", buffer_slice);

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
