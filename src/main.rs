#![allow(non_snake_case)]
use std::{borrow::Cow, time::Instant};

use matrixmultiply::sgemm;
use num_traits::Float;
use rand::{distributions::Standard, prelude::Distribution, Rng};
use tera::{Context, Tera};
use wgpu::{util::DeviceExt, InstanceDescriptor};
use wgpu_mm::{WorkgroupCount, Workload, WorkloadDim};

const M: usize = 1024;
const N: usize = 1024;
const K: usize = 1024;

pub fn check() -> bool {
    unsafe {
        sgemm(
            M,
            K,
            N,
            1.0,
            ap.as_ptr(),
            ar,
            ac,
            bp.as_ptr(),
            br,
            bc,
            1.0,
            cp.as_mut_ptr(),
            cr,
            cc,
        );
    }
    let mut max_diff = 0.0;
    for i in 0..m * n {
        let diff = (gpu_out[i] - c_cpu[i]).abs();
        assert!(diff < 0.0001);
        if diff > max_diff {
            max_diff = diff;
        }
    }
    if max_diff < 0.0001 {
        // println!("pass! max diff: {}", max_diff);
        true
    } else {
        println!("fail! max diff: {}", max_diff);
        println!("{:?} {:?}", gpu_out, c_cpu);
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

    let mut context = Context::new();
    context.insert("M", &M);
    context.insert("N", &N);
    context.insert("K", &K);

    let n_blocks = Workload::ceil(M * N, 4 * 4);
    let (x_count, x_size) = Workload::compute_dim(n_blocks, WorkloadDim::X);

    context.insert("workgroup_size_x", &x_size);
    context.insert("workgroup_size_y", &1);
    context.insert("workgroup_size_z", &1);

    let workgroup_count = WorkgroupCount(x_count, 1, 1);

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

    let (A, _) = rand_gpu_buffer::<f32>(&device, M * K, false);
    let (B, _) = rand_gpu_buffer::<f32>(&device, N * K, false);
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

    let warmup_res = to_cpu(&C, &device, &queue).await;
    println!("{:?}", warmup_res[0]);

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
