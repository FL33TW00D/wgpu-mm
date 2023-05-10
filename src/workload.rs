#[derive(Debug)]
pub struct WorkgroupCount(pub u32, pub u32, pub u32); //Analagous to gridDim in CUDA
#[derive(Debug)]
pub struct WorkgroupSize(pub u32, pub u32, pub u32); //Analagous to blockDim in CUDA

///The Workload represents the entire piece of work.
///For more read: https://surma.dev/things/webgpu/
#[derive(Debug)]
pub struct Workload {
    count: WorkgroupCount,
    size: WorkgroupSize,
}

impl Workload {
    pub fn new(count: WorkgroupCount, size: WorkgroupSize) -> Self {
        Self { count, size }
    }

    pub fn count(&self) -> &WorkgroupCount {
        &self.count
    }

    pub fn size(&self) -> &WorkgroupSize {
        &self.size
    }
}

///Used to determine which limit applies
#[derive(Debug, Clone)]
pub enum WorkloadDim {
    X,
    Y,
    Z,
}

impl Workload {
    pub const MAX_WORKGROUP_SIZE_X: usize = 256;
    pub const MAX_WORKGROUP_SIZE_Y: usize = 256;
    pub const MAX_WORKGROUP_SIZE_Z: usize = 64;
    pub const MAX_COMPUTE_WORKGROUPS_PER_DIMENSION: usize = 65535;

    pub fn ceil(num: usize, div: usize) -> usize {
        (num + div - 1) / div
    }

    ///Given a number of work items that need to be processed
    ///Calculate the appropriate size and number of workgroups for a single dimension
    pub fn compute_dim(work_items: usize, dim: WorkloadDim) -> (u32, u32) {
        let max_workgroup_size = match dim {
            WorkloadDim::X => Self::MAX_WORKGROUP_SIZE_X,
            WorkloadDim::Y => Self::MAX_WORKGROUP_SIZE_Y,
            WorkloadDim::Z => Self::MAX_WORKGROUP_SIZE_Z,
        };
        let max_workgroup_count = Self::MAX_COMPUTE_WORKGROUPS_PER_DIMENSION;
        if work_items > max_workgroup_count {
            let workgroup_size = Self::ceil(work_items, max_workgroup_count);
            let workgroup_count = Self::ceil(work_items, workgroup_size);

            if workgroup_count > max_workgroup_count || workgroup_size > max_workgroup_size {
                panic!("Compute limits exceeded");
            }
            (workgroup_count as u32, workgroup_size as u32)
        } else {
            //If the number of work_items is such that we can dispatch one workgroup per work item
            //then we can just use the number of work items as the number of workgroups
            (work_items as u32, 1)
        }
    }
}
