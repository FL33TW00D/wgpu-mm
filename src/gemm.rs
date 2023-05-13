use tera::{Context, Tera};

use crate::{WorkgroupCount, WorkgroupSize, Workload};

const M: usize = 1024;
const N: usize = 1024;
const K: usize = 1024;

pub fn insert_matrix_dims(context: &mut Context) -> (usize, usize, usize) {
    context.insert("M", &M);
    context.insert("N", &N);
    context.insert("K", &K);
    (M, N, K)
}

pub fn kernel_1(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    let workgroup_size_x = 16;
    let workgroup_size_y = 16;
    let workgroup_size_z = 1;
    let workload = Workload::new(
        WorkgroupCount(Workload::ceil(M, 16) as _, Workload::ceil(N, 16) as _, 1),
        WorkgroupSize(workgroup_size_x, workgroup_size_y, workgroup_size_z),
    );
    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);
    let shader = tera.render("kernel_1.wgsl", &context).unwrap();
    (workload, shader)
}

pub fn kernel_2(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    let workgroup_size_x = 256;
    let workgroup_size_y = 1;
    let workgroup_size_z = 1;
    let workload = Workload::new(
        WorkgroupCount(Workload::ceil(M, 16) as _, Workload::ceil(N, 16) as _, 1),
        WorkgroupSize(workgroup_size_x, workgroup_size_y, workgroup_size_z),
    );
    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);
    let shader = tera.render("kernel_2.wgsl", &context).unwrap();
    (workload, shader)
}

pub fn kernel_3(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    context.insert("BLOCKSIZE", &16);
    let workgroup_size_x = 256;
    let workgroup_size_y = 1;
    let workgroup_size_z = 1;
    let workload = Workload::new(
        WorkgroupCount(Workload::ceil(M, 16) as _, Workload::ceil(N, 16) as _, 1),
        WorkgroupSize(workgroup_size_x, workgroup_size_y, workgroup_size_z),
    );
    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);
    let shader = tera.render("kernel_3.wgsl", &context).unwrap();
    (workload, shader)
}

pub fn kernel_4(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    let BM = 16;
    let BN = 16;
    let BK = 8;
    let TM = 8;

    context.insert("BM", &BM);
    context.insert("BN", &BN);
    context.insert("BK", &BK);
    context.insert("TM", &TM);

    let workgroup_size_x = (BM * BN) / TM;
    let workgroup_size_y = 1;
    let workgroup_size_z = 1;
    let workload = Workload::new(
        WorkgroupCount(Workload::ceil(N, BN) as _, Workload::ceil(M, BM) as _, 1),
        WorkgroupSize(workgroup_size_x as _, workgroup_size_y, workgroup_size_z),
    );
    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);
    let shader = tera.render("kernel_4.wgsl", &context).unwrap();
    (workload, shader)
}
