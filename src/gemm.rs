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

pub fn gemm_1(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    tera.add_raw_template("gemm_1.wgsl", include_str!("../shaders/gemm/gemm_1.wgsl"))
        .unwrap();
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
    let shader = tera.render("gemm_1.wgsl", &context).unwrap();
    println!("shader: {}", shader);
    (workload, shader)
}

pub fn gemm_1v(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    tera.add_raw_template("gemm_1v.wgsl", include_str!("../shaders/gemm/gemm_1v.wgsl"))
        .unwrap();
    let workgroup_size_x = 16;
    let workgroup_size_y = 16 / 4;
    let workgroup_size_z = 1;
    let workload = Workload::new(
        WorkgroupCount(Workload::ceil(M, 16) as _, Workload::ceil(N, 16) as _, 1),
        WorkgroupSize(workgroup_size_x, workgroup_size_y, workgroup_size_z),
    );
    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);
    let shader = tera.render("gemm_1v.wgsl", &context).unwrap();
    println!("shader: {}", shader);
    (workload, shader)
}

pub fn gemm_2(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    tera.add_raw_template("gemm_2.wgsl", include_str!("../shaders/gemm/gemm_2.wgsl"))
        .unwrap();
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
    let shader = tera.render("gemm_2.wgsl", &context).unwrap();
    (workload, shader)
}

pub fn gemm_3(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    tera.add_raw_template("gemm_3.wgsl", include_str!("../shaders/gemm/gemm_3.wgsl"))
        .unwrap();
    let BLOCKSIZE = 16;
    context.insert("BLOCKSIZE", &BLOCKSIZE);
    let workgroup_size_x = BLOCKSIZE * BLOCKSIZE;
    let workgroup_size_y = 1;
    let workgroup_size_z = 1;
    let workload = Workload::new(
        WorkgroupCount(
            Workload::ceil(M, BLOCKSIZE) as _,
            Workload::ceil(N, BLOCKSIZE) as _,
            1,
        ),
        WorkgroupSize(workgroup_size_x as _, workgroup_size_y, workgroup_size_z),
    );
    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);
    let shader = tera.render("gemm_3.wgsl", &context).unwrap();
    (workload, shader)
}

pub fn gemm_4(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    tera.add_raw_template("gemm_4.wgsl", include_str!("../shaders/gemm/gemm_4.wgsl"))
        .unwrap();
    let BM = 16;
    let BN = 16;
    let BK = 8;
    let TM = 2;

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
    println!("workload: {:?}", workload);
    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);
    let shader = tera.render("gemm_4.wgsl", &context).unwrap();
    println!("shader: {}", shader);
    (workload, shader)
}

pub fn gemm_5(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    tera.add_raw_template("gemm_5.wgsl", include_str!("../shaders/gemm/gemm_5.wgsl"))
        .unwrap();
    let BM = 32;
    let BN = 32;
    let BK = 16;
    let TM = 4;
    let TN = 4;

    context.insert("BM", &BM);
    context.insert("BN", &BN);
    context.insert("BK", &BK);
    context.insert("TM", &TM);
    context.insert("TN", &TN);

    let workgroup_size_x = (BM * BN) / (TM * TN);
    let workgroup_size_y = 1;
    let workgroup_size_z = 1;
    let workload = Workload::new(
        WorkgroupCount(Workload::ceil(N, BN) as _, Workload::ceil(M, BM) as _, 1),
        WorkgroupSize(workgroup_size_x as _, workgroup_size_y, workgroup_size_z),
    );
    println!("workload: {:?}", workload);
    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);
    let shader = tera.render("gemm_5.wgsl", &context).unwrap();
    println!("shader: {}", shader);
    (workload, shader)
}

pub fn gemm_6(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    tera.add_raw_template("gemm_6.wgsl", include_str!("../shaders/gemm/gemm_6.wgsl"))
        .unwrap();
    let BM = 64;
    let BN = 64;
    let BK = 16
    let TM = 16;
    let TN = 1;

    context.insert("BM", &BM);
    context.insert("BN", &BN);
    context.insert("BK", &BK);
    context.insert("TM", &TM);
    context.insert("TN", &TN);

    let workgroup_size_x = (BM * BN) / (TM * TN);
    let workgroup_size_y = 1;
    let workgroup_size_z = 1;
    let workload = Workload::new(
        WorkgroupCount(Workload::ceil(N, BN) as _, Workload::ceil(M, BM) as _, 1),
        WorkgroupSize(workgroup_size_x as _, workgroup_size_y, workgroup_size_z),
    );
    println!("workload: {:?}", workload);
    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);
    let shader = tera.render("gemm_6.wgsl", &context).unwrap();
    println!("shader: {}", shader);
    (workload, shader)
}

#[cfg(test)]
mod tests {
    use crate::test_harness;

    use super::*;

    macro_rules! gemm_test {
        ($test_name:ident, $gemm_function:ident) => {
            #[tokio::test]
            pub async fn $test_name() {
                let _ = env_logger::builder().is_test(true).try_init();
                let mut tera = tera::Tera::default();
                let mut context = tera::Context::new();
                let dims = insert_matrix_dims(&mut context);
                let (workload, shader) = $gemm_function(&mut tera, &mut context);
                test_harness(workload, shader, dims, false).await;
            }
        };
    }

    gemm_test!(test_gemm_1, gemm_1);
    gemm_test!(test_gemm_1v, gemm_1v);
    gemm_test!(test_gemm_2, gemm_2);
    gemm_test!(test_gemm_3, gemm_3);
    gemm_test!(test_gemm_4, gemm_4);
    gemm_test!(test_gemm_5, gemm_5);
    gemm_test!(test_gemm_6, gemm_6);
}
