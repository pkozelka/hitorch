use cudarc::driver::CudaDevice;
use cudarc::nccl::{group_start, Comm, ReduceOp};

fn main() {

    let n = 2;
    let n_devices = CudaDevice::count().unwrap() as usize;
    let devices : Vec<_> = (0..n_devices).flat_map(CudaDevice::new).collect();
    let comms = Comm::from_devices(devices).unwrap();
    group_start().unwrap();
    let a = (0..n_devices).map(|i| {
        let comm = &comms[i];
        let dev = comm.device();
        let slice = dev.htod_copy(vec![(i + 1) as f32 * 1.0; n]).unwrap();
        let mut slice_receive = dev.alloc_zeros::<f32>(n).unwrap();
        comm.all_reduce(&slice, &mut slice_receive, &ReduceOp::Sum)
            .unwrap();
    });
    group_start().unwrap();
}