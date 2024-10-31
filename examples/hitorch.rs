use tch::{kind, Tensor};

fn main() {
    println!("Cuda available: {}", tch::Cuda::is_available());
    println!("Cudnn available: {}", tch::Cuda::cudnn_is_available());
    let device = tch::Device::cuda_if_available();
    let t = Tensor::from_slice(&[3, 1, 4, 1, 5]).to(device);
    t.print();
    let t = Tensor::randn([15, 9], kind::FLOAT_CPU);
    t.print();
    // (&t + 1.5).print();
    // (&t + 2.5).print();
}
