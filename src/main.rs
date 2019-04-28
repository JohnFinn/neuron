extern crate nalgebra;
use nalgebra::*;

mod net;
use net::*;

fn target(x: f32) -> f32 {
    (1.0 + x.sin()).ln()
}

fn main() {
    let dm = DMatrix::from_row_slice(4, 3, &[
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0
    ]);
//    let mut a = Net::new(&[2, 3]);
//    let tdada = vec![
//        DataPoint{input: 0.0, output: 0.0},
//        DataPoint{input: 0.0, output: 0.0},
//    ];
//    a.train((0..100).map(|a| DataPoint{input: 0.0, output: 0.0}));
//    a.predict(2.0);
//    println!("{0}", a.layer1);
}
