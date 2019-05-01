extern crate nalgebra;
extern crate rand;
use nalgebra::*;
use rand::*;

mod net;
use net::*;

fn target(x: f32) -> f32 {
    (1.0 + x.sin()).ln()
}

fn main() {
    let mut a = net![2, 3, 1];
    let train_data = vec![
        DataPoint::new(vec![0.0, 0.0], vec![0.0]),
        DataPoint::new(vec![0.0, 1.0], vec![1.0]),
        DataPoint::new(vec![1.0, 0.0], vec![1.0]),
        DataPoint::new(vec![1.0, 1.0], vec![0.0]),
    ];

    a.train(train_data);
    let res = a.predict(vec![1.0, 0.0]);
    println!("{0:?}", res);
}
