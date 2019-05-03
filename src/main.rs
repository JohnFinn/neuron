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
    let mut a = net![2, 4, 5, 3, 2];
    println!("~~~~~~~~~~~~~ network ~~~~~~~~~~~~~~~~~~ {0} =================================", a);
    let train_data = vec![
        DataPoint::new(vec![0.0, 0.0], vec![0.0, 1.0]),
        DataPoint::new(vec![0.0, 1.0], vec![1.0, 0.0]),
        DataPoint::new(vec![1.0, 0.0], vec![1.0, 0.0]),
        DataPoint::new(vec![1.0, 1.0], vec![0.0, 1.0]),
    ];
    let res = a.predict(vec![1.0, 0.0]);
    println!("result: {0:?}", res);
    a.train(train_data);
}
