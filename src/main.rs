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
    println!("untrained");
    for x in &train_data {
        let res = a.predict(x.output.clone());
        println!("expected: {0:?} got {1:?}", x.input, res);
    }
    a.train(train_data.clone());
    println!("trained");
    for x in &train_data {
        let res = a.predict(x.output.clone());
        println!("expected: {0:?} got {1:?}", x.input, res);
    }
    let res = a.predict(vec![1.0, 0.0]);
    println!("~~~~~~~~~~~~~ network ~~~~~~~~~~~~~~~~~~ {0} =================================", a);
}
