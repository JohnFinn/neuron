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
    println!("~~~~~~~~~~~~~ network ~~~~~~~~~~~~~~~~~~ {0} =================================", a);
    let train_data = vec![
        DataPoint::new(dvec![0.0, 0.0], dvec![0.0]),
        DataPoint::new(dvec![0.0, 1.0], dvec![1.0]),
        DataPoint::new(dvec![1.0, 0.0], dvec![1.0]),
        DataPoint::new(dvec![1.0, 1.0], dvec![0.0]),
    ];
    println!("untrained");
    for x in &train_data {
        let res = a.predict(x.input.clone());
        println!("expected: {0} got {1}", x.output, res);
    }
    a.train(train_data.clone(), TrainingParameters {epochs: 10000, learning_rate: 1.0});
    println!("trained");
    for x in &train_data {
        let res = a.predict(x.input.clone());
        println!("expected: {0} got {1}", x.output, res);
    }
    let res = a.predict(dvec![1.0, 0.0]);
    println!("~~~~~~~~~~~~~ network ~~~~~~~~~~~~~~~~~~ {0} =================================", a);
}
