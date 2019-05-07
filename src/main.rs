extern crate nalgebra;
extern crate rand;
use nalgebra::*;
use rand::*;

mod net;
use net::*;

fn target1(x: bool, y: bool, z: bool) -> bool {
    (x == y) || (!x && z)
}

fn target(x: f32) -> f32 {
    (1.0 + x.sin()).ln()
}

fn main() {
    let mut a = net![3, 1];
    let train_data: Vec<_> = (0..8)
        .map(|a| ((a & 4) != 0, (a & 2) != 0, (a & 1) != 0))
        .map(|(x, y, z)|((x, y, z), target1(x, y, z)))
        .map(|((x, y, z), r)| ((x as i8 as f32, y as i8 as f32, z as i8 as f32), r as i8 as f32))
        .map(|((x, y, z), r)| DataPoint {
            input:  dvec![x, y, z],
            output: dvec![r],
        })
        .collect();

    println!("untrained");
    for x in &train_data {
        let res = a.predict(x.input.clone());
        println!("expected: {0} got {1}", x.output, res);
    }
    a.train(&train_data, TrainingParameters {epochs: 10000, learning_rate: 1.0});
    println!("trained");
    for x in &train_data {
        let res = a.predict(x.input.clone());
        println!("expected: {0} got {1}", x.output, res);
    }
}
