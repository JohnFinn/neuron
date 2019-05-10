extern crate nalgebra;
extern crate rand;
extern crate gnuplot;
extern crate itertools_num;

use gnuplot::*;
use nalgebra::*;
use rand::*;
use itertools_num::*;
use std::thread;
use std::sync::mpsc;

mod net;
use net::*;
use std::time::Duration;

fn target1(x: bool, y: bool, z: bool) -> bool {
    (x == y) || (!x && z)
}

fn target2(x: f32) -> f32 {
    (1.0 + x.exp()).ln()
}

fn target(x: f32) -> f32 {
    (1.0 + x.sin()).ln() + 18.0
}

fn sigmoid_reversed(x: f32) -> f32 {
    (x/(1.-x)).ln()
}

macro_rules! to_dvec {
    ( $( $x:expr ),* ) => {
        {
            dvec!($($x as i32 as f32),*)
        }
    };
}

fn main() {
    let mut a = net![3, 6, 1];
    let train_data: Vec<_> = (0..8)
        .map(|a| ((a & 4) != 0, (a & 2) != 0, (a & 1) != 0))
        .map(|(x, y, z)| DataPoint {
            input:  to_dvec![x, y, z],
            output: to_dvec![target1(x, y, z)],
        })
        .collect();

    println!("untrained");
    for x in &train_data {
        let res = a.predict(x.input.clone());
        println!("expected: {0} got {1}", x.output[0], res[0]);
    }
    a.train(&train_data, TrainingParameters {epochs: 10000, learning_rate: 1.0});
    println!("trained");
    for x in &train_data {
        let res = a.predict(x.input.clone());
        println!("expected: {0} got {1}", x.output[0], res[0]);
    }

    let mut figure = Figure::new();

    let a = linspace::<f32>(-10., 10., 1000);
    let mut net2 = net![1, 16, 1];
    for i in 1..3000 {
        net2.train(
            &a.clone().map(|x|DataPoint{input: dvec![x], output: dvec![target(x)/20.0]}).collect(),
            TrainingParameters{epochs: 30, learning_rate: 10.0}
        );
        figure.clear_axes();
        figure.axes2d()
            .lines(a.clone(), a.clone().map(target), &[])
            .lines(a.clone(), a.clone().map(|x| net2.predict(dvec![x])[0]*20.0), &[])
        ;
        figure.show();
    }
}
