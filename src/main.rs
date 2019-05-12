extern crate gnuplot;
extern crate itertools_num;

use std::thread;

use gnuplot::*;
use itertools_num::*;
use nalgebra::DVector;

use net::*;

mod net;

fn target1(x: bool, y: bool, z: bool) -> bool {
    (!x || !y) && (!x || z)
}

fn target(x: f32) -> f32 {
    (1.0 + x.sin()).ln()
}

fn sigmoid_reversed(x: f32) -> f32 {
    (x / (1. - x)).ln()
}

macro_rules! to_dvec {
    ( $( $x:expr ),* ) => {
        {
            dvec!($($x as i32 as f32),*)
        }
    };
}

fn main() {
    thread::spawn(train_bool_function);
    train_float_function();
}

fn train_bool_function() {
    let mut a = net![3, 6, 1];
    let train_data: Vec<_> = (0..8)
        .map(|a| ((a & 4) != 0, (a & 2) != 0, (a & 1) != 0))
        .map(|(x, y, z)| DataPoint {
            input: to_dvec![x, y, z],
            output: to_dvec![target1(x, y, z)],
        })
        .collect();

    println!("untrained");
    for x in &train_data {
        let res = a.predict(x.input.clone());
        println!("expected: {0} got {1}", x.output[0], res[0]);
    }
    a.train(&train_data, TrainingParameters { epochs: 10000, learning_rate: 1.0 });
    println!("trained");
    for x in &train_data {
        let res = a.predict(x.input.clone());
        println!("expected: {0} got {1}", x.output[0], res[0]);
    }
}

fn get_points<F: Fn(f32) -> f32>(a: f32, b: f32, n: usize, func: F) -> (DVector<f32>, DVector<f32>) {
    let ls = linspace::<f32>(a, b, n);
    let x = DVector::from_iterator(n, ls.clone());
    let y = DVector::from_iterator(n, ls.clone().map(func));
    (x, y)
}

fn error(a: &DVector<f32>, b: &DVector<f32>) -> f32 {
    (b - a).apply_into(f32::abs).sum() / a.len() as f32
}

fn train_float_function() {
    let mut figure = Figure::new();
    let mut error_figure = Figure::new();
    let (draw_x, draw_y) = get_points(-10., 10., 1000, target);
    let train_data = linspace::<f32>(-10., 10., 100)
        .map(|x| DataPoint {
            input:  dvec![x],
            output: dvec![sigmoid(target(x))],
        })
        .collect();
    let mut net = net![1, 16, 1];
    let mut errors = Vec::new();
    for i in 0..3000 {
        net.train(&train_data, TrainingParameters { epochs: 300, learning_rate: 5.0 });
        let predicted = DVector::from_iterator(
            draw_x.len(),
            draw_x.iter().map(|&x| sigmoid_reversed(net.predict(dvec![x])[0]))
        );
        let err = error(&predicted, &draw_y);
        errors.push(err);
        error_figure.clear_axes()
            .axes2d()
            .lines(0..=i, &errors, &[]);
        error_figure.show();
        figure.clear_axes()
            .axes2d()
            .lines(&draw_x, &draw_y, &[])
            .lines(&draw_x, &predicted, &[gnuplot::Caption(&format!("{0:.5}", err))])
        ;
        figure.show();
    }
}
