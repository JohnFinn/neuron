extern crate nalgebra;
extern crate rayon;

use nalgebra::*;
use rayon::prelude::*;

mod layer;
use layer::*;

#[derive(Clone)]
pub struct DataPoint {
    pub input:  DVector<f32>,
    pub output: DVector<f32>,
}

struct CalculatedLayer {
    not_activated: DVector<f32>,
    activated: DVector<f32>,
}

pub struct TrainingParameters {
    pub epochs: usize,
    pub learning_rate: f32
}

pub struct Net {
    layers: Vec<Layer>
}

use std::fmt::{Display, Formatter, Error};
use std::collections::vec_deque::VecDeque;

impl Display for Net {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        for layer in &self.layers {
            write!(f, "{}", layer.weights)?;
        }
        Ok(())
    }
}

trait FromEnd<T> {
    fn from_end(&self, index: usize) -> &T;
}

impl<T> FromEnd<T> for [T] {
    fn from_end(&self, index: usize) -> &T {
        &self[self.len()-1-index]
    }
}

#[macro_export]
macro_rules! dvec {
    ( $( $x:expr ),* ) => {
        {
            nalgebra::DVector::from_vec(vec!($($x),*))
        }
    };
}

impl Net {
    pub fn new(sizes: &[usize]) -> Net {
        assert!(sizes.len() >= 2);
        let mut layers = vec![];
        for i in 0..sizes.len()-1 {
            layers.push(Layer::new_random(sizes[i], sizes[i+1]));
        }
        Net {layers}
    }

    pub fn train(&mut self, data: &Vec<DataPoint>, parameters: TrainingParameters) {
        for _ in 0..parameters.epochs {
            let changes =
                data.par_iter()
                    .map(|dp| self.backprop(dp))
                    .reduce(||self.zero_changes(), |mut acc, item| {
                        for (a, i) in acc.iter_mut().zip(item) {
                            *a += i;
                        }
                        acc
                    });
            for (l, mut x) in self.layers.iter_mut().zip(changes) {
                x *= parameters.learning_rate / data.len() as f32;
                *l += x;
            }
        }
    }

    fn zero_changes(&self) -> VecDeque<Layer> {
        self.layers.iter()
            .map(Layer::shape)
            .map(|(inputs, outputs)| Layer::zeros(inputs, outputs))
            .collect()
    }

    fn activations(&self, input: &DVector<f32>) -> Vec<CalculatedLayer> {
        let mut result = Vec::with_capacity(self.layers.len() + 1);
        result.push(CalculatedLayer{activated: input.clone(), not_activated:dvec![]});
        for i in 0..self.layers.len() {
            let not_activated = self.layers[i].calculate(&result[i].activated);
            let activated = not_activated.clone().apply_into(sigmoid);
            result.push(CalculatedLayer{not_activated, activated});
        }
        result
    }

    pub fn backprop(&self, dp: &DataPoint) -> VecDeque<Layer> {
        let activations = self.activations(&dp.input);
        let mut result = VecDeque::with_capacity(self.layers.len());
        let last_activations = activations.last().unwrap();
        // we need to change weights proportionally
        // and not forget to count activation function impact
        let mut desired_change_before_activation =
            (&dp.output - &last_activations.activated)
            .component_mul(
                &last_activations.not_activated.clone().apply_into(sigmoid_derivative)
            );
        let weights_change =
            &desired_change_before_activation
            * &activations.from_end(1).activated.transpose();
        result.push_front(Layer {
            weights: weights_change,
            biases: desired_change_before_activation.clone()
        });
        for index in (0..self.layers.len()-1).rev() {
            desired_change_before_activation =
                (&self.layers[index+1].weights.transpose() * &desired_change_before_activation)
                .component_mul(
                    &activations[index+1].not_activated.clone().apply_into(sigmoid_derivative)
                );
            let weights_change =
                &desired_change_before_activation *
                &activations[index].activated.transpose();
            result.push_front(Layer {
                weights: weights_change,
                biases: desired_change_before_activation.clone()
            });
        }
        result
    }

    pub fn predict(&self, mut input: DVector<f32>) -> DVector<f32> {
        for x in &self.layers {
            input = x.calculate(&input);
            input.apply(sigmoid);
        }
        input
    }
}

#[macro_export]
macro_rules! net {
    ( $( $x:expr ),* ) => {
        {
            Net::new(&[$($x),*])
        }
    };
}

pub fn sigmoid(x: f32) -> f32 {
    1.0/(1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}
