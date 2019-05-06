extern crate nalgebra;
extern crate rand;

use nalgebra::*;
use rand::*;

#[derive(Clone)]
pub struct DataPoint {
    pub input:  Vec<f32>,
    pub output: Vec<f32>,
}

struct _DataPoint {
    input:  DVector<f32>,
    output: DVector<f32>,
}

impl From<DataPoint> for _DataPoint {
    fn from(dp: DataPoint) -> Self {
        _DataPoint {
            input: DVector::from_vec(dp.input),
            output: DVector::from_vec(dp.output)
        }
    }
}

impl DataPoint {
    pub fn new(input: Vec<f32>, output: Vec<f32>) -> DataPoint {
        DataPoint{input, output}
    }
}

struct CalculatedLayer {
    not_activated: DVector<f32>,
    activated: DVector<f32>,
}

struct Layer {
    weights: DMatrix<f32>,
    biases:  DVector<f32>
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
            layers.push(Layer{
                weights: DMatrix::new_random(sizes[i + 1], sizes[i]),
                biases:  DVector::new_random(sizes[i + 1])
            });
        }
        Net {layers}
    }

    pub fn train<T: IntoIterator<Item=DataPoint>>(&mut self, data: T) {
        self._train(data.into_iter().map(|x|x.into()).collect());
    }

    fn _train(&mut self, data: Vec<_DataPoint>) {
        for i in 0..100000 {
            let mut changes =
                data.iter()
                    .map(|dp| self.backprop(dp))
                    .fold(self.zero_changes(), |mut acc, item| {
                        for (a, i) in acc.iter_mut().zip(item) {
                            a.weights += i.weights;
                            a.biases  += i.biases;
                        }
                        acc
                    });
            for (l, x) in self.layers.iter_mut().zip(changes) {
                l.weights += x.weights.apply_into(|x| x * 0.1 / data.len() as f32);
                l.biases  += x.biases .apply_into(|x| x * 0.1 / data.len() as f32);
            }
        }
    }

    fn zero_changes(&self) -> VecDeque<Layer> {
        self.layers.iter()
            .map(|l| l.weights.shape())
            .map(|(r, c)| Layer {
                weights: DMatrix::zeros(r, c),
                biases:  DVector::zeros(r)
            })
            .collect()
    }

    fn activations(&self, input: &DVector<f32>) -> Vec<CalculatedLayer> {
        let mut result = Vec::with_capacity(self.layers.len());
        let not_activated = &self.layers[0].weights * input + &self.layers[0].biases;
        let mut activated = not_activated.clone().apply_into(sigmoid);
        result.push(CalculatedLayer{not_activated, activated});
        for i in 1..self.layers.len() {
            let not_activated = &self.layers[i].weights * &result[i-1].activated + &self.layers[i].biases;
            let mut activated = not_activated.clone().apply_into(sigmoid);
            result.push(CalculatedLayer{not_activated, activated});
        }
        result
    }

    pub fn backprop(&self, dp: &_DataPoint) -> VecDeque<Layer> {
        let mut activations = self.activations(&dp.input);
        activations.insert(0, CalculatedLayer{activated: dp.input.clone(), not_activated:dvec![]});
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

    pub fn predict(&self, input: Vec<f32>) -> Vec<f32> {
        self._predict(DVector::from_vec(input))
            .iter()
            .map(|&x|x)
            .collect()
    }

    fn _predict(&self, mut input: DVector<f32>) -> DVector<f32> {
        for x in &self.layers {
            input = &x.weights * input + &x.biases;
            input.apply(sigmoid);
        }
        input
    }

    fn all_cost(&self, data: Vec<_DataPoint>) -> f32 {
        data.into_iter().map(|dp| self.cost(&dp)).sum()
    }

    fn cost(&self, dp: &_DataPoint) -> f32 {
        let mut a = self._predict(dp.input.clone()) - &dp.output;
        a.iter().map(|x|x*x).sum()
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

fn cost(x: f32, y: f32) -> f32 {
    (x-y)*(x-y)
}

fn sigmoid(x: f32) -> f32 {
    1.0/(1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}
