extern crate nalgebra;
extern crate rand;

use nalgebra::*;
use rand::*;

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

type Layer = DMatrix<f32>;

pub struct Net {
    layers: Vec<Layer>
}

use std::fmt::{Display, Formatter, Error};

impl Display for Net {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        for layer in &self.layers {
            write!(f, "{}", layer)?;
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
            layers.push(DMatrix::new_random(sizes[i + 1], sizes[i]));
        }
        Net {layers}
    }

    pub fn train<T: IntoIterator<Item=DataPoint>>(&mut self, data: T) {
        self._train(data.into_iter().map(|x|x.into()).collect());
    }

    fn _train(&mut self, data: Vec<_DataPoint>) {
        self.backprop(&data[2]);
    }

    fn activations(&self, input: &DVector<f32>) -> Vec<CalculatedLayer> {
        let mut result = Vec::with_capacity(self.layers.len());
        let not_activated = &self.layers[0] * input;
        let mut activated = not_activated.clone().apply_into(sigmoid);
        result.push(CalculatedLayer{not_activated, activated});
        for i in 1..self.layers.len() {
            let not_activated = &self.layers[i] * &result[i-1].activated;
            let mut activated = not_activated.clone().apply_into(sigmoid);
            result.push(CalculatedLayer{not_activated, activated});
        }
        result
    }

    pub fn backprop(&self, dp: &_DataPoint) -> Vec<Layer> {
        let mut activations = self.activations(&dp.input);
        print!("~~~~~~~~~~~~~ activations ~~~~~~~~~~~~~~~~~~ {0}", dp.input.transpose());
        for x in &activations {
            print!("{0}", x.activated.transpose());
        }
        println!("==========================================",);
        let mut result: Vec<Layer> = Vec::new();
        result.resize(self.layers.len(), DMatrix::zeros(1,1));
        let last_activations = activations.last().unwrap();
        // we need to change weights proportionally
        // and not forget to count activation function impact
        let mut desired_change_before_activation =
            (&dp.output - &last_activations.activated)
            .component_mul(
                &last_activations.not_activated.clone().apply_into(sigmoid_derivative)
            );
        println!("desired change {0}", desired_change_before_activation);
        let weights_change =
            &desired_change_before_activation
            * &activations.from_end(1).activated.transpose();
        println!("desired weights change {0}", weights_change);
        activations.insert(0, CalculatedLayer{activated: dp.input.clone(), not_activated:dvec![]});
        for index in (0..self.layers.len()-1).rev() {
            desired_change_before_activation =
                (&self.layers[index+1].transpose() * &desired_change_before_activation)
                .component_mul(
                    &activations[index+1].not_activated.clone().apply_into(sigmoid_derivative)
                );
            print!("desired change {0}", &desired_change_before_activation);
            print!("desired  weights change {0}",
                &desired_change_before_activation *
                &activations[index].activated.transpose()
            );
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
            input = x * input;
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
