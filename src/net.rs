extern crate nalgebra;
extern crate rand;

use nalgebra::*;
use rand::*;

pub struct DataPoint {
    pub input:  Vec<f32>,
    pub output: Vec<f32>,
}

impl DataPoint {
    pub fn new(input: Vec<f32>, output: Vec<f32>) -> DataPoint {
        DataPoint{input, output}
    }
}

pub struct Net {
    layers: Vec<DMatrix<f32>>
}

impl Net {
    pub fn new(sizes: &[usize]) -> Net {
        let mut layers = vec![];
        for i in 0..sizes.len()-1 {
            layers.push(DMatrix::new_random(sizes[i + 1], sizes[i]));
        }
        Net {layers}
    }

    pub fn train<T: IntoIterator<Item=DataPoint>>(&mut self, data: T) {
        unimplemented!()
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
        }
        input
    }
}
