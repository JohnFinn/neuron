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
    layers: Vec<DVector<f32>>
}

impl Net {
    pub fn new(sizes: &[usize]) -> Net {
        Net {
            layers: sizes
                .iter()
                .map(|&l|
                    DVector::from_fn(l, |w, h|
                        thread_rng().gen_range(0.0, 1.0)
                    )
                )
                .collect()
        }
    }

    pub fn train<T: IntoIterator<Item=DataPoint>>(&mut self, data: T) {
        unimplemented!()
    }

    pub fn predict(&self, input: Vec<f32>) -> Vec<f32> {
        unimplemented!()
    }
}
