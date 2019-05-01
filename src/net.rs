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
        self._train(data.into_iter().map(|x|x.into()).collect());
    }

    fn _train(&mut self, data: Vec<_DataPoint>) {
        let cost = self.all_cost(data);
        println!("cost: {0}", cost);
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

    fn all_cost(&self, data: Vec<_DataPoint>) -> f32 {
        data.into_iter().map(|dp| self.cost(&dp)).sum()
    }

    fn cost(&self, dp: &_DataPoint) -> f32 {
        let mut a = self._predict(dp.input.clone()) - &dp.output;
        a.iter().map(|x|x*x).sum()
    }
}

fn cost(x: f32, y: f32) -> f32 {
    (x-y)*(x-y)
}
