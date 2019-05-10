extern crate nalgebra;

use nalgebra::{DMatrix, DVector};

pub struct Layer {
    pub weights: DMatrix<f32>,
    pub biases:  DVector<f32>
}

impl Layer {
    pub fn new_random(inputs: usize, outputs: usize) -> Layer {
        Layer{
            weights: DMatrix::new_random(outputs, inputs),
            biases:  DVector::new_random(outputs)
        }
    }

    pub fn zeros(inputs: usize, outputs: usize) -> Layer {
        Layer {
            weights: DMatrix::zeros(outputs, inputs),
            biases:  DVector::zeros(outputs)
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.weights.ncols(), self.weights.nrows())
    }

    pub fn calculate(&self, input: &DVector<f32>) -> DVector<f32> {
        &self.weights * input + &self.biases
    }
}

impl std::ops::AddAssign<Layer> for Layer {
    fn add_assign(&mut self, rhs: Layer) {
        self.weights += rhs.weights;
        self.biases  += rhs.biases;
    }
}
