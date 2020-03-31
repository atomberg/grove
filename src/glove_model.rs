extern crate num;
extern crate rand;

use super::sparse::Serialize;
use hogwild::{HogwildArray1, HogwildArray2};
use ndarray::{Array1, Array2};
use num::Float;
use rand::{distributions::Uniform, Rng};

// Unsafe arrays for Hogwild method of parallel Stochastic Gradient descent
#[derive(Clone)]
pub struct Model<F: Float + Serialize> {
    pub focus_vectors: HogwildArray2<F>,
    pub focus_bias: HogwildArray1<F>,
    pub context_vector: HogwildArray2<F>,
    pub context_bias: HogwildArray1<F>,
    pub grad_focus_vectors: HogwildArray2<F>,
    pub grad_focus_bias: HogwildArray1<F>,
    pub grad_context_vector: HogwildArray2<F>,
    pub grad_context_bias: HogwildArray1<F>,
}

impl<F: Float + Serialize> Model<F> {
    pub fn new(vocab_size: usize, vector_size: usize) -> Self {
        Model {
            focus_vectors: Array2::zeros((vocab_size, vector_size)).into(),
            focus_bias: Array1::zeros(vocab_size).into(),
            context_vector: Array2::zeros((vocab_size, vector_size)).into(),
            context_bias: Array1::zeros(vocab_size).into(),

            grad_focus_vectors: Array2::zeros((vocab_size, vector_size)).into(),
            grad_focus_bias: Array1::zeros(vocab_size).into(),
            grad_context_vector: Array2::zeros((vocab_size, vector_size)).into(),
            grad_context_bias: Array1::zeros(vocab_size).into(),
        }
    }

    pub fn weights_to_bytes(&self) -> Vec<u8> {
        let n_bytes = self.focus_vectors.view().len()
            + self.focus_bias.view().len()
            + self.context_vector.view().len()
            + self.context_bias.view().len();
        let mut res = Vec::<u8>::with_capacity(n_bytes * F::BYTE_SIZE);
        for x in self.focus_bias.view().iter() {
            res.extend(&x.to_bytes());
        }
        for x in self.focus_bias.view().iter() {
            res.extend(&x.to_bytes());
        }
        for x in self.context_vector.view().iter() {
            res.extend(&x.to_bytes());
        }
        for x in self.context_bias.view().iter() {
            res.extend(&x.to_bytes());
        }
        res
    }

    pub fn gradients_to_bytes(&self) -> Vec<u8> {
        let n_bytes = self.grad_focus_vectors.view().len()
            + self.grad_focus_bias.view().len()
            + self.grad_context_vector.view().len()
            + self.grad_context_bias.view().len();
        let mut res = Vec::<u8>::with_capacity(n_bytes * F::BYTE_SIZE);
        for x in self.grad_focus_bias.view().iter() {
            res.extend(&x.to_bytes());
        }
        for x in self.grad_focus_bias.view().iter() {
            res.extend(&x.to_bytes());
        }
        for x in self.grad_context_vector.view().iter() {
            res.extend(&x.to_bytes());
        }
        for x in self.grad_context_bias.view().iter() {
            res.extend(&x.to_bytes());
        }
        res
    }

    // pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
    //     if bytes.len() % F::BYTE_SIZE != 0 || (bytes.len() / F::BYTE_SIZE - 2) % 2 != 0 {
    //         None
    //     } else {
    //         let n = (bytes.len() / F::BYTE_SIZE - 2) / 2;
    //         let mut weights = Vec::<F>::with_capacity(n);
    //         let mut weights_gradsq = Vec::<F>::with_capacity(n);
    //         let mut buf: [u8; 8] = [0; 8];
    //         // Unpack weights
    //         for i in range_step(0, n * F::BYTE_SIZE, F::BYTE_SIZE) {
    //             buf[0..F::BYTE_SIZE].copy_from_slice(&bytes[i..(i + F::BYTE_SIZE)]);
    //             weights.push(F::from_bytes(buf[0..F::BYTE_SIZE].to_vec()));
    //         }

    //         // Unpack bias
    //         buf[0..F::BYTE_SIZE].copy_from_slice(&bytes[(n * F::BYTE_SIZE)..((n + 1) * F::BYTE_SIZE)]);
    //         let bias = F::from_bytes(buf[0..F::BYTE_SIZE].to_vec());

    //         // Unpack weight gradients
    //         for i in range_step((n + 1) * F::BYTE_SIZE, (2 * n + 1) * F::BYTE_SIZE, F::BYTE_SIZE) {
    //             buf[0..F::BYTE_SIZE].copy_from_slice(&bytes[i..(i + F::BYTE_SIZE)]);
    //             weights_gradsq.push(F::from_bytes(buf[0..F::BYTE_SIZE].to_vec()));
    //         }

    //         // Unpack bias gradients
    //         buf[0..F::BYTE_SIZE].copy_from_slice(&bytes[((2 * n + 1) * F::BYTE_SIZE)..((2 * n + 2) * F::BYTE_SIZE)]);
    //         let bias_gradsq = F::from_bytes(buf[0..F::BYTE_SIZE].to_vec());

    //         Some(WordVector {
    //             weights: arr1(&weights),
    //             bias,
    //             weights_gradsq: arr1(&weights_gradsq),
    //             bias_gradsq,
    //         })
    //     }
    // }
}

impl<F: Float + Serialize> PartialEq for Model<F> {
    fn eq(&self, other: &Self) -> bool {
        self.focus_vectors.view() == other.focus_vectors.view()
            && self.focus_bias.view() == other.focus_bias.view()
            && self.context_vector.view() == other.context_vector.view()
            && self.context_bias.view() == other.context_bias.view()
            && self.grad_focus_vectors.view() == other.grad_focus_vectors.view()
            && self.grad_focus_bias.view() == other.grad_focus_bias.view()
            && self.grad_context_vector.view() == other.grad_context_vector.view()
            && self.grad_context_bias.view() == other.grad_context_bias.view()
    }
}

impl<F: Float + Serialize> Eq for Model<F> {}

impl Model<f64> {
    pub fn with_random_weights(vocab_size: usize, vector_size: usize) -> Self {
        let shape = (vocab_size, vector_size);
        let n = vocab_size * vector_size;
        let bound = 0.5 / (vector_size + 1) as f64;
        let mut rng_iter = rand::thread_rng().sample_iter(Uniform::new(-bound, bound));
        let focus_vectors = match Array2::from_shape_vec(shape, rng_iter.take(n).collect()) {
            Ok(v) => v,
            Err(e) => panic!("Could not randomly initialize the focus vectors: {}", e.to_string()),
        };
        rng_iter = rand::thread_rng().sample_iter(Uniform::new(-bound, bound));
        let context_vector = match Array2::from_shape_vec(shape, rng_iter.take(n).collect()) {
            Ok(v) => v,
            Err(e) => panic!("Could not randomly initialize the context vectors: {}", e.to_string()),
        };
        rng_iter = rand::thread_rng().sample_iter(Uniform::new(-bound, bound));
        let focus_bias = match Array1::from_shape_vec(vocab_size, rng_iter.take(vocab_size).collect()) {
            Ok(v) => v,
            Err(e) => panic!("Could not randomly initialize the focus biases: {}", e.to_string()),
        };
        rng_iter = rand::thread_rng().sample_iter(Uniform::new(-bound, bound));
        let context_bias = match Array1::from_shape_vec(vocab_size, rng_iter.take(vocab_size).collect()) {
            Ok(v) => v,
            Err(e) => panic!("Could not randomly initialize the context biases: {}", e.to_string()),
        };
        Model {
            focus_vectors: focus_vectors.into(),
            focus_bias: focus_bias.into(),
            context_vector: context_vector.into(),
            context_bias: context_bias.into(),

            grad_focus_vectors: Array2::ones((vocab_size, vector_size)).into(),
            grad_focus_bias: Array1::ones(vocab_size).into(),
            grad_context_vector: Array2::ones((vocab_size, vector_size)).into(),
            grad_context_bias: Array1::ones(vocab_size).into(),
        }
    }
}

impl Model<f32> {
    pub fn with_random_weights(vocab_size: usize, vector_size: usize) -> Self {
        let shape = (vocab_size, vector_size);
        let n = vocab_size * vector_size;
        let bound = 0.5 / (vector_size + 1) as f32;
        let mut rng_iter = rand::thread_rng().sample_iter(Uniform::new(-bound, bound));
        let focus_vectors = match Array2::from_shape_vec(shape, rng_iter.take(n).collect()) {
            Ok(v) => v,
            Err(e) => panic!("Could not randomly initialize the focus vectors: {}", e.to_string()),
        };
        rng_iter = rand::thread_rng().sample_iter(Uniform::new(-bound, bound));
        let context_vector = match Array2::from_shape_vec(shape, rng_iter.take(n).collect()) {
            Ok(v) => v,
            Err(e) => panic!("Could not randomly initialize the context vectors: {}", e.to_string()),
        };
        rng_iter = rand::thread_rng().sample_iter(Uniform::new(-bound, bound));
        let focus_bias = match Array1::from_shape_vec(vocab_size, rng_iter.take(vocab_size).collect()) {
            Ok(v) => v,
            Err(e) => panic!("Could not randomly initialize the focus biases: {}", e.to_string()),
        };
        rng_iter = rand::thread_rng().sample_iter(Uniform::new(-bound, bound));
        let context_bias = match Array1::from_shape_vec(vocab_size, rng_iter.take(vocab_size).collect()) {
            Ok(v) => v,
            Err(e) => panic!("Could not randomly initialize the context biases: {}", e.to_string()),
        };
        Model {
            focus_vectors: focus_vectors.into(),
            focus_bias: focus_bias.into(),
            context_vector: context_vector.into(),
            context_bias: context_bias.into(),

            grad_focus_vectors: Array2::ones((vocab_size, vector_size)).into(),
            grad_focus_bias: Array1::ones(vocab_size).into(),
            grad_context_vector: Array2::ones((vocab_size, vector_size)).into(),
            grad_context_bias: Array1::ones(vocab_size).into(),
        }
    }
}
