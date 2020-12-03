use hogwild::{HogwildArray1, HogwildArray2};
use ndarray::{Array1, Array2};
use rand::{distributions::Uniform, Rng};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::fs;
use std::io::{BufReader, BufWriter, Write};

#[derive(Debug)]
pub enum ReadWriteError {
    IoError(std::io::Error),
    EncodeError,
    DecodeError,
}

#[derive(Clone, Deserialize, Serialize)]
struct ModelWeights {
    focus_vectors: Array2<f32>,
    focus_bias: Array1<f32>,
    context_vector: Array2<f32>,
    context_bias: Array1<f32>,
}

impl ModelWeights {
    pub fn to_file(&self, path: &str) -> Result<(), ReadWriteError> {
        let bytes = match bincode::serialize(&self) {
            Ok(b) => b,
            Err(_e) => return Err(ReadWriteError::EncodeError),
        };
        let mut fout = BufWriter::new(match fs::File::create(path) {
            Ok(fout) => fout,
            Err(e) => return Err(ReadWriteError::IoError(e)),
        });
        match fout.write_all(bytes.as_slice()) {
            Ok(x) => Ok(x),
            Err(e) => Err(ReadWriteError::IoError(e)),
        }
    }

    pub fn from_file(path: &str) -> Result<Self, ReadWriteError> {
        let fin = BufReader::new(match fs::File::open(path) {
            Ok(fin) => fin,
            Err(e) => return Err(ReadWriteError::IoError(e)),
        });
        match bincode::deserialize_from(fin) {
            Ok(x) => Ok(x),
            Err(_e) => Err(ReadWriteError::DecodeError),
        }
    }
}

// Unsafe arrays for Hogwild method of parallel Stochastic Gradient descent
#[derive(Clone)]
pub struct Model {
    pub focus_vectors: HogwildArray2<f32>,
    pub focus_bias: HogwildArray1<f32>,
    pub context_vector: HogwildArray2<f32>,
    pub context_bias: HogwildArray1<f32>,
    pub grad_focus_vectors: HogwildArray2<f32>,
    pub grad_focus_bias: HogwildArray1<f32>,
    pub grad_context_vector: HogwildArray2<f32>,
    pub grad_context_bias: HogwildArray1<f32>,
}

impl Model {
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

    pub fn from_file(weights_path: &str, gradients_path: Option<&str>) -> Result<Self, ReadWriteError> {
        let weights = match ModelWeights::from_file(weights_path) {
            Ok(w) => w,
            Err(e) => return Err(e),
        };
        let grads = match gradients_path {
            Some(path) => match ModelWeights::from_file(path) {
                Ok(w) => w,
                Err(e) => return Err(e),
            },
            None => ModelWeights {
                focus_vectors: Array2::ones(weights.focus_vectors.dim()),
                focus_bias: Array1::ones(weights.focus_bias.dim()),
                context_vector: Array2::ones(weights.context_vector.dim()),
                context_bias: Array1::ones(weights.context_bias.dim()),
            },
        };
        Ok(Model {
            focus_vectors: weights.focus_vectors.into(),
            focus_bias: weights.focus_bias.into(),
            context_vector: weights.context_vector.into(),
            context_bias: weights.context_bias.into(),

            grad_focus_vectors: grads.focus_vectors.into(),
            grad_focus_bias: grads.focus_bias.into(),
            grad_context_vector: grads.context_vector.into(),
            grad_context_bias: grads.context_bias.into(),
        })
    }

    pub fn weights_to_file(&self, path: &str) -> Result<(), ReadWriteError> {
        let weights = ModelWeights {
            focus_vectors: self.focus_vectors.view().into_owned(),
            focus_bias: self.focus_bias.view().into_owned(),
            context_vector: self.context_vector.view().into_owned(),
            context_bias: self.context_bias.view().into_owned(),
        };
        weights.to_file(path)
    }

    pub fn gradients_to_file(&self, path: &str) -> Result<(), ReadWriteError> {
        let weights = ModelWeights {
            focus_vectors: self.grad_focus_vectors.view().into_owned(),
            focus_bias: self.grad_focus_bias.view().into_owned(),
            context_vector: self.grad_context_vector.view().into_owned(),
            context_bias: self.grad_context_bias.view().into_owned(),
        };
        weights.to_file(path)
    }
}

impl PartialEq for Model {
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

impl Eq for Model {}
