extern crate num;
extern crate rand;

use super::sparse::Serialize;
use ndarray::{arr1, Array1};
use num::{range_step, Float};
use rand::{distributions::Uniform, Rng};

pub struct SGDParams {
    alpha: f32,
    x_max: f32,
    eta: f32,
    grad_clip_value: f32,
}

#[derive(Debug, Clone)]
pub struct WordVector<F: Float + Serialize> {
    weights: Array1<F>,
    bias: F,
    weights_gradsq: Array1<F>,
    bias_gradsq: F,
}

impl<F: Float + Serialize> WordVector<F> {
    pub fn new(vector_size: usize) -> Self {
        WordVector {
            weights: Array1::zeros(vector_size),
            bias: F::zero(),
            weights_gradsq: Array1::ones(vector_size),
            bias_gradsq: F::one(),
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut res = Vec::<u8>::with_capacity((self.weights.len() + self.weights_gradsq.len() + 2) * F::BYTE_SIZE);
        for x in (&self.weights).iter() {
            res.extend(&x.to_bytes());
        }
        res.extend(&self.bias.to_bytes());
        for x in (&self.weights_gradsq).iter() {
            res.extend(&x.to_bytes());
        }
        res.extend(&self.bias_gradsq.to_bytes());
        res
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() % F::BYTE_SIZE != 0 || (bytes.len() / F::BYTE_SIZE - 2) % 2 != 0 {
            None
        } else {
            let n = (bytes.len() / F::BYTE_SIZE - 2) / 2;
            let mut weights = Vec::<F>::with_capacity(n);
            let mut weights_gradsq = Vec::<F>::with_capacity(n);
            let mut buf: [u8; 8] = [0; 8];
            // Unpack weights
            for i in range_step(0, n * F::BYTE_SIZE, F::BYTE_SIZE) {
                buf[0..F::BYTE_SIZE].copy_from_slice(&bytes[i..(i + F::BYTE_SIZE)]);
                weights.push(F::from_bytes(buf[0..F::BYTE_SIZE].to_vec()));
            }

            // Unpack bias
            buf[0..F::BYTE_SIZE].copy_from_slice(&bytes[(n * F::BYTE_SIZE)..((n + 1) * F::BYTE_SIZE)]);
            let bias = F::from_bytes(buf[0..F::BYTE_SIZE].to_vec());

            // Unpack weight gradients
            for i in range_step((n + 1) * F::BYTE_SIZE, (2 * n + 1) * F::BYTE_SIZE, F::BYTE_SIZE) {
                buf[0..F::BYTE_SIZE].copy_from_slice(&bytes[i..(i + F::BYTE_SIZE)]);
                weights_gradsq.push(F::from_bytes(buf[0..F::BYTE_SIZE].to_vec()));
            }

            // Unpack bias gradients
            buf[0..F::BYTE_SIZE].copy_from_slice(&bytes[((2 * n + 1) * F::BYTE_SIZE)..((2 * n + 2) * F::BYTE_SIZE)]);
            let bias_gradsq = F::from_bytes(buf[0..F::BYTE_SIZE].to_vec());

            Some(WordVector {
                weights: arr1(&weights),
                bias,
                weights_gradsq: arr1(&weights_gradsq),
                bias_gradsq,
            })
        }
    }
}

impl<F: Float + Serialize> PartialEq for WordVector<F> {
    fn eq(&self, other: &Self) -> bool {
        self.weights == other.weights
            && self.bias == other.bias
            && self.weights_gradsq == other.weights_gradsq
            && self.bias_gradsq == other.bias_gradsq
    }
}

impl<F: Float + Serialize> Eq for WordVector<F> {}

impl WordVector<f64> {
    pub fn with_random_weights(vector_size: usize) -> Self {
        let bound = 0.5 / (vector_size + 1) as f64;
        let between = Uniform::new(-bound, bound);
        let rng = rand::thread_rng();
        let weights: Vec<f64> = rng.sample_iter(between).take(vector_size).collect();
        WordVector {
            weights: arr1(&weights),
            bias: 0f64,
            weights_gradsq: Array1::ones(vector_size),
            bias_gradsq: 1f64,
        }
    }
}

impl WordVector<f32> {
    pub fn with_random_weights(vector_size: usize) -> Self {
        let bound = 0.5 / (vector_size + 1) as f32;
        let between = Uniform::new(-bound, bound);
        let rng = rand::thread_rng();
        let weights: Vec<f32> = rng.sample_iter(between).take(vector_size).collect();
        WordVector {
            weights: arr1(&weights),
            bias: 0f32,
            weights_gradsq: Array1::ones(vector_size),
            bias_gradsq: 1f32,
        }
    }
}

pub fn sgd_step(
    focus: &mut WordVector<f32>,
    context: &mut WordVector<f32>,
    target_value: f32,
    sgd_params: SGDParams,
) -> Option<f32> {
    let diff = focus.weights.dot(&context.weights) + focus.bias + context.bias - target_value.ln();
    let mut fdiff: f32 = if target_value > sgd_params.x_max {
        diff
    } else {
        (target_value / sgd_params.x_max).powf(sgd_params.alpha) * diff
    };
    if !diff.is_finite() || !fdiff.is_finite() {
        return None;
    }

    let mut focus_update = context.weights.map(|v| {
        (fdiff * v)
            .max(-sgd_params.grad_clip_value)
            .min(sgd_params.grad_clip_value)
    });
    let mut context_update = focus.weights.map(|v| {
        (fdiff * v)
            .max(-sgd_params.grad_clip_value)
            .min(sgd_params.grad_clip_value)
    });

    focus
        .weights_gradsq
        .scaled_add(sgd_params.eta, &(focus_update.map(|v| v * v)));
    context
        .weights_gradsq
        .scaled_add(sgd_params.eta, &(context_update.map(|v| v * v)));

    focus_update = focus_update / focus.weights_gradsq.mapv(f32::sqrt);
    context_update = context_update / context.weights_gradsq.mapv(f32::sqrt);

    if focus_update.sum().is_finite() && context_update.sum().is_finite() {
        focus.weights.scaled_add(-sgd_params.eta, &focus_update);
        context.weights.scaled_add(-sgd_params.eta, &context_update);
    }

    if (fdiff / focus.bias_gradsq.sqrt()).is_finite() {
        focus.bias -= fdiff / focus.bias_gradsq.sqrt();
    }
    if (fdiff / context.bias_gradsq.sqrt()).is_finite() {
        context.bias -= fdiff / context.bias_gradsq.sqrt();
    }
    fdiff *= fdiff;
    focus.bias_gradsq += fdiff;
    context.bias_gradsq += fdiff;

    Some(0.5 * fdiff * diff) // weighted squared error for this step
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_to_from_bytes_f32() {
        let a = WordVector::<f32> {
            weights: Array::zeros(2),
            bias: 0f32,
            weights_gradsq: Array::zeros(2),
            bias_gradsq: 0f32,
        };
        let mut buf: [u8; 24] = [0; 24];
        buf.copy_from_slice(&a.to_bytes());
        assert_eq!(a, WordVector::<f32>::from_bytes(&buf).unwrap());
    }

    #[test]
    fn test_to_from_bytes_f64() {
        let a = WordVector::<f64> {
            weights: Array1::<f64>::zeros(2),
            bias: 0f64,
            weights_gradsq: Array1::<f64>::zeros(2),
            bias_gradsq: 0f64,
        };
        let mut buf: [u8; 48] = [0; 48];
        buf.copy_from_slice(&a.to_bytes());
        print!("{:?}", &buf[0..24]);
        let b = WordVector::<f64>::from_bytes(&buf);
        print!("{:?}", b);
        assert_eq!(a, WordVector::<f64>::from_bytes(&buf).unwrap());
    }

    #[test]
    fn test_sgd_with_zero_loss() {
        let sgd_params = SGDParams {
            alpha: 1f32,
            x_max: 100f32,
            eta: 1f32,
            grad_clip_value: 100f32,
        };
        let mut focus = WordVector {
            weights: Array::zeros(2),
            bias: 0f32,
            weights_gradsq: Array::zeros(2),
            bias_gradsq: 0f32,
        };
        let mut context = WordVector {
            weights: Array::zeros(2),
            bias: 0f32,
            weights_gradsq: Array::zeros(2),
            bias_gradsq: 0f32,
        };
        let n = sgd_step(&mut focus, &mut context, 1.0, sgd_params);

        if let Some(x) = n {
            assert!((x - 0.0).abs() < 1e-3);
        } else {
            assert!(n.is_some());
        }
    }

    #[test]
    fn test_sgd_with_non_zero_loss() {
        let sgd_params = SGDParams {
            alpha: 1f32,
            x_max: 100f32,
            eta: 1f32,
            grad_clip_value: 100f32,
        };
        let mut focus = WordVector {
            weights: Array::ones(2),
            bias: 0f32,
            weights_gradsq: Array::ones(2),
            bias_gradsq: 0f32,
        };
        let mut context = WordVector {
            weights: Array::zeros(2),
            bias: 0f32,
            weights_gradsq: Array::zeros(2),
            bias_gradsq: 0f32,
        };
        let n = sgd_step(&mut focus, &mut context, 15.0, sgd_params);

        if let Some(x) = n {
            assert!((x + 0.223).abs() < 1e-3);
        } else {
            assert!(n.is_some());
        }
    }
}
