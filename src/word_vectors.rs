use ndarray::{arr1, Array1};
use rand::{distributions::Uniform, Rng};
use serde::{Deserialize, Serialize as SerdeSerialize};
use std::fmt;

#[derive(Debug, Clone)]
pub struct SGDParams {
    pub alpha: f32,
    pub x_max: f32,
    pub learning_rate: f32,
    pub grad_clip_value: f32,
}

#[derive(Clone, Debug, Deserialize, SerdeSerialize)]
pub struct WordVector {
    pub weights: Array1<f32>,
    pub bias: f32,
    pub weights_gradsq: Array1<f32>,
    pub bias_gradsq: f32,
}

impl WordVector {
    pub fn new(vector_size: usize) -> Self {
        WordVector {
            weights: Array1::zeros(vector_size),
            bias: 0f32,
            weights_gradsq: Array1::ones(vector_size),
            bias_gradsq: 1f32,
        }
    }
}

impl fmt::Display for WordVector {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "<V = {} + {}, ∇²V = {} + {}>",
            self.weights.to_string(),
            self.bias,
            self.weights_gradsq.to_string(),
            self.bias_gradsq
        )
    }
}

impl PartialEq for WordVector {
    fn eq(&self, other: &Self) -> bool {
        self.weights == other.weights
            && self.bias == other.bias
            && self.weights_gradsq == other.weights_gradsq
            && self.bias_gradsq == other.bias_gradsq
    }
}

impl Eq for WordVector {}

impl WordVector {
    pub fn with_random_weights(vector_size: usize) -> Self {
        let bound = 0.5 / (vector_size + 1) as f32;
        let between = Uniform::new(-bound, bound);
        let rng = rand::thread_rng();
        let weights: Vec<f32> = rng.sample_iter(between).take(vector_size).collect();
        WordVector {
            weights: arr1(&weights),
            bias: rng.sample_iter(between).next().unwrap(),
            weights_gradsq: Array1::ones(vector_size),
            bias_gradsq: 1f32,
        }
    }
}

pub fn sgd_step(
    focus: &mut WordVector,
    context: &mut WordVector,
    target_value: f32,
    sgd_params: SGDParams,
) -> Option<f32> {
    let diff = focus.weights.dot(&context.weights) + focus.bias + context.bias - target_value.ln();
    let fdiff = if target_value > sgd_params.x_max {
        diff
    } else {
        (target_value / sgd_params.x_max).powf(sgd_params.alpha) * diff
    };
    let cost = 0.5 * fdiff * diff; // weighted squared error for this step

    if !diff.is_finite() || !fdiff.is_finite() {
        return None;
    }

    let focus_update = context.weights.map(|v| {
        (fdiff * v)
            .max(-sgd_params.grad_clip_value)
            .min(sgd_params.grad_clip_value)
            * sgd_params.learning_rate
    });
    let context_update = focus.weights.map(|v| {
        (fdiff * v)
            .max(-sgd_params.grad_clip_value)
            .min(sgd_params.grad_clip_value)
            * sgd_params.learning_rate
    });

    let focus_adapt_update = &focus_update / &focus.weights_gradsq.mapv(f32::sqrt);
    let context_adapt_update = &context_update / &context.weights_gradsq.mapv(f32::sqrt);

    focus.weights_gradsq += &(&focus_update * &focus_update);
    context.weights_gradsq += &(&context_update * &context_update);

    if focus_adapt_update.sum().is_finite() && context_adapt_update.sum().is_finite() {
        focus.weights -= &focus_adapt_update;
        context.weights -= &context_adapt_update;
    }

    if (fdiff / focus.bias_gradsq.sqrt()).is_finite() {
        focus.bias -= fdiff / focus.bias_gradsq.sqrt();
    }
    if (fdiff / context.bias_gradsq.sqrt()).is_finite() {
        context.bias -= fdiff / context.bias_gradsq.sqrt();
    }
    focus.bias_gradsq += fdiff * fdiff;
    context.bias_gradsq += fdiff * fdiff;

    Some(cost)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_scaled_add() {
        let mut v = Array::zeros(2);
        v.scaled_add(1f32, &Array::ones(2));
        assert_eq!(v, Array::ones(2));
    }

    #[test]
    fn test_serialization() {
        let a = WordVector {
            weights: Array::zeros(2),
            bias: 0f32,
            weights_gradsq: Array::zeros(2),
            bias_gradsq: 0f32,
        };
        let mut buf: [u8; 58] = [0; 58];
        if let Ok(b) = bincode::serialize(&a) {
            buf.copy_from_slice(&b);
            let c: WordVector = bincode::deserialize(&buf).unwrap();
            assert_eq!(a, c);
        } else {
            unreachable!();
        }
    }

    #[test]
    fn test_sgd_with_zero_loss() {
        let sgd_params = SGDParams {
            alpha: 1f32,
            x_max: 100f32,
            learning_rate: 1f32,
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
            x_max: 1f32,
            learning_rate: 1f32,
            grad_clip_value: 100f32,
        };
        let mut focus = WordVector {
            weights: Array::ones(2),
            bias: 1f32,
            weights_gradsq: Array::ones(2),
            bias_gradsq: 0f32,
        }; //<V = [1, 1] + 1, ∇V = [1, 1] + 0>
        let mut context = WordVector {
            weights: Array::from_elem(2, 0.7),
            bias: 1f32,
            weights_gradsq: Array::ones(2),
            bias_gradsq: 0f32,
        }; // <V = [0.7, 0.7] + 1, ∇V = [1, 1] + 0>
        let n = sgd_step(&mut focus, &mut context, 10.0, sgd_params);

        if let Some(x) = n {
            // By an independent calculation, for these 2 vectors, we have:
            // diff=1.097, fdiff=1.097, cost=0.6022
            // w_upd_focus = -0.768, w_upd_context = -1.097
            // grad_upd_focus = 0.590, grad_upd_context = 1.204
            assert!((x - 0.5 * 1.097 * 1.097).abs() < 1e-3); // diff = fdiff = 1.097 here
            assert!(focus.weights.all_close(&Array::from_elem(2, 1.0 - 0.768), 1e-3));
            assert!(focus.weights_gradsq.all_close(&Array::from_elem(2, 1.590), 1e-3));
            assert!(context.weights.all_close(&Array::from_elem(2, 0.7 - 1.097), 1e-3));
            assert!(context
                .weights_gradsq
                .all_close(&Array::from_elem(2, 2.204), 1e-3));
        } else {
            assert!(n.is_some());
        }
    }
}
