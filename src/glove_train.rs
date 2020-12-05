extern crate rand;

use log::info;
use ndarray::Axis;
use std::fs;
use std::io::{BufReader, Seek, SeekFrom};

use super::word_vectors::{sgd_step, SGDParams, WordVector};
use super::{glove_model::Model, record_io::Records};

#[derive(Clone)]
pub struct TrainParams {
    pub n_lines: usize,
    pub n_epochs: usize,
    pub n_threads: usize,
    pub input_file: String,
}

pub fn train_epoch(model: &Model, train: TrainParams, sgd: SGDParams) -> f32 {
    let threads: Vec<_> = (0..train.n_threads)
        .map(|j| {
            let start = train.n_lines / train.n_threads * j;
            let end = if j != train.n_threads - 1 {
                train.n_lines / train.n_threads * (j + 1)
            } else {
                train.n_lines
            };
            let model = model.clone();
            let path = train.input_file.clone();
            let params = sgd.clone();
            std::thread::spawn(move || glove_thread(model, params, path, start, end))
        })
        .collect();
    let total_cost = threads
        .into_iter()
        .map(|t| match t.join() {
            Ok(c) => c,
            Err(e) => panic!("{:?}", e),
        })
        .fold(0.0, |a, b| a + b);
    total_cost
}

pub fn glove_thread(
    mut model: Model,
    sgd_params: SGDParams,
    input_file: String,
    start: usize,
    end: usize,
) -> f32 {
    let mut cost = 0f32;
    let mut example_counter: usize = start;
    let reader = match fs::File::open(input_file.clone()) {
        Ok(mut file_in) => match file_in.seek(SeekFrom::Start((start * 20) as u64)) {
            Ok(_) => Records {
                buffer: [0; 20],
                filename: input_file,
                reader: BufReader::new(file_in),
            },
            Err(e) => panic!("Could not seek to position {}: {}", start * 20, e.to_string()),
        },
        Err(e) => panic!("Could not open the file: {}", e.to_string()),
    };

    for next in reader {
        let record = match next {
            Ok(n) => {
                example_counter += 1;
                if example_counter >= end {
                    break;
                }
                n
            }
            Err(e) => panic!(
                "Could not parse the record at counter={}: {}",
                example_counter,
                e.to_string()
            ),
        };
        let mut focus = WordVector {
            weights: model
                .focus_vectors
                .subview_mut(Axis(0), record.row as usize)
                .into_owned(),
            bias: model.focus_bias.view_mut()[record.row as usize],
            weights_gradsq: model
                .grad_focus_vectors
                .subview_mut(Axis(0), record.row as usize)
                .into_owned(),
            bias_gradsq: model.grad_focus_bias.view_mut()[record.row as usize],
        };
        let mut context = WordVector {
            weights: model
                .context_vector
                .subview_mut(Axis(0), record.col as usize)
                .into_owned(),
            bias: model.context_bias.view_mut()[record.col as usize],
            weights_gradsq: model
                .grad_context_vector
                .subview_mut(Axis(0), record.col as usize)
                .into_owned(),
            bias_gradsq: model.grad_context_bias.view_mut()[record.col as usize],
        };
        if let Some(c) = sgd_step(&mut focus, &mut context, record.val, sgd_params.clone()) {
            cost += c;
        } else {
            info!("Caught NaN in diff or fdiff for thread. Skipping update");
        }
    }
    model.weights_to_file("weights_1.mpk").unwrap();
    model.gradients_to_file("grads_1.mpk").unwrap();
    cost
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};

    use super::super::sparse::SparseRecord;
    use super::*;
    use std::io::{BufWriter, Write};

    #[test]
    fn test_glove_thread() {
        let n = 5usize;
        let params = SGDParams {
            alpha: 1.0,
            learning_rate: 1.0,
            grad_clip_value: 100.0,
            x_max: 1.0,
        };
        let filename = "/tmp/tmp_file.test";
        let mut writer = BufWriter::new(fs::File::create(filename.to_string()).unwrap());
        for row in 0..n {
            for col in 0..n {
                let rec = SparseRecord {
                    row,
                    col,
                    val: 1f32 / (row * col + 1) as f32,
                };
                writer.write_all(&bincode::serialize(&rec).unwrap()).unwrap();
            }
        }
        writer.flush().unwrap();

        let model = Model {
            focus_vectors: Array2::from_elem((n, 2), 0.3).into(),
            focus_bias: Array1::from_elem(n, 1.3).into(),
            context_vector: Array2::from_elem((n, 2), 0.5).into(),
            context_bias: Array1::from_elem(n, 1.5).into(),

            grad_focus_vectors: Array2::ones((n, 2)).into(),
            grad_focus_bias: Array1::ones(n).into(),
            grad_context_vector: Array2::ones((n, 2)).into(),
            grad_context_bias: Array1::ones(n).into(),
        };

        let cost = glove_thread(model, params, filename.to_string(), 0, n * n);
        assert!(cost - 75.087 < 1e-3); // We expect 75.08659
    }

    #[test]
    fn test_train_epoch() {
        let n = 5usize;
        let params = SGDParams {
            alpha: 1.0,
            learning_rate: 1.0,
            grad_clip_value: 100.0,
            x_max: 1.0,
        };
        let tparams = TrainParams {
            n_lines: n * n,
            n_epochs: 0,
            n_threads: 3,
            input_file: "/tmp/tmp_file.test".to_string(),
        };
        let mut writer = BufWriter::new(fs::File::create(tparams.input_file.clone()).unwrap());
        for row in 0..n {
            for col in 0..n {
                let rec = SparseRecord {
                    row,
                    col,
                    val: 1f32 / (row * col + 1) as f32,
                };
                writer.write_all(&bincode::serialize(&rec).unwrap()).unwrap();
            }
        }
        writer.flush().unwrap();

        let model = Model {
            focus_vectors: Array2::from_elem((n, 2), 0.3).into(),
            focus_bias: Array1::from_elem(n, 1.3).into(),
            context_vector: Array2::from_elem((n, 2), 0.5).into(),
            context_bias: Array1::from_elem(n, 1.5).into(),

            grad_focus_vectors: Array2::ones((n, 2)).into(),
            grad_focus_bias: Array1::ones(n).into(),
            grad_context_vector: Array2::ones((n, 2)).into(),
            grad_context_bias: Array1::ones(n).into(),
        };

        let cost = train_epoch(&model, tparams.clone(), params);
        assert!(cost - 75.087 < 1e-3); // We expect 75.08659
    }
}
