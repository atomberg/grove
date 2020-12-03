#[macro_use]
extern crate log;
extern crate rand;

use env_logger;
use getopts::Options;
use ndarray::Axis;
use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Seek, SeekFrom};

use grove::word_vectors::{sgd_step, SGDParams, WordVector};
use grove::{glove_model::Model, record_io::Records};

struct Params {
    vector_size: usize,
    n_lines: Option<usize>,
    n_iter: usize,
    n_threads: usize,
    sgd: SGDParams,
    binary: SaveVectorsFormat,
    checkpoint_every: Option<usize>,
    vocab_file: String,
    input_file: String,
    save_file: String,
    gradsq_file: String,
}

#[allow(clippy::enum_variant_names)]
enum WordVectorOutput {
    BothVectorsSeparatelyWithBiasTerms,
    BothVectorsSeparatelyNoBiasTerms,
    AddlVectorsExcludeBiasTerms,
}

enum SaveVectorsFormat {
    Binary,
    Text(WordVectorOutput),
    Both(WordVectorOutput),
}

fn main() {
    env_logger::init();
    info!("Starting");
    let mut params = parse_args();
    let vocab_size = BufReader::new(match fs::File::open(params.vocab_file.clone()) {
        Ok(fp) => fp,
        Err(e) => panic!("Could not open the vocabulary file: {}", e.to_string()),
    })
    .lines()
    .count();
    params.n_lines = match fs::metadata(params.input_file.clone()) {
        Ok(meta) => Some(meta.len() as usize / 20),
        Err(e) => panic!("Could not read the metadata of the input file: {}", e.to_string()),
    };
    if let Some(n_lines) = params.n_lines {
        info!("Read {} lines.", n_lines);
    }

    let model = Model::with_random_weights(vocab_size, params.vector_size);
    model.weights_to_file("weights_0.mpk").unwrap();
    model.gradients_to_file("grads_0.mpk").unwrap();
    let model = match Model::from_file("weights_0.mpk", Some("grads_0.mpk")) {
        Ok(m) => m,
        Err(e) => panic!("{:?}", e),
    };
    // train_step(&model, &params);
    glove_thread(model, params.sgd.clone(), params.input_file, 0, 1000);
}

fn train_step(model: &Model, params: &Params) {
    info!("Starting to train GloVe");

    let n_lines = match params.n_lines {
        Some(s) => s,
        None => unreachable!(),
    };

    // let output_file: String;
    // if nb_iter <= 0 {
    //     output_file = format!("{}.bin", params.save_file);
    // } else {
    //     output_file = format!("{}.{:>03}.bin", params.save_file, nb_iter);
    // }
    info!("Initialized the weights");
    for i in 0..params.n_iter {
        let threads: Vec<_> = (0..params.n_threads)
            .map(|j| {
                let start = n_lines / params.n_threads * j;
                let end = if j != params.n_threads - 1 {
                    n_lines / params.n_threads * (j + 1)
                } else {
                    n_lines
                };
                let model = model.clone();
                let path = params.input_file.clone();
                let params = params.sgd.clone();
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
        info!("iter: {:>03}, cost: {}", i + 1, total_cost / n_lines as f32);
    }
}

fn glove_thread(mut model: Model, sgd_params: SGDParams, input_file: String, start: usize, end: usize) -> f32 {
    let mut cost = 0f32;
    let mut example_counter: usize = start;
    let reader = match fs::File::open(input_file.clone()) {
        Ok(mut file_in) => match file_in.seek(SeekFrom::Start((start * 20) as u64)) {
            Ok(_) => Records {
                buffer: [0; 20],
                filename: input_file,
                reader: BufReader::new(file_in),
            },
            Err(e) => panic!("Could seek to position {}: {}", start * 20, e.to_string()),
        },
        Err(e) => panic!("Could open the file: {}", e.to_string()),
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

// fn train_loop(params: Params) -> i32 {
//     info!("TRAINING MODEL");
//     info!("Initializing parameters...");
//     let mut w: Vec<f64> = vec![];
//     let mut gradsq: Vec<f64> = vec![];
//     initialize_parameters(&mut w, &mut gradsq, params.vector_size, params.vocab_size);
//     info!("done.");
//     info!("vector size: {}", params.vector_size);
//     info!("vocab size: {}", params.vocab_size);
//     info!("x_max: {}", params.sgd.x_max);
//     info!("alpha: {}", params.sgd.alpha);
//     let input_file = params.input_file.to_string();
//     for i in 0..params.n_iter {
//         {
//             let w_slice = UnsafeSlice::new(&mut w);
//             let gradsq_slice = UnsafeSlice::new(&mut gradsq);
//             crossbeam::scope(|scope| {
//                 let threads: Vec<_> = (0..params.n_threads)
//                     .map(|j| {
//                         let start = params.n_lines / params.n_threads * j;
//                         let end = if j != params.n_threads - 1 {
//                             params.n_lines / params.n_threads * (j + 1)
//                         } else {
//                             params.n_lines
//                         };
//                         let input_file = &input_file;
//                         scope.spawn(move || {
//                             glove_thread(
//                                 w_slice,
//                                 gradsq_slice,
//                                 params.sgd,
//                                 input_file.to_string(),
//                                 params.vocab_size,
//                                 start,
//                                 end,
//                             )
//                         })
//                     })
//                     .collect();
//                 let total_cost = threads.into_iter().map(|e| e.join()).fold(0f64, |a, b| a + b);
//                 info!(
//                     "{}, iter: {:>03}, cost: {}",
//                     time::strftime("%x - %I:%M.%S%p", &time::now()).unwrap(),
//                     i + 1,
//                     total_cost / params.n_lines as f64
//                 );
//             });
//         }
//         if params.checkpoint_every > 0 && (i + 1) % params.checkpoint_every == 0 {
//             info!("    saving intermediate parameters for iter {:>03}...", i + 1);
//             if params.binary > 0 {
//                 save_params_bin(&w, params.save_file, i + 1);
//                 if params.save_gradsq {
//                     save_gsq_bin(&gradsq, params.gradsq_file, i + 1);
//                 }
//             }
//             if params.binary != 1 {
//                 save_params_txt(
//                     &w,
//                     params.save_file,
//                     params.vocab_file,
//                     params.vector_size,
//                     params.vocab_size,
//                     i + 1,
//                     params.model,
//                 );
//                 if params.save_gradsq {
//                     save_gsq_txt(
//                         &gradsq,
//                         params.gradsq_file,
//                         params.vocab_file,
//                         params.vector_size,
//                         params.vocab_size,
//                         i + 1,
//                     );
//                 }
//             }
//         }
//     }
//     let mut retval = 0i32;
//     if params.binary > 0 {
//         retval |= save_params_bin(&w, params.save_file, 0);
//         if params.save_gradsq {
//             retval |= save_gsq_bin(&gradsq, params.gradsq_file, 0);
//         }
//     }
//     if params.binary != 1 {
//         retval |= save_params_txt(
//             &w,
//             params.save_file,
//             params.vocab_file,
//             params.vector_size,
//             params.vocab_size,
//             0,
//             params.model,
//         );
//         if params.save_gradsq {
//             retval |= save_gsq_txt(
//                 &gradsq,
//                 params.gradsq_file,
//                 params.vocab_file,
//                 params.vector_size,
//                 params.vocab_size,
//                 00000000,
//             );
//         }
//     }
//     retval
// }

fn parse_args() -> Params {
    let args: Vec<String> = env::args().collect();
    let mut opts = Options::new();
    let mut params = Params {
        sgd: SGDParams {
            alpha: 0.75,
            x_max: 100.0,
            eta: 0.05,
            grad_clip_value: 100.0,
        },
        n_iter: 25,
        n_threads: 8,
        n_lines: None,
        vector_size: 50,
        binary: SaveVectorsFormat::Binary,
        checkpoint_every: None,
        vocab_file: "vocab.txt".to_string(),
        input_file: "cooccurrence.shuf.bin".to_string(),
        save_file: "vectors".to_string(),
        gradsq_file: "gradsq".to_string(),
    };

    opts.optopt(
        "",
        "vector-size",
        "Dimension of word vector representations (excluding bias term); default 50",
        "<int>",
    );
    opts.optopt("", "threads", "Number of threads; default 8", "<int>");
    opts.optopt("", "iter", "Number of training iterations; default 25", "<int>");
    opts.optopt("", "eta", "Initial leaning rate; default 0.05", "<float>");
    opts.optopt(
        "",
        "alpha",
        "Parameter in exponent of weighting function; default 0.75",
        "<float>",
    );
    opts.optopt(
        "",
        "x-max",
        "Parameter specifying cutoff in weighting function; default 100.0",
        "<float>",
    );
    opts.optopt(
        "",
        "binary",
        "Save output in binary format (0: text, 1: binary, 2: both); default 0",
        "<int>",
    );
    opts.optopt(
        "",
        "input-file",
        "Binary input file of shuffled cooccurrence data (produced by 'cooccur' and 'shuffle'); default cooccurrence.shuf.bin",
        "<file>",
    );
    opts.optopt(
        "",
        "vocab-file",
        "File containing vocabulary (truncated unigram counts, produced by 'vocab_count'); default vocab.txt",
        "<file>",
    );
    opts.optopt(
        "",
        "save-file",
        "Filename, excluding extension, for word vector output; default vectors",
        "<file>",
    );
    opts.optopt(
        "",
        "gradsq-file",
        "Filename, excluding extension, for squared gradient output; defaut gradsq",
        "<file>",
    );
    opts.optopt(
        "",
        "save-gradsq",
        "Save accumulated squared gradients; default 0 (off); ignored if gradsq-file is specified",
        "<int>",
    );
    opts.optopt(
        "",
        "checkpoint-every",
        "Checkpoint a model every <int> iterations; default 0 (off)",
        "<int>",
    );
    opts.optopt(
        "",
        "gradsq-file",
        "Filename, excluding extension, for squared gradient output; defaut gradsq",
        "<file>",
    );
    opts.optopt(
        "",
        "model",
        "Model for word vector output (for text output only); default 2
        0: output all data, for both word and context word vectors, including bias terms
        1: output word vectors, excluding bias terms
        2: output word vectors + context word vectors, excluding bias terms",
        "<int>",
    );

    opts.optflag("h", "help", "print this help menu");
    let matches = match opts.parse(&args[1..]) {
        Ok(m) => m,
        Err(f) => panic!(f.to_string()),
    };
    if matches.opt_present("help") {
        print!("{}", opts.usage("Usage: ./glove [options]"));
        std::process::exit(0);
    } else {
        params.vector_size = match matches.opt_get_default("vector-size", params.vector_size) {
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
        params.n_threads = match matches.opt_get_default("threads", params.n_threads) {
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
        params.n_iter = match matches.opt_get_default("iter", params.n_iter) {
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
        params.sgd.eta = match matches.opt_get_default("eta", params.sgd.eta) {
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
        params.sgd.alpha = match matches.opt_get_default("alpha", params.sgd.alpha) {
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
        params.sgd.x_max = match matches.opt_get_default("x-max", params.sgd.x_max) {
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
        params.binary = match matches.opt_get_default("binary", 0i32) {
            Ok(b) => {
                let word_vector_output = match matches.opt_get_default("model", 2i32) {
                    Ok(m) => match m {
                        0 => WordVectorOutput::BothVectorsSeparatelyWithBiasTerms,
                        1 => WordVectorOutput::BothVectorsSeparatelyNoBiasTerms,
                        2 => WordVectorOutput::AddlVectorsExcludeBiasTerms,
                        _ => panic!("Only available values for 'model' are 0, 1, or 2"),
                    },
                    Err(f) => panic!("Could not parse 'model' value to i32: {}", f.to_string()),
                };
                match b {
                    0 => SaveVectorsFormat::Binary,
                    1 => SaveVectorsFormat::Text(word_vector_output),
                    2 => SaveVectorsFormat::Both(word_vector_output),
                    _ => panic!("Only available values for 'binary' are 0, 1, or 2"),
                }
            }
            Err(f) => panic!("Could not parse 'model' value to i32: {}", f.to_string()),
        };
        params.input_file = match matches.opt_get_default("input-file", params.input_file) {
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
        params.vocab_file = match matches.opt_get_default("vocab-file", params.vocab_file) {
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
        params.save_file = match matches.opt_get_default("save-file", params.save_file) {
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
        params.gradsq_file = match matches.opt_get_default("gradsq-file", params.gradsq_file) {
            Ok(m) => m,
            Err(f) => panic!(f.to_string()),
        };
        params.checkpoint_every = match matches.opt_get_default("checkpoint-every", 0usize) {
            Ok(m) => match m {
                0 => None,
                _ => Some(m),
            },
            Err(f) => panic!(f.to_string()),
        };
    }
    params
}

#[cfg(test)]
mod tests {
    use super::*;
    use grove::sparse::SparseRecord;
    use ndarray::Array;
    use std::io::{BufWriter, Write};

    #[test]
    fn test_sgd_with_non_zero_loss() {
        let sgd_params = SGDParams {
            alpha: 1f32,
            x_max: 20f32,
            eta: 1f32,
            grad_clip_value: 1000f32,
        };
        let mut focus = WordVector {
            weights: Array::ones(2),
            bias: 1f32,
            weights_gradsq: Array::ones(2),
            bias_gradsq: 1f32,
        };
        let mut context = WordVector {
            weights: Array::ones(2),
            bias: 1f32,
            weights_gradsq: Array::ones(2),
            bias_gradsq: 1f32,
        };

        let d = focus.weights.dot(&context.weights) + 2f32 - (20f32).ln();
        assert!((d - 1.0).abs() < 1e-2);

        let upd = context.weights.map(|v| d * v);
        print!("{}", upd);
        // let mut focus = focus_ref.clone();
        // let mut context = context_ref.clone();
        let n = sgd_step(&mut focus, &mut context, 20.0, sgd_params);

        if let Some(x) = n {
            print!("{}", x);
            assert!((x - 0.500).abs() < 1e-2);
            assert_ne!(focus.weights, Array::ones(2));
            assert_ne!(focus.weights_gradsq, Array::ones(2));
            assert_ne!(context.weights, Array::ones(2));
            assert_ne!(focus.weights_gradsq, Array::ones(2));
        } else {
            assert!(n.is_some());
        }
    }

    #[test]
    fn test_glove_thread() {
        let n = 5usize;
        let model = Model::with_random_weights(n, 2);
        let params = SGDParams {
            alpha: 1.0,
            eta: 1.0,
            grad_clip_value: 100.0,
            x_max: 1.0,
        };
        let filename = "/tmp/tmp_file.test";
        {
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
        }
        let x = glove_thread(model, params, filename.to_string(), 0, n * n);
        println!("{}", x);
        assert!(x - 0.56 < std::f32::EPSILON);
        assert_eq!(0, 1);
    }
}
