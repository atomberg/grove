#[macro_use]
extern crate log;
extern crate rand;

use env_logger;
use getopts::Options;
use std::env;
use std::fmt;
use std::fs;
use std::io::{BufRead, BufReader};

use grove::glove_model::Model;
use grove::glove_train::{train_epoch, TrainParams};
use grove::word_vectors::SGDParams;

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
#[derive(Debug)]
enum WordVectorOutput {
    BothVectorsSeparatelyWithBiasTerms,
    BothVectorsSeparatelyNoBiasTerms,
    AddlVectorsExcludeBiasTerms,
}

impl fmt::Display for WordVectorOutput {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            WordVectorOutput::BothVectorsSeparatelyWithBiasTerms => write!(
                f,
                "output all data, for both word and context word vectors, including bias terms"
            ),
            WordVectorOutput::BothVectorsSeparatelyNoBiasTerms => {
                write!(f, "output word vectors, excluding bias terms")
            }
            WordVectorOutput::AddlVectorsExcludeBiasTerms => {
                write!(
                    f,
                    "output word vectors + context word vectors, excluding bias terms"
                )
            }
        }
    }
}

#[derive(Debug)]
enum SaveVectorsFormat {
    Binary,
    Text(WordVectorOutput),
    Both(WordVectorOutput),
}

impl fmt::Display for SaveVectorsFormat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            SaveVectorsFormat::Binary => write!(f, "binary"),
            SaveVectorsFormat::Text(_) => write!(f, "text"),
            SaveVectorsFormat::Both(_) => write!(f, "both"),
        }
    }
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
        let tparams = TrainParams {
            n_epochs: params.n_iter,
            n_lines,
            n_threads: params.n_threads,
            input_file: params.input_file.clone(),
        };
        let model = Model::with_random_weights(vocab_size, params.vector_size);
        model.weights_to_file("weights_0.mpk").unwrap();
        model.gradients_to_file("grads_0.mpk").unwrap();
        let model = match Model::from_file("weights_0.mpk", Some("grads_0.mpk")) {
            Ok(m) => m,
            Err(e) => panic!("{:?}", e),
        };
        train_loop(&model, tparams.clone(), params.sgd.clone());
    }
}

fn train_loop(model: &Model, train: TrainParams, sgd: SGDParams) {
    info!("Starting to train GloVe");

    // info!("Initialized the weights");
    for i in 0..train.n_epochs {
        let cost = train_epoch(model, train.clone(), sgd.clone());
        info!("iter: {:>03}, cost: {}", i + 1, cost / train.n_lines as f32);
    }
}

fn parse_args() -> Params {
    let args: Vec<String> = env::args().collect();
    let mut opts = Options::new();
    let mut params = Params {
        sgd: SGDParams {
            alpha: 0.75,
            x_max: 100.0,
            learning_rate: 0.05,
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
        &format!(
            "Dimension of word vector representations (excluding bias term); default {}",
            params.vector_size
        ),
        "<int>",
    );
    opts.optopt(
        "",
        "threads",
        &format!("Number of threads; default {}", params.n_threads),
        "<int>",
    );
    opts.optopt(
        "",
        "iter",
        &format!("Number of training iterations; default {}", params.n_iter),
        "<int>",
    );
    opts.optopt(
        "",
        "eta",
        &format!("Initial leaning rate; default {}", params.sgd.learning_rate),
        "<float>",
    );
    opts.optopt(
        "",
        "alpha",
        &format!(
            "Parameter in exponent of weighting function; default {}",
            params.sgd.alpha
        ),
        "<float>",
    );
    opts.optopt(
        "",
        "x-max",
        &format!(
            "Parameter specifying cutoff in weighting function; default {}",
            params.sgd.x_max
        ),
        "<float>",
    );
    opts.optopt(
        "",
        "binary",
        &format!(
            "Save output in binary format (0: text, 1: binary, 2: both); default {}",
            params.binary
        ),
        "<int>",
    );
    opts.optopt(
        "",
        "input-file",
        &format!(
            "Binary input file of shuffled cooccurrence data (produced by 'cooccur' and 'shuffle'); default {}",
            params.input_file
        ),
        "<file>",
    );
    opts.optopt(
        "",
        "vocab-file",
        &format!(
            "File containing vocabulary (truncated unigram counts, produced by 'vocab_count'); default {}",
            params.vocab_file
        ),
        "<file>",
    );
    opts.optopt(
        "",
        "save-file",
        &format!(
            "Filename, excluding extension, for word vector output; default {}",
            params.save_file
        ),
        "<file>",
    );
    opts.optopt(
        "",
        "gradsq-file",
        &format!(
            "Filename, excluding extension, for squared gradient output; defaut {}",
            params.gradsq_file
        ),
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
        "Checkpoint a model every <int> iterations; default 0",
        "<int>",
    );
    opts.optopt(
        "",
        "gradsq-file",
        &format!(
            "Filename, excluding extension, for squared gradient output; defaut {}",
            params.gradsq_file
        ),
        "<file>",
    );
    opts.optopt(
        "",
        "model",
        &format!(
            "Model for word vector output (for text output only); default 2
        0: {}
        1: {}
        2: {}",
            WordVectorOutput::BothVectorsSeparatelyWithBiasTerms,
            WordVectorOutput::BothVectorsSeparatelyNoBiasTerms,
            WordVectorOutput::BothVectorsSeparatelyWithBiasTerms,
        ),
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
        params.sgd.learning_rate = match matches.opt_get_default("eta", params.sgd.learning_rate) {
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
